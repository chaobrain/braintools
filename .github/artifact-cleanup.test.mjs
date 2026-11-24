import { readFileSync } from 'node:fs';
import { test } from 'node:test';
import assert from 'node:assert/strict';

const workflow = readFileSync(new URL('./workflows/artifact-cleanup.yml', import.meta.url), 'utf8').replace(/\r\n/g, '\n');
const script = workflow.split('          script: |\n')[1].split('\n').map(s => s.slice(12)).join('\n');
const runScript = new (Object.getPrototypeOf(async function(){}).constructor)('context', 'github', 'core', script);
const releaseName = JSON.parse(script.match(/const RELEASE_NAMES = (.*);/)[1])[0];
const intermediateNames = JSON.parse(script.match(/const INTERMEDIATE_NAMES = (.*);/)[1]);
const intermediatePrefixes = JSON.parse(script.match(/const INTERMEDIATE_PREFIXES = (.*);/)[1]);

async function simulate({source = {}, current = {}, artifacts, failDelete = false} = {}) {
  const run = {id: 42, name: 'Publish to PyPI', status: 'completed', conclusion: 'success',
    created_at: '2026-11-25T00:00:00Z', run_attempt: 1,
    head_repository: {full_name: 'chaobrain/example'}, ...source};
  const deleted = [], queries = [], logs = [];
  const api = {
    getWorkflowRun: async args => {queries.push(args); return {data: {...run, ...current}};},
    listWorkflowRunArtifacts: () => {},
    deleteArtifact: async args => {if (failDelete) throw new Error('Forbidden'); deleted.push(args);}
  };
  const github = {rest: {actions: api}, paginate: async (route, args) => {
    assert.equal(route, api.listWorkflowRunArtifacts);
    assert.equal(args.run_id, 42);
    assert.equal(args.per_page, 100);
    return artifacts ?? [{id: 1, name: releaseName, created_at: '2026-11-25T00:01:00Z'}];
  }};
  await runScript({repo: {owner: 'chaobrain', repo: 'example'}, payload: {workflow_run: run}}, github, {info: s => logs.push(s)});
  return {deleted, queries, logs};
}

test('successful publication deletes only its scoped distribution', async () => {
  const {deleted} = await simulate();
  assert.deepEqual(deleted, [{owner: 'chaobrain', repo: 'example', artifact_id: 1}]);
});
for (const conclusion of ['failure', 'cancelled', 'skipped', 'timed_out', null]) {
  test(`keeps ${conclusion} runs and failed publication packages`, async () => {
    const r = await simulate({source: {conclusion}});
    assert.equal(r.deleted.length, 0); assert.equal(r.queries.length, 0);
  });
}
test('keeps all existing runs, including a later retry', async () => {
  const r = await simulate({source: {created_at: '2026-11-24T04:00:00Z', run_attempt: 9}});
  assert.equal(r.deleted.length, 0); assert.equal(r.queries.length, 0);
});
test('keeps foreign or missing source repository', async () => {
  for (const head_repository of [{full_name: 'someone/fork'}, null]) {
    assert.equal((await simulate({source: {head_repository}})).deleted.length, 0);
  }
});
test('a stale completion event cannot clean a queued or newer attempt', async () => {
  for (const current of [{status: 'in_progress'}, {run_attempt: 2}, {conclusion: 'failure'}]) {
    assert.equal((await simulate({current})).deleted.length, 0);
  }
});
test('protects mappings, final reports, docs, expired and older artifacts', async () => {
  const names = ['test-map','impact-map','impact-partial','night','main-assembled','main-bisected',
    'night-assembled','night-bisected','coverage-xml','docs-html','unexpected'];
  const artifacts = names.map((name, i) => ({id: i+1, name, created_at: '2026-11-25T00:01:00Z'}));
  artifacts.push({id: 90, name: releaseName, created_at: '2026-11-24T00:00:00Z'},
    {id: 91, name: releaseName, created_at: 'invalid'},
    {id: 92, name: releaseName, created_at: '2026-11-25T00:01:00Z', expired: true});
  assert.equal((await simulate({artifacts})).deleted.length, 0);
});
test('publication cleanup is not available to a CI workflow', async () => {
  assert.equal((await simulate({source: {name: 'CI'}})).deleted.length, 0);
});
test('the CI allowlist cleans consumed inputs and preserves mappings', async () => {
  const names = [...intermediateNames, ...intermediatePrefixes.map(p => p+'example')];
  const artifacts = [...names, 'test-map', 'impact-map', 'night', releaseName].map((name,i) =>
    ({id:i+1, name, created_at:'2026-11-25T00:01:00Z'}));
  const {deleted} = await simulate({source: {name:'CI'}, artifacts});
  assert.equal(deleted.length, names.length);
});
test('paginates more than 100 artifacts', async () => {
  const artifacts = Array.from({length: 130}, (_,i) =>
    ({id:i+1, name:releaseName, created_at:'2026-11-25T00:01:00Z'}));
  assert.equal((await simulate({artifacts})).deleted.length, 130);
});
test('API errors remain visible rather than claiming cleanup succeeded', async () => {
  await assert.rejects(simulate({failDelete:true}), /Forbidden/);
});
test('privileged workflow cannot execute source code or sweep history manually', () => {
  assert.doesNotMatch(workflow, /actions\/checkout|workflow_dispatch|schedule:/);
  assert.match(workflow, /actions: write/);
  assert.match(workflow, /types: \[completed\]/);
});
