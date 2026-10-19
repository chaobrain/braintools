# Benchmarks

Scripts that time host-side work in `braintools`. They are plain Python
programs (no pytest needed) and always import the `braintools` of the
checkout they live in.

## `conn_build.py`: building large networks with `braintools.conn`

```bash
python benchmarks/conn_build.py            # full sizes, N = 1e5
python benchmarks/conn_build.py --quick    # N = 1e4 smoke run
python benchmarks/conn_build.py --case c --n 30000   # one case, custom size
```

| Case | Size | Connectivity | Edges |
|---|---|---|---|
| a | N | `FixedProb(0.001)` | about 1e-3 N^2 |
| b | N | `FixedInDegree(100)`, exported with `to_fixed_num_per_post()` | 100 N |
| c | N neurons uniform on a 2-D sheet (10 um mean spacing) | `Gaussian(GaussianProfile(sigma=40 um, max_distance=120 um))` | about 100 N |
| d | N / 10 neurons (80 % E, 20 % I) | E/I `Network`, `FixedInDegree(80)` from E and `FixedInDegree(20)` from I, exported with `Network.to_csr()` | 10 N |

Cases b and d are reported as `skipped:...` when `braintools.conn` does not
provide `FixedInDegree` / `Population` / `Projection` / `Network`.

Each case runs in its own Python process (`--inline` runs in the current
one), so its peak resident set size is its own. The script prints one line per
case,

```text
RESULT case=<c> n=<N> edges=<E> time_s=<t> peak_rss_mb=<m> status=<ok|skipped:...>
```

then a Markdown table. `time_s` covers the build (and the export, where the
case has one), not interpreter start-up or imports. `peak_rss_mb` is the
process's `ru_maxrss`, the number `/usr/bin/time -f %M` reports; it includes
about 330 MB for importing JAX, brainunit and braintools.
`benchmarks/conn_build_test.py` runs `--quick` and checks this format.

### Results

Measured on a 16-core WSL2 machine (23 GB RAM), NumPy 2.4, SciPy 1.17.

Before (0.6, dense `N_pre x N_post` intermediates) and after (0.7) at N = 1e4:

| Case | 0.6 time (s) | 0.6 peak RSS (MB) | 0.7 time (s) | 0.7 peak RSS (MB) |
|---|---|---|---|---|
| a `FixedProb(0.001)` | 1.17 | 1188 | 0.013 | 335 |
| c 2-D sheet `Gaussian` with `max_distance` | 3.94 | 3051 | 0.50 | 766 |

At N = 1e5, 0.6 needs about 100 times the N = 1e4 working memory for these
cases (dense per-pair arrays: roughly 90 GB for case a and 250 GB for case c),
so it cannot build them. 0.7 at N = 1e5:

| Case | Edges | Time (s) | Peak RSS (MB) |
|---|---|---|---|
| a | 9 999 978 | 2.3 | 722 |
| b | 10 000 000 | 3.3 | 849 |
| c | 9 746 768 | 4.8 | 1105 |
| d | 1 000 000 | 3.0 | 635 |

Every case stays well under the 4 GB peak-RSS budget. Timings vary with
machine load.
