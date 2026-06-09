# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for the value-spec encoders in ``braintools.cogtask.encoder``.

These exercise the per-feature ``(ctx, feature) -> jnp.ndarray`` contract:
shape, default-value behavior when the context key is missing, and the
math-specific properties of each encoder (one-hot peak, cosine tuning peaks
at the preferred direction, coherence scaling, etc.). The tests deliberately
prefer concrete numeric assertions over loose smoke checks.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from braintools.cogtask.context import Context
from braintools.cogtask.encoder import (
    one_hot, circular, scalar, ctx_value, von_mises, gaussian, identity, cos_sin,
)
from braintools.cogtask.feature import Feature


def _ctx(**state):
    ctx = Context()
    for k, v in state.items():
        ctx[k] = v
    return ctx


# ---------------------------------------------------------------------------
# one_hot
# ---------------------------------------------------------------------------

def test_one_hot_active_index_only():
    enc = one_hot('cls')
    out = np.asarray(enc(_ctx(cls=2), Feature(5, 'x')))
    assert out.shape == (5,)
    expected = np.zeros(5)
    expected[2] = 1.0
    np.testing.assert_allclose(out, expected)


def test_one_hot_custom_active_value_and_num_classes():
    enc = one_hot('cls', num_classes=3, active_value=0.5)
    out = np.asarray(enc(_ctx(cls=1), Feature(3, 'x')))
    assert out.shape == (3,)
    np.testing.assert_allclose(out, [0.0, 0.5, 0.0])


def test_one_hot_missing_key_defaults_to_zero_index():
    enc = one_hot('cls')
    out = np.asarray(enc(_ctx(), Feature(4, 'x')))
    np.testing.assert_allclose(out, [1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# circular
# ---------------------------------------------------------------------------

def test_circular_peaks_at_preferred_direction_with_max_coherence():
    enc = circular('direction', 'coherence', base_value=0.5, max_coherence=100.0)
    # 8 preferred directions evenly on the circle; direction=0 → unit 0 peak.
    out = np.asarray(enc(_ctx(direction=0.0, coherence=100.0), Feature(8, 'x')))
    assert out.shape == (8,)
    assert np.argmax(out) == 0
    # With coherence=100 and max=100 the modulation is cos(...) / 2 + 0.5.
    expected = jnp.cos(jnp.linspace(0, 2 * jnp.pi, 8, endpoint=False)) / 2.0 + 0.5
    np.testing.assert_allclose(out, np.asarray(expected), atol=1e-5)


def test_circular_returns_baseline_when_coherence_is_zero():
    enc = circular('direction', 'coherence', base_value=0.5, max_coherence=100.0)
    out = np.asarray(enc(_ctx(direction=jnp.pi, coherence=0.0), Feature(8, 'x')))
    np.testing.assert_allclose(out, np.full(8, 0.5), atol=1e-6)


def test_circular_uses_max_coherence_when_coherence_key_is_none():
    enc = circular('direction', None, base_value=0.0, max_coherence=10.0)
    out = np.asarray(enc(_ctx(direction=0.0), Feature(4, 'x')))
    # Equivalent to circular with coherence==max_coherence.
    enc_max = circular('direction', 'coh', base_value=0.0, max_coherence=10.0)
    expected = np.asarray(enc_max(_ctx(direction=0.0, coh=10.0), Feature(4, 'x')))
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_circular_as_index_maps_idx_to_radians():
    enc = circular('idx', None, as_index=True, num_dirs=4, base_value=0.0, max_coherence=1.0)
    out0 = np.asarray(enc(_ctx(idx=0), Feature(4, 'x')))
    out2 = np.asarray(enc(_ctx(idx=2), Feature(4, 'x')))
    # idx=0 → direction=0, peak at unit 0. idx=2 → π, peak at unit 2 (which is π).
    assert np.argmax(out0) == 0
    assert np.argmax(out2) == 2


# ---------------------------------------------------------------------------
# scalar / ctx_value
# ---------------------------------------------------------------------------

def test_scalar_broadcasts_value_with_scale_and_offset():
    enc = scalar('v', scale=2.0, offset=1.0)
    out = np.asarray(enc(_ctx(v=3.0), Feature(4, 'x')))
    np.testing.assert_allclose(out, np.full(4, 7.0))


def test_scalar_missing_key_uses_zero():
    enc = scalar('v')
    out = np.asarray(enc(_ctx(), Feature(3, 'x')))
    np.testing.assert_allclose(out, np.zeros(3))


def test_ctx_value_passthrough_and_default():
    enc = ctx_value('payload', default=42)
    assert enc(_ctx(payload=7), Feature(1, 'x')) == 7
    assert enc(_ctx(), Feature(1, 'x')) == 42


# ---------------------------------------------------------------------------
# von_mises
# ---------------------------------------------------------------------------

def test_von_mises_peaks_at_target_index_and_baseline_at_opposite_index():
    enc = von_mises('idx', None, kappa=4.0, base_value=0.1, as_index=True, num_dirs=8)
    out = np.asarray(enc(_ctx(idx=0), Feature(8, 'x')))
    assert out.shape == (8,)
    # Maximum is at the same direction as the target.
    assert np.argmax(out) == 0
    # The opposite preferred direction (4 of 8) is the minimum and equals base_value.
    assert np.argmin(out) == 4
    assert out[4] == pytest.approx(0.1, abs=1e-5)


def test_von_mises_returns_flat_baseline_when_coherence_is_zero():
    enc = von_mises('idx', 'coh', kappa=4.0, base_value=0.2, as_index=True, num_dirs=8)
    out = np.asarray(enc(_ctx(idx=3, coh=0.0), Feature(8, 'x')))
    np.testing.assert_allclose(out, np.ones(8), atol=1e-5)


def test_von_mises_radian_mode_matches_index_mode_at_aligned_inputs():
    n_dirs = 8
    idx_enc = von_mises('a', None, kappa=2.0, base_value=0.0, as_index=True, num_dirs=n_dirs)
    rad_enc = von_mises('a', None, kappa=2.0, base_value=0.0, as_index=False)
    by_index = np.asarray(idx_enc(_ctx(a=3), Feature(n_dirs, 'x')))
    by_radians = np.asarray(rad_enc(_ctx(a=2 * jnp.pi * 3 / n_dirs), Feature(n_dirs, 'x')))
    np.testing.assert_allclose(by_index, by_radians, atol=1e-5)


# ---------------------------------------------------------------------------
# gaussian
# ---------------------------------------------------------------------------

def test_gaussian_peaks_at_value_with_evenly_spaced_centers():
    enc = gaussian('v', sigma=0.1)
    out = np.asarray(enc(_ctx(v=0.5), Feature(5, 'x')))
    # centers = linspace(0, 1, 5) = [0, .25, .5, .75, 1]; peak at idx 2.
    assert np.argmax(out) == 2
    assert out[2] == pytest.approx(1.0, abs=1e-6)


def test_gaussian_custom_centers_must_match_feature_dim():
    enc = gaussian('v', centers=jnp.array([0.0, 1.0, 2.0]), sigma=0.5)
    with pytest.raises(ValueError, match="centers"):
        enc(_ctx(v=1.0), Feature(4, 'x'))


def test_gaussian_baseline_floor():
    enc = gaussian('v', sigma=0.1, base_value=0.3)
    out = np.asarray(enc(_ctx(v=100.0), Feature(5, 'x')))
    # Value far outside any center → response ≈ 0 before baseline, ≈ base_value after.
    np.testing.assert_allclose(out, np.full(5, 0.3), atol=1e-5)


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------

def test_identity_passes_array_through():
    enc = identity('payload')
    arr = jnp.asarray([0.1, 0.2, 0.3])
    out = np.asarray(enc(_ctx(payload=arr), Feature(3, 'x')))
    np.testing.assert_allclose(out, [0.1, 0.2, 0.3])


def test_identity_default_zeros_when_missing():
    enc = identity('payload')
    out = np.asarray(enc(_ctx(), Feature(3, 'x')))
    np.testing.assert_allclose(out, np.zeros(3))


def test_identity_rejects_shape_mismatch():
    enc = identity('payload')
    with pytest.raises(ValueError, match="shape"):
        enc(_ctx(payload=jnp.zeros(4)), Feature(3, 'x'))


def test_identity_broadcasts_scalar_to_feature_num():
    """A 0-d scalar is broadcast to (feature.num,). This is what makes
    DelayComparison(value_encoding='identity') work: the trial stores a scalar
    sample/test value while the stimulus feature has num == 1."""
    enc = identity('payload')
    out1 = np.asarray(enc(_ctx(payload=0.7), Feature(1, 'x')))
    np.testing.assert_allclose(out1, [0.7])
    out3 = np.asarray(enc(_ctx(payload=0.7), Feature(3, 'x')))
    np.testing.assert_allclose(out3, [0.7, 0.7, 0.7])


# ---------------------------------------------------------------------------
# cos_sin
# ---------------------------------------------------------------------------

def test_cos_sin_default_is_cos_then_sin_repeated():
    enc = cos_sin('idx', num_dirs=4, repeats=2, map_to_01=False)
    out = np.asarray(enc(_ctx(idx=0), Feature(4, 'x')))
    # idx=0 → theta=0 → cos=1, sin=0. Block layout: [cos, cos, sin, sin].
    np.testing.assert_allclose(out, [1.0, 1.0, 0.0, 0.0], atol=1e-6)


def test_cos_sin_idx_at_quarter_circle():
    enc = cos_sin('idx', num_dirs=4, repeats=1, map_to_01=False)
    out = np.asarray(enc(_ctx(idx=1), Feature(2, 'x')))
    # theta = π/2 → cos ≈ 0, sin = 1.
    np.testing.assert_allclose(out, [0.0, 1.0], atol=1e-6)


def test_cos_sin_map_to_01_shifts_to_baseline():
    enc = cos_sin('idx', num_dirs=4, repeats=1, base_value=0.2, map_to_01=True)
    out = np.asarray(enc(_ctx(idx=0), Feature(2, 'x')))
    # cos=1, sin=0; mapped [-1,1]→[0,1]→[0.2,1].
    # cos=1 stays at 1.0; sin=0 maps to 0.5 then scales to 0.5*0.8+0.2=0.6.
    np.testing.assert_allclose(out, [1.0, 0.6], atol=1e-5)


def test_cos_sin_validates_feature_dim_against_repeats():
    enc = cos_sin('idx', num_dirs=4, repeats=3, map_to_01=False)
    with pytest.raises(ValueError, match="feature.num"):
        enc(_ctx(idx=0), Feature(4, 'x'))


def test_cos_sin_rejects_non_positive_arguments():
    with pytest.raises(ValueError):
        cos_sin('idx', num_dirs=0, repeats=1)
    with pytest.raises(ValueError):
        cos_sin('idx', num_dirs=4, repeats=0)
