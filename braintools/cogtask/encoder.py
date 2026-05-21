# Copyright 2025 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
Value-spec encoders for declarative phases.

A value spec is a function ``f(ctx, feature) -> jnp.ndarray`` of shape
``(feature.num,)`` that DeclarativePhase calls to fill the input slice for one
feature. All encoders here return such a function.

Conventions
-----------
- ``key`` is a context key (set in trial_init) holding the value to encode.
- For circular-direction encoders (``circular``, ``von_mises``), the value can be
  either a continuous angle in radians (``as_index=False``) or a discrete
  direction index that the encoder maps to ``2π * idx / n_dirs``
  (``as_index=True``). The default is ``as_index=False`` (radians) to match
  perceptual decision-making conventions. Working-memory tasks that pass
  ``sample_idx`` should set ``as_index=True``.
- ``base_value`` is the baseline added under the modulation, so the response
  is in ``[base_value, 1]`` for circular encoders.
"""

from typing import Optional, Callable, Any

import jax.numpy as jnp

from .context import Context
from .feature import Feature

__all__ = [
    'one_hot',
    'circular',
    'scalar',
    'ctx_value',
    'von_mises',
    'gaussian',
    'identity',
    'cos_sin',
]


# =============================================================================
# Value Specification Helpers
# =============================================================================
def cos_sin(
    key: str,
    num_dirs: int,
    repeats: int = 1,
    *,
    base_value: float = 0.0,
    map_to_01: bool = True,
) -> Callable[[Context, Feature], jnp.ndarray]:
    """
    Encode a discrete direction index into repeated [cos(theta), sin(theta)] features.

    Output dim: 2 * repeats
      [cos(theta)] repeated 'repeats' times, then [sin(theta)] repeated 'repeats' times.

    Parameters
    ----------
    key : str
        Context key containing discrete direction index (int).
    num_dirs : int
        Number of discrete directions (e.g., num_stimuli).
    repeats : int
        How many duplicated cos features and sin features.
    base_value : float
        Baseline after mapping to [0,1]. Ignored if map_to_01=False.
    map_to_01 : bool
        If True, map cos/sin from [-1,1] to [0,1], then to [base_value, 1].
        If False, keep raw cos/sin in [-1,1].

    Notes
    -----
    - This encoder ignores feature.num (it will validate it equals 2*repeats).
    """

    if num_dirs <= 0:
        raise ValueError(f"num_dirs must be positive, got {num_dirs}")
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")

    def encode(ctx: Context, feature: Feature) -> jnp.ndarray:
        expected = 2 * repeats
        if feature.num != expected:
            raise ValueError(
                f"cos_sin('{key}') expects feature.num == {expected}, got {feature.num}"
            )

        idx = ctx.get(key, 0)
        # idx may be JAX scalar; keep it as jnp
        idx = jnp.asarray(idx)

        theta = (2.0 * jnp.pi) * (idx / float(num_dirs))
        c = jnp.cos(theta)
        s = jnp.sin(theta)

        if map_to_01:
            # [-1,1] -> [0,1]
            c = 0.5 * (c + 1.0)
            s = 0.5 * (s + 1.0)
            # [0,1] -> [base_value, 1]
            c = c * (1.0 - base_value) + base_value
            s = s * (1.0 - base_value) + base_value

        cos_block = jnp.full((repeats,), c, dtype=jnp.float32)
        sin_block = jnp.full((repeats,), s, dtype=jnp.float32)
        return jnp.concatenate([cos_block, sin_block], axis=0)

    encode.__name__ = f"cos_sin('{key}', num_dirs={num_dirs}, repeats={repeats})"
    return encode

def one_hot(
    key: str,
    num_classes: Optional[int] = None,
    active_value: float = 1.0
) -> Callable[[Context, Feature], jnp.ndarray]:
    """
    Create a one-hot encoding value specification.

    Parameters
    ----------
    key : str
        Context key containing the class index.
    num_classes : int, optional
        Number of classes. If None, inferred from feature.num.
    active_value : float
        Value for the active class (default 1.0).

    Returns
    -------
    Callable
        Encoder function (ctx, feature) -> np.ndarray

    Examples
    --------
    >>> inputs={'stimulus': one_hot('sample_idx')}
    >>> inputs={'stimulus': one_hot('sample_idx', num_classes=8)}
    """

    def encode(ctx: Context, feature: Feature) -> jnp.ndarray:
        idx = ctx.get(key, 0)
        n = num_classes if num_classes is not None else feature.num
        result = jnp.zeros(n)
        result = result.at[idx].set(active_value)
        return result

    encode.__name__ = f"one_hot('{key}')"
    return encode


def circular(
    key: str,
    coherence_key: Optional[str] = None,
    base_value: float = 0.5,
    max_coherence: float = 100.0,
    as_index: bool = False,
    num_dirs: Optional[int] = None,
) -> Callable[[Context, Feature], jnp.ndarray]:
    """
    Cosine-tuned directional encoder.

    The output of unit i (for i in [0, feature.num)) is
    ``base_value + (coh / (2 * max_coherence)) * cos(pref_i - direction)``,
    where preferred directions ``pref_i`` are uniformly spaced on the circle.

    Parameters
    ----------
    key : str
        Context key holding the direction. By default this is an angle in
        radians; if ``as_index=True``, it is interpreted as an integer index in
        ``[0, num_dirs)`` and converted to ``2π * idx / num_dirs``.
    coherence_key : str, optional
        Context key with a coherence in ``[0, max_coherence]``. If ``None`` the
        encoder uses ``max_coherence`` (full strength).
    base_value : float
        Additive baseline (default ``0.5``).
    max_coherence : float
        Coherence normalization constant (default ``100``).
    as_index : bool
        Interpret ``ctx[key]`` as an integer index rather than radians.
    num_dirs : int, optional
        Number of discrete directions when ``as_index=True``. Defaults to
        ``feature.num``.

    Examples
    --------
    >>> inputs={'motion': circular('direction', 'coherence')}
    >>> inputs={'stimulus': circular('sample_idx', as_index=True, num_dirs=8)}
    """

    def encode(ctx: Context, feature: Feature) -> jnp.ndarray:
        raw = ctx.get(key, 0.0)
        n = feature.num
        if as_index:
            n_dirs = num_dirs if num_dirs is not None else n
            direction = (2.0 * jnp.pi) * (jnp.asarray(raw, dtype=jnp.float32) / float(n_dirs))
        else:
            direction = jnp.asarray(raw, dtype=jnp.float32)

        coherence = ctx.get(coherence_key, max_coherence) if coherence_key else max_coherence
        coherence = jnp.asarray(coherence, dtype=jnp.float32)
        pref_dirs = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        response = jnp.cos(pref_dirs - direction)
        response = response * (coherence / (2.0 * max_coherence)) + base_value
        return response

    encode.__name__ = f"circular('{key}'{', as_index' if as_index else ''})"
    return encode


def scalar(
    key: str,
    scale: float = 1.0,
    offset: float = 0.0
) -> Callable[[Context, Feature], jnp.ndarray]:
    """
    Create a scalar encoding that broadcasts to all feature dimensions.

    Parameters
    ----------
    key : str
        Context key containing the scalar value.
    scale : float
        Multiplicative scale (default 1.0).
    offset : float
        Additive offset (default 0.0).

    Returns
    -------
    Callable
        Encoder function (ctx, feature) -> np.ndarray

    Examples
    --------
    >>> inputs={'intensity': scalar('stimulus_value')}
    >>> inputs={'intensity': scalar('stimulus_value', scale=0.5, offset=0.25)}
    """

    def encode(ctx: Context, feature: Feature) -> jnp.ndarray:
        value = ctx.get(key, 0.0)
        return jnp.full(feature.num, value * scale + offset)

    encode.__name__ = f"scalar('{key}')"
    return encode


def ctx_value(
    key: str,
    default: Any = 0
) -> Callable[[Context, Feature], Any]:
    """
    Create a dynamic value that reads directly from context.

    Parameters
    ----------
    key : str
        Context key to read.
    default : Any
        Default value if key not found (default 0).

    Returns
    -------
    Callable
        Value function (ctx, feature) -> value

    Examples
    --------
    >>> inputs={'fixation': ctx_value('fixation_strength', default=1.0)}
    """

    def get_value(ctx: Context, feature: Feature) -> Any:
        return ctx.get(key, default)

    get_value.__name__ = f"ctx_value('{key}')"
    return get_value


def von_mises(
    key: str,
    coherence_key: Optional[str] = None,
    kappa: float = 2.0,
    base_value: float = 0.5,
    max_coherence: float = 100.0,
    as_index: bool = True,
    num_dirs: Optional[int] = None,
) -> Callable[[Context, Feature], jnp.ndarray]:
    """
    Von Mises (circular-normal) directional encoder.

    Output is a normalized von Mises tuning curve in ``[base_value, 1]``.

    Parameters
    ----------
    key : str
        Context key holding the direction. When ``as_index=True`` (default) the
        value is an integer index in ``[0, num_dirs)`` and is converted to
        ``2π * idx / num_dirs``. When ``as_index=False`` it is an angle in
        radians (matching ``circular``).
    coherence_key : str, optional
        Context key with a coherence in ``[0, max_coherence]``. If ``None`` the
        encoder uses ``max_coherence`` (full strength).
    kappa : float
        Concentration parameter; higher values give sharper tuning.
    base_value : float
        Floor of the response after normalization.
    max_coherence : float
        Coherence normalization constant.
    as_index : bool
        Interpret ``ctx[key]`` as an integer index rather than radians. The
        default matches the working-memory tasks which pass ``sample_idx``.
    num_dirs : int, optional
        Number of discrete directions when ``as_index=True``. Defaults to
        ``feature.num``.

    Examples
    --------
    >>> inputs={'motion': von_mises('direction', 'coherence', as_index=False)}
    >>> inputs={'stimulus': von_mises('sample_idx', num_dirs=8)}
    """

    def encode(ctx: Context, feature: Feature) -> jnp.ndarray:
        raw = ctx.get(key, 0)
        n = feature.num
        if as_index:
            n_dirs = num_dirs if num_dirs is not None else n
            direction = (2.0 * jnp.pi) * (jnp.asarray(raw, dtype=jnp.float32) / float(n_dirs))
        else:
            direction = jnp.asarray(raw, dtype=jnp.float32)

        coherence = ctx.get(coherence_key, max_coherence) if coherence_key else max_coherence
        coherence = jnp.asarray(coherence, dtype=jnp.float32)
        pref_dirs = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)

        effective_kappa = kappa * (coherence / max_coherence)
        # Numerically stable normalization: shift exponent by -effective_kappa.
        shifted = effective_kappa * (jnp.cos(pref_dirs - direction) - 1.0)
        raw_resp = jnp.exp(shifted)
        denom = 1.0 - jnp.exp(-2.0 * effective_kappa)
        # Guard against effective_kappa == 0 (uniform response).
        response = jnp.where(denom > 1e-12, (raw_resp - jnp.exp(-2.0 * effective_kappa)) / denom, jnp.ones_like(raw_resp))
        response = response * (1.0 - base_value) + base_value
        return response

    encode.__name__ = f"von_mises('{key}'{', as_index' if as_index else ''})"
    return encode


def gaussian(
    key: str,
    centers: Optional[jnp.ndarray] = None,
    sigma: float = 0.1,
    base_value: float = 0.0
) -> Callable[[Context, Feature], jnp.ndarray]:
    """
    Create a Gaussian bump encoding value specification.

    Encodes a scalar value using Gaussian tuning curves centered
    at different preferred values.

    Parameters
    ----------
    key : str
        Context key containing the scalar value to encode.
    centers : np.ndarray, optional
        Array of preferred values for each neuron.
        If None, centers are evenly spaced in [0, 1].
    sigma : float
        Width of Gaussian tuning curves (standard deviation).
        Default 0.1.
    base_value : float
        Baseline firing rate (default 0.0).

    Returns
    -------
    Callable
        Encoder function (ctx, feature) -> np.ndarray

    Examples
    --------
    >>> inputs={'value': gaussian('stimulus_value', sigma=0.15)}
    >>> # Custom centers
    >>> centers = np.array([0.2, 0.4, 0.6, 0.8])
    >>> inputs={'value': gaussian('stimulus_value', centers=centers, sigma=0.1)}
    """

    def encode(ctx: Context, feature: Feature) -> jnp.ndarray:
        value = ctx.get(key, 0.0)

        # Determine centers
        if centers is None:
            # Evenly spaced centers in [0, 1]
            n = feature.num
            center_vals = jnp.linspace(0, 1, n)
        else:
            center_vals = jnp.asarray(centers)
            if len(center_vals) != feature.num:
                raise ValueError(
                    f"Number of centers ({len(center_vals)}) must match "
                    f"feature dimensions ({feature.num})"
                )

        # Gaussian tuning: exp(-0.5 * ((x - mu) / sigma)^2)
        response = jnp.exp(-0.5 * ((value - center_vals) / sigma) ** 2)

        # Add baseline
        response = response * (1.0 - base_value) + base_value

        return response

    encode.__name__ = f"gaussian('{key}')"
    return encode


def identity(
    key: str,
    default: Optional[jnp.ndarray] = None
) -> Callable[[Context, Feature], jnp.ndarray]:
    """
    Create an identity encoding that passes through values directly.

    Useful when the context already contains properly formatted arrays
    that should be used directly without transformation.

    Parameters
    ----------
    key : str
        Context key containing the array value.
    default : np.ndarray, optional
        Default value if key not found. If None, returns zeros.

    Returns
    -------
    Callable
        Encoder function (ctx, feature) -> np.ndarray

    Examples
    --------
    >>> # Context contains pre-computed encoding
    >>> ctx['custom_encoding'] = np.array([0.1, 0.5, 0.8, 0.3])
    >>> inputs={'stimulus': identity('custom_encoding')}
    """

    def encode(ctx: Context, feature: Feature) -> jnp.ndarray:
        value = ctx.get(key, default)
        if value is None:
            return jnp.zeros(feature.num)

        value = jnp.asarray(value)

        # Validate shape
        if value.shape != (feature.num,):
            raise ValueError(
                f"Identity encoding expects array of shape ({feature.num},), "
                f"got {value.shape} for key '{key}'"
            )

        return value

    encode.__name__ = f"identity('{key}')"
    return encode
