"""Vectorised-sampling variant of norm_flow_photons.

Identical to ``norm_flow_photons`` except that the per-pair arrival-time
sampling loop is replaced by a single ``jax.vmap`` call over all pairs whose
photon count is at or below ``BATCH_CAP``.  Pairs above the cap fall back to
the sequential ``sample_single_pair`` path so peak memory stays bounded.

Only ``make_generate_norm_flow_photons`` is modified; the likelihood helpers
are untouched and re-exported from the original module.
"""
import functools
import pickle

import awkward as ak
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

from hyperion.models.photon_arrival_time_nflow.net import (
    eval_log_prob,
    make_counts_net_fn,
    make_shape_conditioner_fn,
    sample_shape_model,
    traf_dist_builder,
)
from prometheus.compat.haiku_unpickler import load as haiku_load

from .utils import sources_to_model_input, sources_to_model_input_per_module

# Re-export unchanged likelihood helpers so callers can swap the import
# without touching anything else.
from .norm_flow_photons import (  # noqa: F401
    make_nflow_photon_likelihood,
    make_nflow_photon_likelihood_per_module,
)

# Pairs whose Poisson-sampled photon count exceeds this threshold are handled
# by the sequential fallback (sample_single_pair) to keep peak memory bounded
# in extreme high-energy events.  The vast majority of pairs are well below
# this limit, so the vectorised path handles almost all work.
BATCH_CAP: int = 1024


def _next_bucket(n, base=2):
    """Return the smallest power of *base* that is >= *n*.

    Parameters
    ----------
    n : int
        Count to bucket.
    base : int, optional
        Bucketing base. Default is 2.

    Returns
    -------
    int
        Smallest ``base ** k`` satisfying ``base ** k >= n``.
    """
    if n <= 0:
        return 1
    log_cnt = np.log(n) / np.log(base)
    return int(np.power(base, np.ceil(log_cnt)))


# @profile
def make_generate_norm_flow_photons(shape_model_path, counts_model_path, c_medium):
    shape_config, shape_params = haiku_load(shape_model_path)
    counts_config, counts_params = haiku_load(counts_model_path)

    shape_conditioner = make_shape_conditioner_fn(
        shape_config["mlp_hidden_size"],
        shape_config["mlp_num_layers"],
        shape_config["flow_num_bins"],
        shape_config["flow_num_layers"],
    )

    @jax.jit
    def apply_fn(params, x):
        return shape_conditioner.apply(params, x)

    dist_builder = traf_dist_builder(
        shape_config["flow_num_layers"],
        (shape_config["flow_rmin"], shape_config["flow_rmax"]),
        return_base=True,
    )

    counts_net = make_counts_net_fn(counts_config)

    @functools.partial(jax.jit, static_argnums=(1,))
    def sample_single_pair(traf_p, n_padded, key):
        """Sequential fallback: sample *n_padded* times for one pair.

        Used only for rare pairs whose photon count exceeds ``BATCH_CAP``.
        Identical to the same function in ``norm_flow_photons``.

        Parameters
        ----------
        traf_p : jnp.ndarray
            Flow transformation parameters, shape ``(n_flow_params,)``.
        n_padded : int
            Number of samples (static, equals ``_next_bucket(n_actual)``).
        key : jax.random.PRNGKey
            JAX PRNG key.

        Returns
        -------
        jnp.ndarray
            Sampled arrival times, shape ``(n_padded,)``.
        """
        base_dist, trafo = dist_builder(traf_p)
        base_samples = base_dist.sample(seed=key, sample_shape=(n_padded,))
        return trafo.forward(base_samples)

    @functools.partial(jax.jit, static_argnums=(1,))
    def sample_batch(traf_params, n_max, keys):
        """Vectorised sampling: draw *n_max* arrival times for every pair.

        A single ``jax.vmap`` call replaces the per-pair Python loop for the
        common case where all photon counts are at most ``n_max``.  Each pair
        receives its own PRNG key from the pre-split ``keys`` array so
        results remain statistically independent.

        Parameters
        ----------
        traf_params : jnp.ndarray
            Flow parameters, shape ``(batch, n_flow_params)``.
        n_max : int
            Samples per pair (static); equals the bucket of the batch maximum.
        keys : jnp.ndarray
            PRNGKeys, shape ``(batch, 2)``.

        Returns
        -------
        jnp.ndarray
            Sampled times, shape ``(batch, n_max)``.
        """
        def _one(p, k):
            base_dist, trafo = dist_builder(p)
            return trafo.forward(base_dist.sample(seed=k, sample_shape=(n_max,)))

        return jax.vmap(_one)(traf_params, keys)

    def generate_norm_flow_photons(
        module_coords,
        module_efficiencies,
        source_pos,
        source_dir,
        source_time,
        source_nphotons,
        seed=31337,
    ):
        """Generate photon arrival times using the vectorised sampling path.

        Identical contract to ``norm_flow_photons.generate_norm_flow_photons``.
        The only behavioural difference is in how PRNG keys are consumed for
        arrival-time sampling: all keys are split in a single JAX call before
        the loop, and pairs below ``BATCH_CAP`` are sampled in one batched
        ``jax.vmap`` kernel rather than individual sequential calls.

        This means photon *counts* per module are bit-for-bit identical to the
        reference implementation (Poisson sampling is unchanged), while arrival
        times are drawn from the same conditional distribution but with
        different PRNG keys.

        Parameters
        ----------
        module_coords : jnp.ndarray
            Detector module positions, shape ``(n_modules, 3)``.
        module_efficiencies : jnp.ndarray
            Per-module quantum efficiency factors, shape ``(n_modules,)``.
        source_pos : jnp.ndarray
            Photon source positions, shape ``(n_sources, 3)``.
        source_dir : jnp.ndarray
            Photon source directions, shape ``(n_sources, 3)``.
        source_time : jnp.ndarray
            Photon source emission times, shape ``(n_sources, 1)``.
        source_nphotons : jnp.ndarray
            Number of photons emitted per source, shape ``(n_sources, 1)``.
        seed : int or jax.random.PRNGKey, optional
            Random seed or JAX PRNG key. Default is 31337.

        Returns
        -------
        ak.Array
            Ragged array of detected photon arrival times, one inner list per
            module, length ``n_modules``.
        """
        if isinstance(seed, int):
            key = random.PRNGKey(seed)
        else:
            key = seed

        n_sources = source_pos.shape[0]

        src_bucket = _next_bucket(n_sources)
        if src_bucket > n_sources:
            src_pad = src_bucket - n_sources
            source_pos_jit = jnp.pad(
                source_pos, ((0, src_pad), (0, 0)), constant_values=1e7
            )
            source_dir_jit = jnp.pad(source_dir, ((0, src_pad), (0, 0)))
            source_time_jit = jnp.pad(source_time, ((0, src_pad), (0, 0)))
        else:
            source_pos_jit = source_pos
            source_dir_jit = source_dir
            source_time_jit = source_time

        inp_pars, time_geo = sources_to_model_input(
            module_coords,
            source_pos_jit,
            source_dir_jit,
            source_time_jit,
            c_medium,
        )

        inp_pars = inp_pars[:n_sources]
        time_geo = time_geo[:n_sources]

        inp_pars = jnp.swapaxes(inp_pars, 0, 1)
        time_geo = jnp.swapaxes(time_geo, 0, 1)

        inp_pars = inp_pars.reshape(
            (n_sources * module_coords.shape[0], inp_pars.shape[-1])
        )
        time_geo = time_geo.reshape(
            (n_sources * module_coords.shape[0], time_geo.shape[-1])
        )
        source_photons = jnp.tile(source_nphotons, module_coords.shape[0]).T.ravel()
        mod_eff_factor = jnp.repeat(module_efficiencies, n_sources)

        distance_mask = inp_pars[:, 0] < np.log10(300)

        inp_params_masked = inp_pars[distance_mask]
        time_geo_masked = time_geo[distance_mask]
        source_photons_masked = source_photons[distance_mask]
        mod_eff_factor_masked = mod_eff_factor[distance_mask]

        n_masked = inp_params_masked.shape[0]
        if n_masked == 0:
            return ak.Array([])

        masked_bucket = _next_bucket(n_masked)
        masked_pad = masked_bucket - n_masked
        inp_params_padded = jnp.pad(inp_params_masked, ((0, masked_pad), (0, 0)))

        ph_frac = jnp.power(
            10, counts_net.apply(counts_params, inp_params_padded)
        ).reshape(-1)[:n_masked]

        n_photons_masked = ph_frac * source_photons_masked * mod_eff_factor_masked

        key, subkey = random.split(key)
        n_photons_masked = (
            random.poisson(subkey, n_photons_masked, shape=n_photons_masked.shape)
            .squeeze()
            .astype(jnp.int32)
        )

        # Materialise counts to CPU once; use for early exit and loop control.
        n_ph_cpu = np.asarray(n_photons_masked)

        if not np.any(n_ph_cpu > 0):
            return ak.Array([])

        traf_params = apply_fn(shape_params, inp_params_padded)[:n_masked]

        # Reconstruct per-module photon counts on CPU to avoid a JAX scatter.
        distance_mask_cpu = np.asarray(distance_mask)
        n_photons_full = np.zeros(n_sources * module_coords.shape[0], dtype=np.int32)
        n_photons_full[distance_mask_cpu] = n_ph_cpu
        n_ph_per_mod = n_photons_full.reshape(module_coords.shape[0], n_sources).sum(axis=1)

        t_geo_cpu = np.atleast_1d(np.asarray(time_geo_masked.squeeze()))

        # ── Vectorised sampling ───────────────────────────────────────────────
        # Generate all PRNG keys in one JAX call (n_masked device ops → 1).
        subkeys = random.split(key, n_masked)

        pair_times: list = [None] * n_masked

        # Low-count pairs (≤ BATCH_CAP): one vmapped kernel call.
        low_idx = np.where((n_ph_cpu > 0) & (n_ph_cpu <= BATCH_CAP))[0]
        if len(low_idx) > 0:
            n_max = int(_next_bucket(int(n_ph_cpu[low_idx].max())))
            batch_raw = np.asarray(
                sample_batch(traf_params[low_idx], n_max, subkeys[low_idx])
            )
            for j, i in enumerate(low_idx):
                n_i = int(n_ph_cpu[i])
                pair_times[i] = batch_raw[j, :n_i] + t_geo_cpu[i]

        # High-count pairs (> BATCH_CAP): sequential fallback to cap memory.
        for i in np.where(n_ph_cpu > BATCH_CAP)[0]:
            n_i = int(n_ph_cpu[i])
            raw = sample_single_pair(traf_params[i], _next_bucket(n_i), subkeys[i])
            pair_times[i] = np.asarray(raw[:n_i]) + t_geo_cpu[i]

        all_pair_times = [t for t in pair_times if t is not None]

        times = np.atleast_1d(
            np.concatenate(all_pair_times) if all_pair_times else np.array([])
        )
        times = np.split(times, np.cumsum(n_ph_per_mod)[:-1])
        return ak.Array(times)

    return generate_norm_flow_photons
