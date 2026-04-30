import functools

import awkward as ak
import jax
import jax.numpy as jnp
import numpy as np
from jax import random, lax

from hyperion.models.photon_arrival_time_nflow.net import (
    make_counts_net_fn,
    make_shape_conditioner_fn,
    traf_dist_builder,
)
from prometheus.compat.haiku_unpickler import load as haiku_load

from .utils import sources_to_model_input


def _next_bucket(n, base=2):
    if n <= 0:
        return 1
    log_cnt = np.log(n) / np.log(base)
    return int(np.power(base, np.ceil(log_cnt)))


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
        base_dist, trafo = dist_builder(traf_p)
        base_samples = base_dist.sample(seed=key, sample_shape=(n_padded,))
        return trafo.forward(base_samples)

    def generate_norm_flow_photons(
        module_coords,
        module_efficiencies,
        source_pos,
        source_dir,
        source_time,
        source_nphotons,
        seed=31337,
    ):
        if isinstance(seed, int):
            key = random.PRNGKey(seed)
        else:
            key = seed

        n_sources = source_pos.shape[0]

        src_bucket = _next_bucket(n_sources)
        if src_bucket > n_sources:
            pad = src_bucket - n_sources
            source_pos_jit = jnp.pad(source_pos, ((0, pad), (0, 0)), constant_values=1e7)
            source_dir_jit = jnp.pad(source_dir, ((0, pad), (0, 0)))
            source_time_jit = jnp.pad(source_time, ((0, pad), (0, 0)))
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

        inp_pars = inp_pars.reshape((-1, inp_pars.shape[-1]))
        time_geo = time_geo.reshape((-1, time_geo.shape[-1]))

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
        pad = masked_bucket - n_masked
        inp_params_padded = jnp.pad(inp_params_masked, ((0, pad), (0, 0)))

        ph_frac = jnp.power(
            10, counts_net.apply(counts_params, inp_params_padded)
        ).reshape(-1)[:n_masked]

        n_photons_masked = ph_frac * source_photons_masked * mod_eff_factor_masked

        key, subkey = random.split(key)
        n_photons_masked = random.poisson(subkey, n_photons_masked).astype(jnp.int32)

        # Materialise counts to CPU; used for n_max, trimming, and per-module split.
        n_ph_cpu = np.asarray(n_photons_masked)

        if not np.any(n_ph_cpu > 0):
            return ak.Array([])

        traf_params = apply_fn(shape_params, inp_params_padded)[:n_masked]

        # ── scan-based sampling ───────────────────────────────────────────────
        # n_max must be a concrete Python int so that lax.scan can trace a
        # fixed-shape body.  We compute it from the materialised CPU counts and
        # capture it as a closure variable.  XLA recompiles the body only when
        # n_max_scan changes value (i.e. when the maximum photon count crosses a
        # power-of-2 boundary), which stabilises after warmup.
        n_max_scan = _next_bucket(int(n_ph_cpu.max()))

        def body(carry, inputs):
            key = carry
            traf_p, t_geo_i = inputs
            key, subkey = random.split(key)
            raw = sample_single_pair(traf_p, n_max_scan, subkey)
            return key, raw + t_geo_i

        _, times_array = lax.scan(
            body,
            key,
            (traf_params, time_geo_masked.squeeze()),
        )

        # times_array: (n_masked, n_max_scan) — trim each row by its actual count.
        times_array_cpu = np.asarray(times_array)
        all_times = [
            times_array_cpu[i, : n_ph_cpu[i]]
            for i in range(n_masked)
            if n_ph_cpu[i] > 0
        ]

        if not all_times:
            return ak.Array([])

        # reconstruct per-module split
        distance_mask_cpu = np.asarray(distance_mask)
        n_ph_full = np.zeros(inp_pars.shape[0], dtype=np.int32)
        n_ph_full[distance_mask_cpu] = n_ph_cpu
        n_ph_per_mod = n_ph_full.reshape(module_coords.shape[0], n_sources).sum(axis=1)

        times = np.concatenate(all_times)
        times = np.split(times, np.cumsum(n_ph_per_mod)[:-1])

        return ak.Array(times)

    return generate_norm_flow_photons