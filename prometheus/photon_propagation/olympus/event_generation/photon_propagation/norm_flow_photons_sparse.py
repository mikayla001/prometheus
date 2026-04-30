"""Sparse-geometry variant of norm_flow_photons.

Uses scipy.spatial.cKDTree to pre-filter source-module pairs to those within
the 300 m flow-model cutoff before any JAX computation.  Geometry (distance,
angle, geometric time) is computed in NumPy for only the surviving pairs, so
the dense O(n_sources × n_modules) intermediate tensor that the reference
implementation materialises via JAX vmap is never allocated.

Network inference and arrival-time sampling use the same grouped-bucket vmap
strategy as norm_flow_photons_fast to avoid per-event XLA recompilation.
"""
import functools

import awkward as ak
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from scipy.spatial import cKDTree

from hyperion.models.photon_arrival_time_nflow.net import (
    make_counts_net_fn,
    make_shape_conditioner_fn,
    traf_dist_builder,
)
from prometheus.compat.haiku_unpickler import load as haiku_load

from .norm_flow_photons import (  # noqa: F401
    make_nflow_photon_likelihood,
    make_nflow_photon_likelihood_per_module,
)

BATCH_CAP: int = 1024
_DISTANCE_CUTOFF_M: float = 300.0


def _next_bucket(n, base=2):
    if n <= 0:
        return 1
    log_cnt = np.log(n) / np.log(base)
    return int(np.power(base, np.ceil(log_cnt)))


def _sparse_pairs(mod_np, src_pos, src_dir, src_t0, c_medium):
    """Compute geometry for source-module pairs within the 300 m cutoff.

    Parameters
    ----------
    mod_np : np.ndarray, shape (n_modules, 3)
    src_pos : np.ndarray, shape (n_sources, 3)
    src_dir : np.ndarray, shape (n_sources, 3)
    src_t0 : np.ndarray, shape (n_sources,)  — emission times
    c_medium : float

    Returns
    -------
    pair_mod : np.ndarray int32, shape (n_pairs,)
    pair_src : np.ndarray int32, shape (n_pairs,)
    inp_pars : np.ndarray float32, shape (n_pairs, 2)  — [log10(dist), angle]
    time_geo : np.ndarray float32, shape (n_pairs,)
    """
    tree = cKDTree(mod_np)

    lists_mod, lists_src, lists_inp, lists_tgeo = [], [], [], []

    for i in range(len(src_pos)):
        near = np.asarray(
            tree.query_ball_point(src_pos[i], r=_DISTANCE_CUTOFF_M), dtype=np.int32
        )
        if near.size == 0:
            continue

        vecs = mod_np[near] - src_pos[i]             # (k, 3)
        dists = np.linalg.norm(vecs, axis=1)          # (k,)

        ok = dists > 0
        if not ok.any():
            continue
        near, vecs, dists = near[ok], vecs[ok], dists[ok]

        cos_a = np.clip((vecs * src_dir[i]).sum(axis=1) / dists, -1.0, 1.0)
        angles = np.arccos(cos_a).astype(np.float32)
        log_dist = np.log10(dists).astype(np.float32)
        t_geo = (dists / c_medium + float(src_t0[i])).astype(np.float32)

        lists_mod.append(near)
        lists_src.append(np.full(near.size, i, dtype=np.int32))
        lists_inp.append(np.stack([log_dist, angles], axis=1))
        lists_tgeo.append(t_geo)

    if not lists_mod:
        return (
            np.empty(0, np.int32),
            np.empty(0, np.int32),
            np.empty((0, 2), np.float32),
            np.empty(0, np.float32),
        )

    pair_mod = np.concatenate(lists_mod)
    pair_src = np.concatenate(lists_src)
    inp_pars = np.concatenate(lists_inp)
    time_geo = np.concatenate(lists_tgeo)

    # Sort by (module, source) for deterministic output ordering.
    order = np.lexsort((pair_src, pair_mod))
    return pair_mod[order], pair_src[order], inp_pars[order], time_geo[order]


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

    @functools.partial(jax.jit, static_argnums=(1,))
    def sample_batch(traf_params, n_max, keys):
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
        if isinstance(seed, int):
            key = random.PRNGKey(seed)
        else:
            key = seed

        n_modules = module_coords.shape[0]

        mod_np = np.asarray(module_coords)
        src_pos_np = np.asarray(source_pos)
        src_dir_np = np.asarray(source_dir)
        src_t0_np = np.asarray(source_time).ravel()
        src_n_np = np.asarray(source_nphotons).ravel()
        mod_eff_np = np.asarray(module_efficiencies)

        pair_mod, pair_src, inp_pars_np, time_geo_np = _sparse_pairs(
            mod_np, src_pos_np, src_dir_np, src_t0_np, c_medium
        )

        n_pairs = inp_pars_np.shape[0]
        if n_pairs == 0:
            return ak.Array([])

        # ── network inference ─────────────────────────────────────────────────
        bucket = _next_bucket(n_pairs)
        pad = bucket - n_pairs
        inp_jax = jnp.array(np.pad(inp_pars_np, ((0, pad), (0, 0))))

        ph_frac = jnp.power(
            10, counts_net.apply(counts_params, inp_jax)
        ).reshape(-1)[:n_pairs]

        src_ph_jax = jnp.array(src_n_np[pair_src])
        mod_eff_jax = jnp.array(mod_eff_np[pair_mod])
        n_ph_float = ph_frac * src_ph_jax * mod_eff_jax

        key, subkey = random.split(key)
        n_ph = random.poisson(subkey, n_ph_float).astype(jnp.int32)
        n_ph_cpu = np.asarray(n_ph)

        if not np.any(n_ph_cpu > 0):
            return ak.Array([])

        traf_params = apply_fn(shape_params, inp_jax)[:n_pairs]

        # ── grouped-bucket vmap sampling ──────────────────────────────────────
        subkeys = random.split(key, n_pairs)
        pair_times: list = [None] * n_pairs

        bucket_groups: dict[int, list[int]] = {}
        for i in np.where(n_ph_cpu > 0)[0]:
            n_i = int(n_ph_cpu[i])
            b = _next_bucket(n_i)
            if b <= BATCH_CAP:
                bucket_groups.setdefault(b, []).append(int(i))
            else:
                raw = sample_single_pair(traf_params[i], b, subkeys[i])
                pair_times[i] = np.asarray(raw[:n_i]) + time_geo_np[i]

        for bucket_size, indices in bucket_groups.items():
            idx = np.array(indices, dtype=np.int32)
            n_in_batch = len(idx)
            batch_b = _next_bucket(n_in_batch)
            # Pad to power-of-2 so XLA sees a stable shape across events.
            if batch_b > n_in_batch:
                idx = np.concatenate(
                    [idx, np.zeros(batch_b - n_in_batch, dtype=np.int32)]
                )
            batch_raw = np.asarray(
                sample_batch(traf_params[idx], bucket_size, subkeys[idx])
            )
            for j, i in enumerate(indices):
                pair_times[i] = batch_raw[j, : int(n_ph_cpu[i])] + time_geo_np[i]

        # ── reconstruct per-module arrays ─────────────────────────────────────
        module_times: list = [[] for _ in range(n_modules)]
        for k in range(n_pairs):
            if pair_times[k] is not None:
                module_times[pair_mod[k]].append(pair_times[k])

        result = [
            np.concatenate(t) if t else np.empty(0, dtype=np.float32)
            for t in module_times
        ]
        return ak.Array(result)

    return generate_norm_flow_photons
