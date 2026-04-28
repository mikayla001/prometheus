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


def _next_bucket(n, base=2):
    """Return the smallest power of *base* that is >= *n*.

    Base 2 is the default: it limits worst-case padding to < 2× the true count
    (vs up to 4× with base 4), which matters when individual pairs carry millions
    of photons at high neutrino energies.

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
        """Sample *n_padded* photon arrival times for one (source, module) pair.

        Accepts 1-D flow parameters for a single pair so the spline knot arrays
        stay at shape ``(num_bins + 1,)`` rather than ``(n_photons, num_bins + 1)``.
        The ``jnp.vectorize`` inside ``_Flow.forward`` broadcasts the 1-D knots
        over the ``n_padded`` base samples, giving the same result as the former
        ``jnp.repeat`` approach while using O(num_bins) memory for knots instead
        of O(n_photons × num_bins).

        ``n_padded`` is a static argument so JAX compiles one kernel per bucket
        size (powers of 2) and reuses it across events.

        Parameters
        ----------
        traf_p : jnp.ndarray
            Flow transformation parameters for one pair, shape ``(n_flow_params,)``.
        n_padded : int
            Number of samples to draw; must equal ``_next_bucket(n_actual)``.
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

    def generate_norm_flow_photons(
        module_coords,
        module_efficiencies,
        source_pos,
        source_dir,
        source_time,
        source_nphotons,
        seed=31337,
    ):
        """Generate photon arrival times at each detector module using the normalizing flow.

        Computes expected photon counts via the count network, samples the
        detected photon count per module with Poisson statistics, then samples
        arrival times from the normalizing-flow shape model.

        Source arrays and distance-masked input arrays are bucket-padded to the
        next power-of-2 length before every JIT-compiled network call so that
        JAX reuses compiled kernels across events with different source counts
        rather than retracing on every invocation. Arrival times are then
        sampled per (source, module) pair to avoid materialising a
        ``[total_photons, n_flow_params]`` array.

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

        # Bucket-pad source arrays to the next power-of-4 length so that
        # sources_to_model_input (jit + vmap over sources) sees a stable shape
        # across events and does not retrace for every unique source count.
        # Padded positions are placed far from the detector so the 300 m
        # distance mask filters them out; all real source-module pairs are
        # preserved because the vmap axis is sources, not modules.
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

        # Discard padded rows before any further computation.
        inp_pars = inp_pars[:n_sources]
        time_geo = time_geo[:n_sources]

        inp_pars = jnp.swapaxes(inp_pars, 0, 1)
        time_geo = jnp.swapaxes(time_geo, 0, 1)

        # Flatten: densely pack [modules, sources] into a 1-D index.
        inp_pars = inp_pars.reshape(
            (n_sources * module_coords.shape[0], inp_pars.shape[-1])
        )
        time_geo = time_geo.reshape(
            (n_sources * module_coords.shape[0], time_geo.shape[-1])
        )
        source_photons = jnp.tile(source_nphotons, module_coords.shape[0]).T.ravel()
        mod_eff_factor = jnp.repeat(module_efficiencies, n_sources)

        # Normalizing flows only built up to 300 m.
        distance_mask = inp_pars[:, 0] < np.log10(300)

        inp_params_masked = inp_pars[distance_mask]
        time_geo_masked = time_geo[distance_mask]
        source_photons_masked = source_photons[distance_mask]
        mod_eff_factor_masked = mod_eff_factor[distance_mask]

        n_masked = inp_params_masked.shape[0]
        if n_masked == 0:
            return ak.Array([])

        # Bucket-pad the distance-masked arrays to the next power-of-2 length
        # so that the JIT-compiled count network and flow conditioner see a
        # stable shape and do not retrace for every unique masked count.
        # Outputs are sliced back to n_masked to discard padding rows.
        masked_bucket = _next_bucket(n_masked)
        masked_pad = masked_bucket - n_masked
        inp_params_padded = jnp.pad(inp_params_masked, ((0, masked_pad), (0, 0)))

        # Evaluate count network to obtain photon survival fraction.
        ph_frac = jnp.power(
            10, counts_net.apply(counts_params, inp_params_padded)
        ).reshape(-1)[:n_masked]

        # Sample number of detected photons per (source, module) pair.
        n_photons_masked = ph_frac * source_photons_masked * mod_eff_factor_masked

        key, subkey = random.split(key)
        n_photons_masked = (
            random.poisson(subkey, n_photons_masked, shape=n_photons_masked.shape)
            .squeeze()
            .astype(jnp.int32)
        )

        if jnp.all(n_photons_masked == 0):
            return ak.Array([])

        # Obtain flow transformation parameters; slice to discard padding rows.
        traf_params = apply_fn(shape_params, inp_params_padded)[:n_masked]

        # Fill detected photon counts back into the full flat (module × source)
        # layout so the per-module split indices stay aligned.
        n_photons = jnp.zeros(n_sources * module_coords.shape[0], dtype=jnp.int32)
        n_photons = n_photons.at[distance_mask].set(n_photons_masked)
        n_photons = n_photons.reshape(module_coords.shape[0], n_sources)
        n_ph_per_mod = np.sum(n_photons, axis=1)

        # Materialise counts and geometric times to CPU once to avoid repeated
        # device syncs inside the loop below.
        n_ph_cpu = np.asarray(n_photons_masked)
        t_geo_cpu = np.atleast_1d(np.asarray(time_geo_masked.squeeze()))

        # Sample arrival times per (source, module) pair using 1-D flow params.
        # Avoids materialising the [total_photons, n_flow_params] array that the
        # previous jnp.repeat approach required; knot arrays stay at O(num_bins)
        # instead of O(n_photons × num_bins) for the full event.
        all_pair_times = []
        for i in range(n_masked):
            n_i = int(n_ph_cpu[i])
            if n_i == 0:
                continue
            key, subkey = random.split(key)
            raw = sample_single_pair(traf_params[i], _next_bucket(n_i), subkey)
            all_pair_times.append(np.asarray(raw[:n_i]) + t_geo_cpu[i])

        times = np.atleast_1d(
            np.concatenate(all_pair_times) if all_pair_times else np.array([])
        )
        times = np.split(times, np.cumsum(n_ph_per_mod)[:-1])
        return ak.Array(times)

    return generate_norm_flow_photons


def make_nflow_photon_likelihood_per_module(
    shape_model_path,
    counts_model_path,
    mode="full",
):
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
    )

    counts_net = make_counts_net_fn(counts_config)

    @jax.jit
    def counts_net_apply_fn(params, x):
        return counts_net.apply(params, x)

    @jax.jit
    def eval_l_p(traf_params, samples):
        return eval_log_prob(dist_builder, traf_params, samples)

    def per_module_shape_lh(t_res, inp_pars, source_weight):
        traf_params = apply_fn(shape_params, inp_pars)
        traf_params = traf_params.reshape((inp_pars.shape[0], traf_params.shape[-1]))

        distance_mask = (inp_pars[..., 0] < np.log10(300))[:, np.newaxis]
        finite_times = jnp.isfinite(t_res)
        physical = t_res > -4

        mask = distance_mask & finite_times & physical

        traf_params = traf_params.reshape((traf_params.shape[0], 1, traf_params.shape[1]))

        # Sanitize likelihood evaluation to avoid nans.
        sanitized_times = jnp.where(mask, t_res, jnp.zeros_like(t_res))
        shape_lh = eval_l_p(traf_params, sanitized_times)

        # Mask the scale factor. This will remove unwanted source-time pairs from the logsumexp
        scale_factor = source_weight[:, np.newaxis] * mask + 1e-15

        # logsumexp is log( sum_i b_i * (exp (a_i)))
        shape_lh = jax.scipy.special.logsumexp(shape_lh, b=scale_factor, axis=0)

        return shape_lh

    # per_module_shape_lh_v = jax.vmap(per_module_shape_lh, in_axes=[0, None, None])
    # per_module_shape_lh_v_j = jax.jit(jax.vmap(per_module_shape_lh, in_axes = [0, None, None]))

    def eval_per_module_likelihood(
        time,
        n_measured,
        module_coords,
        source_pos,
        source_dir,
        source_time,
        source_photons,
        c_medium,
        noise_rate,
    ):

        inp_pars, time_geo = sources_to_model_input_per_module(
            module_coords,
            source_pos,
            source_dir,
            source_time,
            c_medium,
        )

        inp_pars = inp_pars.reshape((source_pos.shape[0], inp_pars.shape[-1]))
        time_geo = time_geo.reshape((source_pos.shape[0], time_geo.shape[-1]))

        t_res = time - time_geo

        ph_frac = jnp.power(10, counts_net_apply_fn(counts_params, inp_pars)).reshape(
            source_pos.shape[0]
        )

        noise_window_len = 5000
        noise_photons = noise_rate * noise_window_len

        n_photons = jnp.reshape(ph_frac * source_photons.squeeze(), (source_pos.shape[0],))

        n_ph_pred_per_mod = jnp.sum(n_photons)
        n_ph_pred_per_mod_total = n_ph_pred_per_mod + noise_photons

        counts_lh = jnp.sum(
            -n_ph_pred_per_mod_total + n_measured * jnp.log(n_ph_pred_per_mod_total)
        )

        if mode == "counts" or time.shape[0] == 0:
            return counts_lh

        def total_shape_lh(t_res):
            source_weight = n_photons / jnp.sum(n_photons)

            shape_lh = per_module_shape_lh(t_res, inp_pars, source_weight)
            noise_lh = -jnp.log(noise_window_len)

            total_shape_lh = jnp.logaddexp(
                noise_lh + jnp.log(noise_photons / n_ph_pred_per_mod_total),
                shape_lh + jnp.log(n_ph_pred_per_mod / n_ph_pred_per_mod_total),
            )
            return total_shape_lh

        if mode == "full":
            return total_shape_lh(t_res).sum() + counts_lh

        elif mode == "tfirst":
            tfirst = jnp.min(time)
            tvanilla = jnp.linspace(-1000, tfirst, 5000)
            # tsamples = tvanilla / 5000 * (tfirst + 1000) - 1000
            tsamples = tvanilla - time_geo

            cumul = jnp.trapz(jnp.exp(total_shape_lh(tsamples)), x=tvanilla)

            llh = (
                jnp.log(n_measured)
                + total_shape_lh(tfirst - time_geo)
                + jnp.log(1 - cumul) * n_measured
            )

            return llh + counts_lh

    return eval_per_module_likelihood


def make_nflow_photon_likelihood(shape_model_path, counts_model_path):
    raise RuntimeError("Add noise")

    shape_config, shape_params = pickle.load(open(shape_model_path, "rb"))
    counts_config, counts_params = pickle.load(open(counts_model_path, "rb"))

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
    )

    counts_net = make_counts_net_fn(counts_config)

    @jax.jit
    def counts_net_apply_fn(params, x):
        return counts_net.apply(params, x)

    @jax.jit
    def eval_l_p(traf_params, samples):
        return eval_log_prob(dist_builder, traf_params, samples)

    def eval_likelihood(
        event,
        module_coords,
        source_pos,
        source_dir,
        source_time,
        source_photons,
        c_medium,
    ):
        inp_pars, time_geo = sources_to_model_input(
            module_coords,
            source_pos,
            source_dir,
            source_time,
            c_medium,
        )

        distance_mask = inp_pars[..., 0] < np.log10(300)
        inp_pars = inp_pars.reshape(
            (source_pos.shape[0] * module_coords.shape[0], inp_pars.shape[-1])
        )

        traf_params = apply_fn(shape_params, inp_pars)
        traf_params = traf_params.reshape(
            (source_pos.shape[0], module_coords.shape[0], traf_params.shape[-1])
        )

        hits_per_mod = jnp.asarray(ak.count(event, axis=1))

        flat_ev = jnp.asarray(ak.ravel(event))
        traf_params_rep = jnp.repeat(traf_params, hits_per_mod, axis=1)
        time_geo_rep = jnp.repeat(time_geo, hits_per_mod, axis=1).squeeze()
        distance_mask_rep = jnp.repeat(distance_mask, hits_per_mod, axis=1)

        t_res = flat_ev - time_geo_rep

        mask = distance_mask_rep & (t_res >= -4)
        shape_lh = jnp.where(
            mask, eval_l_p(traf_params_rep, t_res), jnp.zeros_like(distance_mask_rep)
        )

        ph_frac = jnp.power(10, counts_net_apply_fn(counts_params, inp_pars)).reshape(
            source_pos.shape[0], module_coords.shape[0]
        )

        n_photons = ph_frac * source_photons
        n_ph_pred_per_mod = jnp.sum(n_photons, axis=0)

        counts_lh = -n_ph_pred_per_mod + hits_per_mod * jnp.log(n_ph_pred_per_mod)

        return shape_lh.sum() + counts_lh.sum()

        lhsum = 0
        for imod in range(module_coords.shape[0]):
            if ak.count(event[imod]) == 0:
                continue

            dist_pars = traf_params[:, imod]
            mask = distance_mask[:, imod]

            if jnp.all(~mask):
                continue
            masked_pars = dist_pars[mask]

            t_res = jnp.asarray(event[imod]) - time_geo[:, imod][mask]

            per_mod_lh = eval_l_p(masked_pars, t_res.T)
            t_res_mask = t_res > -4

            zero_fill = jnp.zeros_like(per_mod_lh)

            lhsum += jnp.sum(jnp.where(t_res_mask.T, per_mod_lh, zero_fill))

            # lhsum += jnp.sum(per_mod_lh[t_res_mask.T])

        return lhsum

    return eval_likelihood
