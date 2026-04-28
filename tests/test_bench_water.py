"""Timing benchmark for the water (Olympus/JAX) simulation stack.

Marked ``timing`` — only runs when ``--timing`` is passed::

    pytest tests/test_bench_water.py --timing -s

``-s`` lets the timing table print to stdout.

Three stages are measured for each repetition:

- **Total**        — full ``prom.sim()`` wall time (injection + propagation + I/O)
- **Fennel**       — cumulative time spent in ``fennel_total_light_yield`` and
                     ``fennel_frac_long_light_yield`` across all events
- **Flow inference** — cumulative time spent in the normalizing-flow photon
                     generator (``_gen_ph``) across all events
"""

import copy
import time

import numpy as np
import pytest

N_WARMUP = 5
N_EVENTS = 100
N_REPS = 3
FIXED_ENERGY_GEV = 1e3  # 1 TeV — narrow range so cascade length is effectively fixed


def _make_cfg(tmp_path, nevents, seed):
    from prometheus import config as _config

    cfg = copy.deepcopy(_config)
    cfg.run.run_number = seed
    cfg.run.random_state_seed = seed
    cfg.run.nevents = nevents
    cfg.run.storage_prefix = str(tmp_path / f"run_{seed}") + "/"
    cfg.injection.name = "LeptonInjector"
    cfg.injection.lepton_injector.simulation.is_ranged = False
    cfg.injection.lepton_injector.simulation.minimal_energy = FIXED_ENERGY_GEV
    cfg.injection.lepton_injector.simulation.maximal_energy = FIXED_ENERGY_GEV + 1.0
    cfg.detector.geo_file = "resources/geofiles/demo_water.geo"
    return cfg


@pytest.mark.timing
def test_bench_water(tmp_path):
    """Benchmark water simulation: total, fennel, and flow-inference timing.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Pytest-provided temporary directory for simulation output.
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")

    from prometheus import Prometheus
    import olympus.event_generation.lightyield as ly

    # ── Warmup: trigger JAX JIT compilation, results discarded ───────────────
    prom_warmup = Prometheus(_make_cfg(tmp_path, N_WARMUP, seed=0))
    prom_warmup.sim()

    # ── Benchmark ─────────────────────────────────────────────────────────────
    total_times = []
    fennel_times = []
    flow_times = []

    _orig_total = ly.fennel_total_light_yield
    _orig_frac = ly.fennel_frac_long_light_yield

    for rep in range(N_REPS):
        rep_fennel = []
        rep_flow = []

        def timed_fennel_total(*args, **kwargs):
            t0 = time.perf_counter()
            result = _orig_total(*args, **kwargs)
            rep_fennel.append(time.perf_counter() - t0)
            return result

        def timed_fennel_frac(*args, **kwargs):
            t0 = time.perf_counter()
            result = _orig_frac(*args, **kwargs)
            rep_fennel.append(time.perf_counter() - t0)
            return result

        ly.fennel_total_light_yield = timed_fennel_total
        ly.fennel_frac_long_light_yield = timed_fennel_frac

        try:
            prom = Prometheus(_make_cfg(tmp_path, N_EVENTS, seed=100 + rep))

            _orig_gen_ph = prom._photon_propagator._gen_ph

            def timed_gen_ph(*args, **kwargs):
                t0 = time.perf_counter()
                result = _orig_gen_ph(*args, **kwargs)
                rep_flow.append(time.perf_counter() - t0)
                return result

            prom._photon_propagator._gen_ph = timed_gen_ph

            t0 = time.perf_counter()
            prom.sim()
            total_times.append(time.perf_counter() - t0)
        finally:
            ly.fennel_total_light_yield = _orig_total
            ly.fennel_frac_long_light_yield = _orig_frac

        fennel_times.append(sum(rep_fennel))
        flow_times.append(sum(rep_flow))

    # ── Report ────────────────────────────────────────────────────────────────
    total = np.array(total_times)
    fennel = np.array(fennel_times)
    flow = np.array(flow_times)

    def ms_per_evt(t):
        return f"{np.mean(t) / N_EVENTS * 1e3:.1f} ms/evt"

    def pct(t):
        return f"{np.mean(t) / np.mean(total) * 100:.1f}%"

    col = 20
    print(f"\n{'=' * 62}")
    print(f"  Water benchmark  ({N_EVENTS} events × {N_REPS} reps, {FIXED_ENERGY_GEV/1e3:.0f} TeV fixed)")
    print(f"{'=' * 62}")
    print(f"  {'Stage':<{col}} {'Mean (s)':>10} {'Std (s)':>9} {'ms/evt':>11} {'% total':>8}")
    print(f"  {'-' * 60}")
    print(f"  {'Total':<{col}} {np.mean(total):>10.2f} {np.std(total):>9.2f} {ms_per_evt(total):>11} {'100.0%':>8}")
    print(f"  {'Fennel':<{col}} {np.mean(fennel):>10.2f} {np.std(fennel):>9.2f} {ms_per_evt(fennel):>11} {pct(fennel):>8}")
    print(f"  {'Flow inference':<{col}} {np.mean(flow):>10.2f} {np.std(flow):>9.2f} {ms_per_evt(flow):>11} {pct(flow):>8}")
    print(f"{'=' * 62}")
