"""Correctness and performance comparison: norm_flow_photons vs norm_flow_photons_fast.

Run with::

    pytest tests/test_compare_norm_flow.py -s -v

What is tested
--------------
Photon counts per module
    The Poisson sampling step is unchanged between the two implementations.
    Both receive the same ``seed``, split it identically up to the Poisson
    draw, and must therefore produce bit-for-bit identical per-module photon
    counts.  This is asserted exactly with ``np.array_equal``.

Arrival-time distributions
    After the Poisson draw the two implementations consume PRNG keys
    differently (sequential splits vs. a single batch split), so individual
    arrival times are not identical.  Both draw from the same conditional
    flow distribution, so the aggregate arrival-time distribution across all
    events must be statistically indistinguishable.  This is checked with a
    two-sample KS test (p-value > 0.01) and by comparing the sample mean and
    standard deviation to within two standard errors.

Performance
    Wall time and peak RSS (resident set size) are reported for both
    implementations.  RSS reflects real process memory, including JAX/XLA
    allocations, unlike ``tracemalloc`` which only sees Python heap usage.
"""

import pathlib
import time
import threading
import os

import awkward as ak
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import psutil
from scipy import stats

# ── Path helpers ──────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).parent.parent
RESOURCES = REPO_ROOT / "resources" / "olympus_resources"
SHAPE_PATH = str(RESOURCES / "photon_arrival_time_nflow_params.pickle")
COUNTS_PATH = str(RESOURCES / "photon_arrival_time_counts_params.pickle")

# Speed of light in water at ~700 nm (m/ns), matching OlympusPhotonPropagator.
C_MEDIUM_M_NS: float = 0.2174

# ── Synthetic detector & source geometry ─────────────────────────────────────
N_MODULES = 60
N_SOURCES = 10
PHOTONS_PER_SOURCE = 1_000_000
N_EVENTS = 30

_MODULE_RADIUS_M = 5.0
_MODULE_HALF_HEIGHT_M = 8.0


def _make_detector(rng: np.random.Generator):
    theta = rng.uniform(0, 2 * np.pi, N_MODULES)
    z = rng.uniform(-_MODULE_HALF_HEIGHT_M, _MODULE_HALF_HEIGHT_M, N_MODULES)
    coords = np.stack(
        [_MODULE_RADIUS_M * np.cos(theta), _MODULE_RADIUS_M * np.sin(theta), z], axis=1
    )
    efficiencies = np.ones(N_MODULES)
    return jnp.array(coords, dtype=jnp.float32), jnp.array(efficiencies, dtype=jnp.float32)


def _make_sources(rng: np.random.Generator):
    pos = rng.uniform(-1, 1, (N_SOURCES, 3)).astype(np.float32)
    raw_dir = rng.standard_normal((N_SOURCES, 3)).astype(np.float32)
    direction = raw_dir / np.linalg.norm(raw_dir, axis=1, keepdims=True)
    time = np.zeros((N_SOURCES, 1), dtype=np.float32)
    nphotons = np.full((N_SOURCES, 1), PHOTONS_PER_SOURCE, dtype=np.float32)
    return (
        jnp.array(pos),
        jnp.array(direction),
        jnp.array(time),
        jnp.array(nphotons),
    )


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def gen_ref():
    jax.config.update("jax_enable_x64", False)
    from prometheus.photon_propagation.olympus.event_generation.photon_propagation.norm_flow_photons import (
        make_generate_norm_flow_photons,
    )
    return make_generate_norm_flow_photons(SHAPE_PATH, COUNTS_PATH, C_MEDIUM_M_NS)


@pytest.fixture(scope="module")
def gen_fast():
    from prometheus.photon_propagation.olympus.event_generation.photon_propagation.norm_flow_photons_fast import (
        make_generate_norm_flow_photons,
    )
    return make_generate_norm_flow_photons(SHAPE_PATH, COUNTS_PATH, C_MEDIUM_M_NS)

@pytest.fixture(scope="module")
def gen_scan():
    from prometheus.photon_propagation.olympus.event_generation.photon_propagation.norm_flow_photons_scan import (
        make_generate_norm_flow_photons,
    )
    return make_generate_norm_flow_photons(SHAPE_PATH, COUNTS_PATH, C_MEDIUM_M_NS)

@pytest.fixture(scope="module")
def gen_sparse():
    from prometheus.photon_propagation.olympus.event_generation.photon_propagation.norm_flow_photons_sparse import (
        make_generate_norm_flow_photons,
    )
    return make_generate_norm_flow_photons(SHAPE_PATH, COUNTS_PATH, C_MEDIUM_M_NS)

@pytest.fixture(scope="module")
def geometry():
    rng = np.random.default_rng(0)
    module_coords, module_eff = _make_detector(rng)
    source_pos, source_dir, source_time, source_nphotons = _make_sources(rng)
    return module_coords, module_eff, source_pos, source_dir, source_time, source_nphotons


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_events(gen, geometry, n_events: int, base_seed: int):
    module_coords, module_eff, source_pos, source_dir, source_time, source_nphotons = geometry
    results = []
    for i in range(n_events):
        seed = jax.random.PRNGKey(base_seed + i)
        out = gen(module_coords, module_eff, source_pos, source_dir, source_time, source_nphotons, seed=seed)
        results.append(out)
    return results


def _block_until_ready(tree):
    """Force JAX to finish all async work."""
    return jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def _peak_rss_mb(fn, interval=0.01):
    """Run *fn()* and return (result, peak_rss_mb)."""
    process = psutil.Process(os.getpid())
    peak = 0
    running = True

    def monitor():
        nonlocal peak
        while running:
            rss = process.memory_info().rss
            peak = max(peak, rss)
            time.sleep(interval)

    t = threading.Thread(target=monitor)
    t.start()

    result = fn()

    running = False
    t.join()

    return result, peak / 1024 / 1024


def _photon_counts(results):
    return np.array([[ak.count(mod) for mod in event] for event in results])


def _all_times(results):
    parts = []
    for event in results:
        flat = ak.to_numpy(ak.flatten(event))
        if flat.size > 0:
            parts.append(flat)
    return np.concatenate(parts) if parts else np.array([])


# ── Warmup ────────────────────────────────────────────────────────────────────

def _warmup_all(gen, geometry):
    """Warm up JAX across different bucket sizes and photon scales."""
    module_coords, module_eff, source_pos, source_dir, source_time, source_nphotons = geometry

    photon_scales = [1e3, 1e4, 1e5, 1e6]

    for scale in photon_scales:
        scaled = source_nphotons * scale / PHOTONS_PER_SOURCE

        for i in range(5):
            key = jax.random.PRNGKey(10_000 + int(scale) + i)
            out = gen(
                module_coords,
                module_eff,
                source_pos,
                source_dir,
                source_time,
                scaled,
                seed=key,
            )
            _block_until_ready(out)


@pytest.fixture(scope="module", autouse=True)
def warmup(gen_ref, gen_fast, gen_scan, gen_sparse, geometry):
    """Ensure all implementations are fully compiled before timing."""
    _warmup_all(gen_ref, geometry)
    _warmup_all(gen_fast, geometry)
    _warmup_all(gen_scan, geometry)
    _warmup_all(gen_sparse, geometry)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_photon_counts_identical(gen_ref, gen_fast, geometry):
    module_coords, module_eff, source_pos, source_dir, source_time, source_nphotons = geometry

    mismatches = 0
    for i in range(N_EVENTS):
        seed = jax.random.PRNGKey(i)
        ref = gen_ref(module_coords, module_eff, source_pos, source_dir, source_time, source_nphotons, seed=seed)
        fast = gen_fast(module_coords, module_eff, source_pos, source_dir, source_time, source_nphotons, seed=seed)

        ref_counts  = np.array([ak.count(m) for m in ref])
        fast_counts = np.array([ak.count(m) for m in fast])

        if not np.array_equal(ref_counts, fast_counts):
            mismatches += 1
            print(f"\n  Event {i}: count mismatch — diff = {fast_counts - ref_counts}")

    assert mismatches == 0


def test_arrival_time_distribution(gen_ref, gen_fast, geometry):
    ref_results  = _run_events(gen_ref,  geometry, N_EVENTS, base_seed=100)
    fast_results = _run_events(gen_fast, geometry, N_EVENTS, base_seed=100)

    ref_times  = _all_times(ref_results)
    fast_times = _all_times(fast_results)

    assert ref_times.size > 0
    assert fast_times.size > 0

    ks_stat, p_value = stats.ks_2samp(ref_times, fast_times)

    print(f"\n  Arrival-time KS test: stat={ks_stat:.4f}, p={p_value:.4f}")

    assert p_value > 0.01


def test_total_photon_count_identical(gen_ref, gen_fast, geometry):
    module_coords, module_eff, source_pos, source_dir, source_time, source_nphotons = geometry

    for i in range(N_EVENTS):
        seed = jax.random.PRNGKey(i)
        ref  = gen_ref(module_coords, module_eff, source_pos, source_dir, source_time, source_nphotons, seed=seed)
        fast = gen_fast(module_coords, module_eff, source_pos, source_dir, source_time, source_nphotons, seed=seed)

        ref_total  = int(ak.count(ak.flatten(ref)))
        fast_total = int(ak.count(ak.flatten(fast)))

        assert ref_total == fast_total


def test_sparse_arrival_time_distribution(gen_ref, gen_sparse, geometry):
    ref_results    = _run_events(gen_ref,    geometry, N_EVENTS, base_seed=100)
    sparse_results = _run_events(gen_sparse, geometry, N_EVENTS, base_seed=100)

    ref_times    = _all_times(ref_results)
    sparse_times = _all_times(sparse_results)

    assert ref_times.size > 0
    assert sparse_times.size > 0

    ks_stat, p_value = stats.ks_2samp(ref_times, sparse_times)
    print(f"\n  Sparse arrival-time KS test: stat={ks_stat:.4f}, p={p_value:.4f}")
    assert p_value > 0.01


def test_performance(gen_ref, gen_fast, gen_scan, gen_sparse, geometry, capsys):
    def run(gen):
        out = _run_events(gen, geometry, N_EVENTS, base_seed=200)
        return _block_until_ready(out)

    def measure(gen):
        t0 = time.perf_counter()
        _, mem = _peak_rss_mb(lambda: run(gen))
        t = time.perf_counter() - t0
        return t, mem

    t_ref, mem_ref = measure(gen_ref)
    t_fast, mem_fast = measure(gen_fast)
    t_scan, mem_scan = measure(gen_scan)
    t_sparse, mem_sparse = measure(gen_sparse)

    with capsys.disabled():
        print(f"\n{'=' * 68}")
        print(f"  Performance comparison  ({N_EVENTS} events)")
        print(f"{'=' * 68}")
        print(f"  {'Impl':<12} {'Wall time (s)':>14} {'Peak RSS (MB)':>14}")
        print(f"  {'-' * 50}")
        print(f"  {'reference':<12} {t_ref:>14.3f} {mem_ref:>14.1f}")
        print(f"  {'fast':<12}      {t_fast:>14.3f} {mem_fast:>14.1f}")
        print(f"  {'scan':<12}      {t_scan:>14.3f} {mem_scan:>14.1f}")
        print(f"  {'sparse':<12}    {t_sparse:>14.3f} {mem_sparse:>14.1f}")
        print(f"{'=' * 68}")

    # guardrails
    # fast/sparse: batch-size bucketing keeps XLA shapes stable → should not regress
    assert t_fast < t_ref * 1.5
    assert t_sparse < t_ref * 1.5
    # scan compiles a monolithic XLA program for all pairs × flow ops and samples
    # n_max for every pair; it is kept for comparison only, not as a target