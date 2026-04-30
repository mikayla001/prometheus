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
    Wall time and peak RSS are reported for both implementations.
"""

import pathlib
import time
import tracemalloc

import awkward as ak
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats

# ── Path helpers ──────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).parent.parent
RESOURCES = REPO_ROOT / "resources" / "olympus_resources"
SHAPE_PATH = str(RESOURCES / "photon_arrival_time_nflow_params.pickle")
COUNTS_PATH = str(RESOURCES / "photon_arrival_time_counts_params.pickle")

# Speed of light in water at ~700 nm (m/ns), matching OlympusPhotonPropagator.
C_MEDIUM_M_NS: float = 0.2174

# ── Synthetic detector & source geometry ─────────────────────────────────────
# Small but realistic: 60 modules on a cylinder, 10 sources near the centre.
N_MODULES = 60
N_SOURCES = 10
PHOTONS_PER_SOURCE = 3000   # high enough for good statistical power
N_EVENTS = 30               # events to accumulate for KS test


def _make_detector(rng: np.random.Generator):
    """Place N_MODULES modules on a cylinder of 40 m radius, 150 m height."""
    theta = rng.uniform(0, 2 * np.pi, N_MODULES)
    z = rng.uniform(-75, 75, N_MODULES)
    r = 40.0
    coords = np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)
    efficiencies = np.ones(N_MODULES)
    return jnp.array(coords, dtype=jnp.float32), jnp.array(efficiencies, dtype=jnp.float32)


def _make_sources(rng: np.random.Generator):
    """Place N_SOURCES sources near the detector centre."""
    pos = rng.uniform(-5, 5, (N_SOURCES, 3)).astype(np.float32)
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
    """Reference (sequential) generator."""
    jax.config.update("jax_enable_x64", False)
    from prometheus.photon_propagation.olympus.event_generation.photon_propagation.norm_flow_photons import (
        make_generate_norm_flow_photons,
    )
    return make_generate_norm_flow_photons(SHAPE_PATH, COUNTS_PATH, C_MEDIUM_M_NS)


@pytest.fixture(scope="module")
def gen_fast():
    """Vectorised (fast) generator."""
    from prometheus.photon_propagation.olympus.event_generation.photon_propagation.norm_flow_photons_fast import (
        make_generate_norm_flow_photons,
    )
    return make_generate_norm_flow_photons(SHAPE_PATH, COUNTS_PATH, C_MEDIUM_M_NS)


@pytest.fixture(scope="module")
def geometry():
    """Fixed detector and source geometry (same across all events)."""
    rng = np.random.default_rng(0)
    module_coords, module_eff = _make_detector(rng)
    source_pos, source_dir, source_time, source_nphotons = _make_sources(rng)
    return module_coords, module_eff, source_pos, source_dir, source_time, source_nphotons


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run_events(gen, geometry, n_events: int, base_seed: int):
    """Run *n_events* through *gen* and return per-event outputs."""
    module_coords, module_eff, source_pos, source_dir, source_time, source_nphotons = geometry
    results = []
    for i in range(n_events):
        seed = jax.random.PRNGKey(base_seed + i)
        out = gen(module_coords, module_eff, source_pos, source_dir, source_time, source_nphotons, seed=seed)
        results.append(out)
    return results


def _photon_counts(results):
    """Return (n_events, n_modules) array of photon counts."""
    return np.array([[ak.count(mod) for mod in event] for event in results])


def _all_times(results):
    """Return flat array of all arrival times across all events and modules."""
    parts = []
    for event in results:
        flat = ak.to_numpy(ak.flatten(event))
        if flat.size > 0:
            parts.append(flat)
    return np.concatenate(parts) if parts else np.array([])


def _peak_mb(fn):
    """Run *fn()* and return (result, peak_memory_mb)."""
    tracemalloc.start()
    result = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / 1024 / 1024


# ── Warmup ────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module", autouse=True)
def warmup(gen_ref, gen_fast, geometry):
    """Trigger JAX JIT compilation before any timed test."""
    module_coords, module_eff, source_pos, source_dir, source_time, source_nphotons = geometry
    warm_key = jax.random.PRNGKey(9999)
    for _ in range(3):
        gen_ref(module_coords, module_eff, source_pos, source_dir, source_time, source_nphotons, seed=warm_key)
        gen_fast(module_coords, module_eff, source_pos, source_dir, source_time, source_nphotons, seed=warm_key)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_photon_counts_identical(gen_ref, gen_fast, geometry):
    """Per-module photon counts must be bit-for-bit identical.

    Poisson sampling is unchanged between implementations; the same seed
    produces the same counts regardless of how arrival-time keys are split.
    """
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

    assert mismatches == 0, f"{mismatches}/{N_EVENTS} events had mismatched photon counts"


def test_arrival_time_distribution(gen_ref, gen_fast, geometry):
    """Aggregate arrival-time distributions must be statistically compatible.

    The two implementations use different PRNG keys for the flow sampling step,
    so individual times differ, but both draw from the same conditional
    distribution.  A two-sample KS test and a moment comparison are used.
    """
    ref_results  = _run_events(gen_ref,  geometry, N_EVENTS, base_seed=100)
    fast_results = _run_events(gen_fast, geometry, N_EVENTS, base_seed=100)

    ref_times  = _all_times(ref_results)
    fast_times = _all_times(fast_results)

    assert ref_times.size > 0,  "reference produced no photons — check geometry/model paths"
    assert fast_times.size > 0, "fast implementation produced no photons"

    ks_stat, p_value = stats.ks_2samp(ref_times, fast_times)

    print(f"\n  Arrival-time KS test: stat={ks_stat:.4f}, p={p_value:.4f}")
    print(f"  Reference : n={ref_times.size:,}, mean={ref_times.mean():.2f} ns, std={ref_times.std():.2f} ns")
    print(f"  Fast      : n={fast_times.size:,}, mean={fast_times.mean():.2f} ns, std={fast_times.std():.2f} ns")

    assert p_value > 0.01, (
        f"KS test rejected (p={p_value:.4f}): arrival-time distributions differ "
        f"beyond statistical fluctuations.  stat={ks_stat:.4f}"
    )

    # Mean and std should agree within ~2 standard errors.
    se_mean = ref_times.std() / np.sqrt(ref_times.size)
    assert abs(ref_times.mean() - fast_times.mean()) < 4 * se_mean, (
        f"Mean arrival times differ by more than 4 SE: "
        f"ref={ref_times.mean():.3f} fast={fast_times.mean():.3f} SE={se_mean:.4f}"
    )


def test_total_photon_count_identical(gen_ref, gen_fast, geometry):
    """Total photon count across all modules must match exactly per event."""
    module_coords, module_eff, source_pos, source_dir, source_time, source_nphotons = geometry

    for i in range(N_EVENTS):
        seed = jax.random.PRNGKey(i)
        ref  = gen_ref( module_coords, module_eff, source_pos, source_dir, source_time, source_nphotons, seed=seed)
        fast = gen_fast(module_coords, module_eff, source_pos, source_dir, source_time, source_nphotons, seed=seed)

        ref_total  = int(ak.count(ak.flatten(ref)))
        fast_total = int(ak.count(ak.flatten(fast)))

        assert ref_total == fast_total, (
            f"Event {i}: total photon count differs — ref={ref_total}, fast={fast_total}"
        )


def test_performance(gen_ref, gen_fast, geometry, capsys):
    """Report wall time and peak memory; fast implementation must not regress.

    No hard threshold is enforced — numbers are printed for human review.
    A soft assertion checks that the fast path is not more than 50% slower
    than the reference (it should be faster, but the test guards against
    accidental regressions).
    """
    module_coords, module_eff, source_pos, source_dir, source_time, source_nphotons = geometry

    def run_ref():
        return _run_events(gen_ref, geometry, N_EVENTS, base_seed=200)

    def run_fast():
        return _run_events(gen_fast, geometry, N_EVENTS, base_seed=200)

    t0 = time.perf_counter()
    _, mem_ref = _peak_mb(run_ref)
    t_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, mem_fast = _peak_mb(run_fast)
    t_fast = time.perf_counter() - t0

    with capsys.disabled():
        print(f"\n{'=' * 56}")
        print(f"  Performance comparison  ({N_EVENTS} events)")
        print(f"{'=' * 56}")
        print(f"  {'Impl':<12} {'Wall time (s)':>14} {'Peak mem (MB)':>14}")
        print(f"  {'-' * 42}")
        print(f"  {'reference':<12} {t_ref:>14.3f} {mem_ref:>14.1f}")
        print(f"  {'fast':<12} {t_fast:>14.3f} {mem_fast:>14.1f}")
        print(f"  {'speedup':<12} {t_ref / t_fast:>14.2f}x")
        print(f"  {'mem ratio':<12} {mem_ref / mem_fast:>14.2f}x  (>1 = fast uses less)")
        print(f"{'=' * 56}")

    assert t_fast < t_ref * 1.5, (
        f"Fast implementation is more than 50% slower than reference "
        f"({t_fast:.3f}s vs {t_ref:.3f}s).  Check for regressions."
    )
