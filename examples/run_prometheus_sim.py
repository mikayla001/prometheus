#!/usr/bin/env python3
"""02_basic_ice.py
modified the 02 ice file to make it run on gpu, instead of cpu, for test purposes
Minimal ice-case example to validate a Prometheus install with PPC.

Runs a single-event GPU-only simulation using the demo ice geo file
and the south-pole PPC ice tables bundled in resources/.

output directory is customizable. export $OUT before running file
"""
import os
from pathlib import Path
import logging
import sys

logger = logging.getLogger(__name__)

try:
    from prometheus import Prometheus, config
except Exception:
    logger.exception(
        "Error importing Prometheus. "
        "Ensure the environment is activated and requirements are installed."
    )
    logger.info(
        "Hint: source scripts/activate.sh .prometheus_env && pip install -r requirements.txt"
    )
    sys.exit(1)

# Use CPU-only JAX
try:
    import jax

    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_platform_name", "cpu")
except Exception:
    pass


def main():
    # Minimal runtime configuration
    config.run.run_number = 3
    config.run.random_state_seed = 7003
    config.run.nevents = 200

    # Point storage_prefix to OUT
    out_dir = os.environ.get("OUT")
    if out_dir is None:
        raise RuntimeError("Environment variable OUT is not set")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config.run.storage_prefix = str(out_dir)

    # Injection: volume (non-ranged) — vertex placed inside the detector volume
    config.injection.name = "LeptonInjector"
    config.injection.lepton_injector.simulation.is_ranged = False
    config.injection.lepton_injector.simulation.final_state_1 = "TauMinus"
    config.injection.lepton_injector.simulation.final_state_2 = "Hadrons"
    config.injection.lepton_injector.simulation.minimal_energy = 1e4
    config.injection.lepton_injector.simulation.maximal_energy = 1e6

    # Use the demo ice geo shipped in resources/

    _geo_default = "resources/geofiles/icecube.geo"
    _geo_path = Path(_geo_default)
    if not _geo_path.is_absolute() and not _geo_path.exists():
        REPO_ROOT = Path(__file__).resolve().parent.parent
        _geo_default = str(REPO_ROOT / _geo_default)
    config.detector.geo_file = _geo_default

    # Force PPC_CUDA as the photon propagator and allow re-use of a stale tmp dir
    # changed PPC to PPC_CUDA to allow to run on GPU, not CPU
    config.photon_propagator.name = "PPC_CUDA"
    config.photon_propagator.ppc_cuda.paths.force = True

    print("Initializing Prometheus (ice / PPC)")
    prom = Prometheus()
    print("Prometheus initialized")

    try:
        prom.sim()
    except Exception:
        logger.exception("Simulation error during prom.sim()")
        logger.info(
            "Hint: check PPC binaries, tables, and that config.photon_propagator is set correctly."
        )
        sys.exit(1)

    print("Simulation completed successfully")


if __name__ == "__main__":
    main()
