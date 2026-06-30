#!/usr/bin/env python3
"""02_basic_ice.py
Minimal ice-case example to validate a Prometheus install with PPC.

Now with CLI flags similar to examples/run_prometheus_sim.py.
"""

import logging
import sys
import argparse
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from prometheus import Prometheus, config
except Exception:
    logger.exception(
        "Error importing Prometheus. "
        "Ensure the environment is activated and requirements are installed."
    )
    logger.info(
        "Hint: source scripts/activate.sh .prometheus_env && "
        "pip install -r requirements.txt"
    )
    sys.exit(1)

# JAX: allow GPU (do NOT force CPU)
try:
    import jax

    jax.config.update("jax_enable_x64", True)
    # Let JAX choose gpu/cpu automatically; don't set jax_platform_name="cpu"
except Exception:
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Minimal Prometheus ice example with GPU and CLI flags."
    )

    # Match the flags you showed
    parser.add_argument("-n", "--nevents", type=int, default=1,
                        help="Number of events")
    parser.add_argument("-s", "--seed", type=int, default=2,
                        help="Seed / run_number (names the output files)")

    parser.add_argument("--final_1", type=str, default="TauMinus",
                        help="Final state 1 (e.g. TauMinus, MuMinus, NuEBar)")
    parser.add_argument("--final_2", type=str, default="Hadrons",
                        help="Final state 2 (e.g. Hadrons)")

    parser.add_argument("--emin", type=float, default=1e4,
                        help="Minimum energy in GeV")
    parser.add_argument("--emax", type=float, default=1e6,
                        help="Maximum energy in GeV")

    parser.add_argument("--geo", type=str,
                        default="resources/geofiles/icecube.geo",
                        help="Detector geofile")

    parser.add_argument("--propagator", type=str, default="PPC_CUDA",
                        help="Photon propagator (e.g. PPC, PPC_CUDA)")

    parser.add_argument("--storage-prefix", type=str, default="./output",
                        help="Output directory (like --storage-prefix in CLI)")

    # Optional: device index if your config supports it
    parser.add_argument("--device", type=int, default=0,
                        help="GPU index (if supported by Prometheus config)")

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Runtime config from CLI ---

    # seed / run_number
    config.run.run_number = args.seed
    config.run.random_state_seed = args.seed
    config.run.nevents = args.nevents

    # channel
    config.injection.name = "LeptonInjector"
    sim = config.injection.lepton_injector.simulation
    sim.is_ranged = False  # you can add a --ranged flag if you want
    sim.final_state_1 = args.final_1
    sim.final_state_2 = args.final_2
    sim.minimal_energy = args.emin
    sim.maximal_energy = args.emax

    # geometry
    _geo_default = args.geo
    _geo_path = Path(_geo_default)
    if not _geo_path.is_absolute() and not _geo_path.exists():
        REPO_ROOT = Path(__file__).resolve().parent.parent
        _geo_default = str(REPO_ROOT / _geo_default)
    config.detector.geo_file = _geo_default

    # propagator
    config.photon_propagator.name = args.propagator
    config.photon_propagator.ppc.paths.force = True

    # storage / output directory
    config.run.storage_prefix = args.storage_prefix

    # Optional: GPU device index (ONLY if Prometheus exposes this field)
    # try:
    #     config.run.device = args.device
    # except AttributeError:
    #     pass

    print(
        f"Initializing Prometheus (propagator={args.propagator}, "
        f"final_1={args.final_1}, final_2={args.final_2})"
    )
    prom = Prometheus()
    print("Prometheus initialized")

    try:
        prom.sim()
    except Exception:
        logger.exception("Simulation error during prom.sim()")
        logger.info(
            "Hint: check PPC binaries/tables, geometry path, "
            "and GPU visibility (CUDA, drivers)."
        )
        sys.exit(1)

    print("Simulation completed successfully")


if __name__ == "__main__":
    main()
