#!/usr/bin/env python3
"""Profile HOOMD-blue GPU kernels with nsys.

Equilibrates a PatchyGaussian system, saves state, then runs a short
profiled segment for kernel-level timing.

Usage:
    # Full profile with nsys:
    nsys profile --stats=true -o profile_mixed \
        python profile_kernels.py --pythonpath build/install_mixed/lib/python3.12/site-packages

    # Or just run directly (no nsys) for a quick check:
    python profile_kernels.py --pythonpath build/install_mixed/lib/python3.12/site-packages
"""
import argparse
import os
import sys
import time

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pythonpath", required=True,
                   help="Path to site-packages for this build")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--n-particles", type=int, default=64000)
    p.add_argument("--chain-length", type=int, default=200)
    p.add_argument("--profile-steps", type=int, default=2000,
                   help="Steps to run under profiler (default: 2000)")
    p.add_argument("--warmup-steps", type=int, default=5000)
    p.add_argument("--state", default=None,
                   help="Pre-equilibrated GSD file (skip equilibration)")
    p.add_argument("--save-state", default=None,
                   help="Save equilibrated state to this path")
    a = p.parse_args()

    # Prepend pythonpath
    sys.path.insert(0, os.path.abspath(a.pythonpath))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)

    import hoomd
    print(f"HOOMD from: {hoomd.__file__}")

    device = hoomd.device.GPU()

    # Import benchmark infrastructure
    sys.path.insert(0, os.path.dirname(__file__))
    from benchmark_chains import equilibrate_and_save, _make_forces

    import math
    import numpy as np

    n_chains = a.n_particles // a.chain_length
    n_particles = n_chains * a.chain_length
    density = 0.3
    volume = n_particles / density
    sphere_radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)

    patchy = (1.0, 0.5, 0.6, 20.0, 1.5, 2)
    attract = (0.7, 1.5)
    dpd_A = 6.0

    # Equilibrate or load
    if a.state:
        import json
        meta_path = a.state + ".json"
        with open(meta_path) as f:
            meta = json.load(f)
        sphere_radius = meta["sphere_radius"]
        gsd_path = a.state
        print(f"Loaded state from {gsd_path}")
    else:
        gsd_path, sphere_radius = equilibrate_and_save(
            device, n_particles, a.chain_length,
            with_angle=True, with_dihedral=True,
            dpd_A=dpd_A, attract=attract, patchy=patchy,
            save_path=a.save_state,
        )

    # Build simulation from saved state
    sim = hoomd.Simulation(device=device, seed=42)
    sim.create_state_from_gsd(filename=gsd_path)

    forces = _make_forces(sphere_radius, with_angle=True, with_dihedral=True,
                          dpd_A=dpd_A, attract=attract, patchy=patchy)
    langevin = hoomd.md.methods.Langevin(
        filter=hoomd.filter.All(), kT=1.0, default_gamma=1.0,
    )
    integrator = hoomd.md.Integrator(
        dt=0.005, methods=[langevin], forces=forces,
    )
    integrator.integrate_rotational_dof = True
    sim.operations.integrator = integrator
    device.gpu_error_checking = False

    # Warmup (outside profiler)
    print(f"\nWarming up ({a.warmup_steps} steps)...")
    t0 = time.perf_counter()
    sim.run(a.warmup_steps)
    t1 = time.perf_counter()
    print(f"  warmup TPS: {a.warmup_steps / (t1 - t0):.1f}")

    # Profiled segment
    print(f"\nProfiling ({a.profile_steps} steps)...")
    t0 = time.perf_counter()
    sim.run(a.profile_steps)
    t1 = time.perf_counter()

    tps = a.profile_steps / (t1 - t0)
    print(f"  Profile TPS: {tps:.1f}")
    print(f"  Done. Use 'nsys stats <report>.nsys-rep' to analyze.")

if __name__ == "__main__":
    main()
