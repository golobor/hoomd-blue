#!/usr/bin/env python3
"""Benchmark HOOMD-blue LJ performance on GPU.

Runs a Lennard-Jones liquid simulation and reports TPS (timesteps per second).
Uses a large enough system to saturate the GPU.
"""

import sys
import time
import os

# Ensure we import from build directory, not source
build_dir = os.path.dirname(os.path.abspath(__file__))
if 'build' not in build_dir:
    build_dir = os.path.join(build_dir, 'build')
sys.path.insert(0, build_dir)

import hoomd
import numpy as np

def run_benchmark(N_per_side=30, warmup_steps=2000, bench_steps=5000, gpu_id=0):
    """Run LJ benchmark and return TPS."""
    
    N = N_per_side ** 3  # total particles
    density = 0.8
    L = (N / density) ** (1.0/3.0)
    
    print(f"System: {N} particles, box L={L:.2f}")
    print(f"GPU: {gpu_id}")
    print(f"HOOMD loaded from: {hoomd.__file__}")
    
    device = hoomd.device.GPU(gpu_id=gpu_id)
    
    # Create initial configuration: FCC lattice
    snap = hoomd.Snapshot(device.communicator)
    if snap.communicator.rank == 0:
        snap.configuration.box = [L, L, L, 0, 0, 0]
        snap.particles.N = N
        snap.particles.types = ['A']
        
        # Simple cubic positions
        positions = []
        spacing = L / N_per_side
        for ix in range(N_per_side):
            for iy in range(N_per_side):
                for iz in range(N_per_side):
                    positions.append([
                        (ix + 0.5) * spacing - L/2,
                        (iy + 0.5) * spacing - L/2,
                        (iz + 0.5) * spacing - L/2
                    ])
        snap.particles.position[:] = positions
        snap.particles.typeid[:] = 0
        snap.particles.body[:] = -1  # NO_BODY (0xFFFFFFFF as uint32)
        # Random velocities
        rng = np.random.default_rng(42)
        snap.particles.velocity[:] = rng.normal(0, 1.0, (N, 3))
        snap.particles.mass[:] = 1.0
    
    sim = hoomd.Simulation(device=device, seed=42)
    sim.create_state_from_snapshot(snap)
    
    # Set up LJ potential
    nlist = hoomd.md.nlist.Cell(buffer=0.4)
    lj = hoomd.md.pair.LJ(nlist=nlist, default_r_cut=2.5)
    lj.params[('A', 'A')] = dict(epsilon=1.0, sigma=1.0)
    
    # NVE integrator
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator = hoomd.md.Integrator(dt=0.005, methods=[nve], forces=[lj])
    sim.operations.integrator = integrator
    
    # Warmup
    print(f"Warming up ({warmup_steps} steps)...")
    sim.run(warmup_steps)
    
    # Benchmark
    print(f"Benchmarking ({bench_steps} steps)...")
    device.gpu_error_checking = False
    
    t0 = time.perf_counter()
    sim.run(bench_steps)
    t1 = time.perf_counter()
    
    elapsed = t1 - t0
    tps = bench_steps / elapsed
    
    print(f"\nResults:")
    print(f"  Elapsed: {elapsed:.3f} s")
    print(f"  TPS:     {tps:.1f}")
    print(f"  ns/day:  {tps * 0.005 * 86400 / 1e6:.2f}")
    
    return tps


if __name__ == "__main__":
    gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    N_per_side = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    tps = run_benchmark(N_per_side=N_per_side, gpu_id=gpu_id)
