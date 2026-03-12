#!/usr/bin/env python3
"""Measure Langevin TPS for HOOMD-blue polymer chain benchmarks.

Runs a Langevin dynamics simulation of polymer chains confined in a sphere
and reports timesteps-per-second (TPS) with per-chunk streaming output.

Equilibration is handled by benchlib; use --save-state / --load-state
to equilibrate once and reuse across runs.

Usage:
    python benchmark_tps.py [options]

Examples:
    # Default (64K particles, chains of 200, all forces):
    python benchmark_tps.py

    # No dihedrals:
    python benchmark_tps.py --no-dihedral

    # Patchy particles:
    python benchmark_tps.py --no-dihedral --patchy 1.0,0.5,0.6,20,1.5,2

    # Different dt:
    python benchmark_tps.py --dt 0.01

    # Custom system size:
    python benchmark_tps.py -N 256000 -L 400

    # Equilibrate once, reuse:
    python benchmark_tps.py --equilibrate-only --save-state /tmp/eq.gsd
    python benchmark_tps.py --load-state /tmp/eq.gsd --dt 0.005
    python benchmark_tps.py --load-state /tmp/eq.gsd --dt 0.01
"""

import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from benchlib import (
    TeeWriter, add_common_args, load_or_equilibrate,
    make_forces, parse_force_kwargs,
)

# ── Constants ─────────────────────────────────────────────────────────────

WARMUP_STEPS = 10_000
BENCH_STEPS = 100_000
REPORT_INTERVAL = 10_000


def run_tps(device, gsd_path, sphere_radius, dt, force_kwargs,
            warmup_steps=WARMUP_STEPS, bench_steps=BENCH_STEPS,
            report_interval=REPORT_INTERVAL):
    """Run Langevin benchmark and return TPS statistics.

    Prints per-chunk TPS as it goes (streaming / tail -f friendly).
    """
    import hoomd

    sim = hoomd.Simulation(device=device, seed=42)
    sim.create_state_from_gsd(filename=gsd_path)

    forces = make_forces(sphere_radius, **force_kwargs)
    has_patchy = force_kwargs.get("patchy") is not None

    langevin = hoomd.md.methods.Langevin(
        filter=hoomd.filter.All(), kT=1.0, default_gamma=1.0,
    )
    integrator = hoomd.md.Integrator(
        dt=dt, methods=[langevin], forces=forces,
    )
    if has_patchy:
        integrator.integrate_rotational_dof = True
    sim.operations.integrator = integrator
    device.gpu_error_checking = False

    # ── Warmup ────────────────────────────────────────────────────────
    print(f"\nWarming up ({warmup_steps} steps, reporting every "
          f"{report_interval})...")
    warmup_tps = []
    for chunk_start in range(0, warmup_steps, report_interval):
        chunk = min(report_interval, warmup_steps - chunk_start)
        t0 = time.perf_counter()
        sim.run(chunk)
        wall = time.perf_counter() - t0
        tps = chunk / wall
        warmup_tps.append(tps)
        step = chunk_start + chunk
        print(f"  warmup step {step:>6d}/{warmup_steps}: {tps:.1f} TPS")
    print()

    # ── Benchmark ─────────────────────────────────────────────────────
    print(f"Benchmarking ({bench_steps} steps, reporting every "
          f"{report_interval})...")
    bench_tps = []
    for chunk_start in range(0, bench_steps, report_interval):
        chunk = min(report_interval, bench_steps - chunk_start)
        t0 = time.perf_counter()
        sim.run(chunk)
        wall = time.perf_counter() - t0
        tps = chunk / wall
        bench_tps.append(tps)
        step = chunk_start + chunk
        print(f"  bench  step {step:>6d}/{bench_steps}: {tps:.1f} TPS")

    # ── Summary ───────────────────────────────────────────────────────
    avg_tps = np.mean(bench_tps)
    std_tps = np.std(bench_tps)
    ns_day = avg_tps * dt * 86400 / 1e6

    print(f"\nBenchmark summary (dt={dt}):")
    print(f"  Mean TPS: {avg_tps:.1f} +/- {std_tps:.1f}")
    print(f"  ns/day:   {ns_day:.3f}")
    print(f"  TPS range: [{min(bench_tps):.1f}, {max(bench_tps):.1f}]")
    print(f"  TPS std/mean: {std_tps / avg_tps * 100:.1f}%")

    return dict(
        warmup_tps=warmup_tps,
        bench_tps=bench_tps,
        avg_tps=float(avg_tps),
        std_tps=float(std_tps),
        ns_day=float(ns_day),
        dt=dt,
    )


def main():
    import argparse

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p)
    args = p.parse_args()

    if args.log:
        sys.stdout = TeeWriter(sys.stdout, args.log)

    import hoomd

    print(f"HOOMD: {hoomd.__file__}")
    print(f"Precision: {hoomd.version.floating_point_precision}")
    print(f"Version: {hoomd.version.version}")
    print()

    device = hoomd.device.GPU(gpu_id=args.gpu)
    gsd_path, sphere_radius = load_or_equilibrate(args, device)

    if args.equilibrate_only:
        print("Equilibration complete (--equilibrate-only).")
        return

    force_kwargs = parse_force_kwargs(args)
    run_tps(device, gsd_path, sphere_radius, args.dt, force_kwargs)


if __name__ == "__main__":
    main()
