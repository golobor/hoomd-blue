#!/usr/bin/env python3
"""NVE energy conservation and force accuracy tests for HOOMD-blue.

Measures:
  - NVE energy drift / fluctuation at a single dt
  - Per-force-type forces and energies at step 0 (saved to .npz)
  - Cross-build force/energy comparison (``compare`` subcommand)

Usage:
    # NVE stability test:
    python benchmark_stability.py --tests nve --dt 0.005

    # Force accuracy (save per-force .npz):
    python benchmark_stability.py --tests accuracy --out-dir /tmp/bench/mixed

    # Both tests:
    python benchmark_stability.py --out-dir /tmp/bench/mixed

    # Compare across builds:
    python benchmark_stability.py compare /tmp/bench/double /tmp/bench/mixed

    # Custom system size:
    python benchmark_stability.py -N 256000 -L 400 --tests nve

    # Equilibrate once, reuse:
    python benchmark_stability.py --equilibrate-only --save-state /tmp/eq.gsd
    python benchmark_stability.py --load-state /tmp/eq.gsd --tests accuracy \
        --out-dir /tmp/bench/mixed
"""

import gc
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from benchlib import (
    TeeWriter, add_common_args, json_default,
    load_or_equilibrate, make_forces, parse_force_kwargs,
)

# ── Constants ─────────────────────────────────────────────────────────────

NVE_STEPS = 50_000
NVE_LOG_PERIOD = 1_000


# ── Force accuracy ───────────────────────────────────────────────────────

def test_force_accuracy(device, gsd_path, sphere_radius, force_kwargs,
                        save_path=None):
    """Compute per-force-type forces and energies at step 0.

    Returns a dict with per-force metrics.  If *save_path* is given,
    saves per-force arrays to an ``.npz`` file for cross-build comparison.
    """
    import hoomd

    sim = hoomd.Simulation(device=device, seed=42)
    sim.create_state_from_gsd(filename=gsd_path)

    forces = make_forces(sphere_radius, **force_kwargs)
    has_patchy = force_kwargs.get("patchy") is not None

    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator = hoomd.md.Integrator(dt=0.005, methods=[nve], forces=forces)
    if has_patchy:
        integrator.integrate_rotational_dof = True
    sim.operations.integrator = integrator
    sim.run(0)

    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)
    sim.run(0)

    # ── Collect per-force data ────────────────────────────────────────
    force_details = {}
    name_counts = {}
    save_arrays = {}

    for force in forces:
        base_name = type(force).__name__
        name_counts[base_name] = name_counts.get(base_name, 0) + 1
        count = name_counts[base_name]
        name = base_name if count == 1 else f"{base_name}_{count}"

        f = np.array(force.forces)
        e = np.array(force.energies)
        f_mag = np.linalg.norm(f, axis=1)

        detail = dict(
            name=name,
            force_mean=float(np.mean(f_mag)),
            force_max=float(np.max(f_mag)),
            force_std=float(np.std(f_mag)),
            energy_mean=float(np.mean(e)),
            energy_std=float(np.std(e)),
            energy_total=float(np.sum(e)),
        )
        force_details[name] = detail
        save_arrays[f"force_{name}"] = f
        save_arrays[f"energy_{name}"] = e

        print(f"  {name:30s}  |F|: mean={detail['force_mean']:.4f} "
              f"max={detail['force_max']:.4f}  "
              f"E: total={detail['energy_total']:.4f}")

    # Total forces
    total_f = np.zeros_like(save_arrays[f"force_{list(force_details)[0]}"])
    for name in force_details:
        total_f += save_arrays[f"force_{name}"]
    save_arrays["total_forces"] = total_f

    # Initial thermo
    KE = thermo.kinetic_energy
    PE = thermo.potential_energy
    TE = KE + PE
    save_arrays["initial_thermo"] = np.array([KE, PE, TE])
    print(f"\n  Thermo: KE={KE:.6f}  PE={PE:.6f}  TE={TE:.6f}")

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        np.savez(save_path, **save_arrays)
        print(f"  Saved forces to {save_path}")

    del sim
    gc.collect()

    return force_details


# ── NVE stability ────────────────────────────────────────────────────────

def test_nve(device, gsd_path, dt, sphere_radius, force_kwargs,
             nve_steps=NVE_STEPS, log_period=NVE_LOG_PERIOD):
    """Run NVE at *dt* and return energy stability metrics."""
    import hoomd

    sim = hoomd.Simulation(device=device, seed=42)
    sim.create_state_from_gsd(filename=gsd_path)

    forces = make_forces(sphere_radius, **force_kwargs)
    has_patchy = force_kwargs.get("patchy") is not None

    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator = hoomd.md.Integrator(dt=dt, methods=[nve], forces=forces)
    if has_patchy:
        integrator.integrate_rotational_dof = True
    sim.operations.integrator = integrator

    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)
    device.gpu_error_checking = False

    try:
        sim.run(0)
    except Exception:
        del sim; gc.collect()
        return dict(status="CRASHED", steps=0)

    E0 = thermo.kinetic_energy + thermo.potential_energy
    energies = [E0]
    temps = [thermo.kinetic_temperature]

    steps_done = 0
    while steps_done < nve_steps:
        chunk = min(log_period, nve_steps - steps_done)
        try:
            sim.run(chunk)
        except Exception:
            del sim; gc.collect()
            return dict(status="CRASHED", steps=steps_done)

        E = thermo.kinetic_energy + thermo.potential_energy
        T = thermo.kinetic_temperature

        if not np.isfinite(E) or (E0 != 0 and abs(E / E0) > 1e6):
            del sim; gc.collect()
            return dict(status="CRASHED", steps=steps_done)

        energies.append(E)
        temps.append(T)
        steps_done += chunk

    del sim; gc.collect()

    E_arr = np.array(energies)
    T_arr = np.array(temps)
    absE0 = abs(E0) if E0 != 0 else 1.0

    drift = (E_arr[-1] - E_arr[0]) / absE0
    rms = (np.std(E_arr) / abs(np.mean(E_arr))
           if np.mean(E_arr) != 0 else float("inf"))
    max_dev = np.max(np.abs(E_arr - E_arr[0])) / absE0
    T_mean, T_std = np.mean(T_arr), np.std(T_arr)

    if max_dev > 0.1:
        status = "UNSTABLE"
    elif max_dev > 0.01:
        status = "DRIFT"
    elif max_dev > 0.001:
        status = "WARN"
    else:
        status = "OK"

    return dict(
        status=status, drift=float(drift), rms_fluctuation=float(rms),
        max_deviation=float(max_dev),
        T_mean=float(T_mean), T_std=float(T_std), E0=float(E0),
    )


def print_nve_result(dt, res, elapsed):
    """Print a single NVE result row."""
    if res["status"] == "CRASHED":
        steps = res.get("steps", 0)
        print(f"  dt={dt:8.4f}  CRASHED @ step {steps}  ({elapsed:.1f}s)")
    else:
        print(f"  dt={dt:8.4f}  drift={res['drift']:+.4e}  "
              f"rms={res['rms_fluctuation']:.4e}  "
              f"max|ΔE/E|={res['max_deviation']:.4e}  "
              f"T={res['T_mean']:.3f}±{res['T_std']:.3f}  "
              f"{res['status']}  ({elapsed:.1f}s)")


# ── Compare ──────────────────────────────────────────────────────────────

def cmd_compare(dirs):
    """Compare force accuracy across builds.

    *dirs*: list of output directories (first is reference).
    """
    labels = [os.path.basename(d.rstrip("/")) for d in dirs]
    ref_label = labels[0]
    ref_dir = dirs[0]

    print("=" * 80)
    print("CROSS-BUILD COMPARISON")
    print(f"  Reference: {ref_label}")
    print(f"  Compare:   {', '.join(labels[1:])}")
    print("=" * 80)

    # Discover available force files in reference dir
    force_files = sorted(
        f for f in os.listdir(ref_dir) if f.startswith("forces_") and f.endswith(".npz")
    )
    if not force_files:
        print("  No force .npz files found in reference directory!")
        return

    for ff in force_files:
        tag = ff.replace("forces_", "").replace(".npz", "")
        ref_path = os.path.join(ref_dir, ff)
        ref = np.load(ref_path)

        print(f"\n{'─' * 80}")
        print(f"  {tag}")
        print(f"{'─' * 80}")

        for label, d in zip(labels[1:], dirs[1:]):
            test_path = os.path.join(d, ff)
            if not os.path.exists(test_path):
                print(f"\n    [{label}] No force data found, skipping.")
                continue

            test = np.load(test_path)
            print(f"\n    {label.upper()} vs {ref_label.upper()}")

            # Per-force comparison
            force_names = sorted(set(
                k.replace("force_", "").replace("energy_", "").replace("virial_", "")
                for k in ref.files
                if k.startswith(("force_", "energy_", "virial_"))
            ))

            for fname in force_names:
                fkey = f"force_{fname}"
                ekey = f"energy_{fname}"
                if fkey not in ref.files or fkey not in test.files:
                    continue

                f_ref = ref[fkey]
                f_test = test[fkey]
                e_ref = ref[ekey]
                e_test = test[ekey]

                f_diff = f_test - f_ref
                f_mag_ref = np.linalg.norm(f_ref, axis=1)
                nonzero = f_mag_ref > 1e-30
                f_rel = np.zeros(len(f_mag_ref))
                f_rel[nonzero] = (
                    np.linalg.norm(f_diff[nonzero], axis=1) / f_mag_ref[nonzero]
                )

                e_diff = e_test - e_ref
                nonzero_e = np.abs(e_ref) > 1e-30
                e_rel = np.zeros(len(e_ref))
                e_rel[nonzero_e] = (
                    np.abs(e_diff[nonzero_e]) / np.abs(e_ref[nonzero_e])
                )

                mean_f_rel = np.mean(f_rel[nonzero]) if nonzero.any() else 0
                mean_e_rel = np.mean(e_rel[nonzero_e]) if nonzero_e.any() else 0

                print(f"      {fname:30s}  "
                      f"F: max_rel={np.max(f_rel):.2e} "
                      f"mean_rel={mean_f_rel:.2e}  "
                      f"E: max_rel={np.max(e_rel):.2e} "
                      f"mean_rel={mean_e_rel:.2e}")

            # Total forces
            if "total_forces" in ref.files and "total_forces" in test.files:
                f_ref = ref["total_forces"]
                f_test = test["total_forces"]
                f_diff = f_test - f_ref
                f_mag_ref = np.linalg.norm(f_ref, axis=1)
                nonzero = f_mag_ref > 1e-30
                f_rel = np.zeros(len(f_mag_ref))
                f_rel[nonzero] = (
                    np.linalg.norm(f_diff[nonzero], axis=1) / f_mag_ref[nonzero]
                )
                mean_f_rel = np.mean(f_rel[nonzero]) if nonzero.any() else 0
                print(f"      {'TOTAL':30s}  "
                      f"F: max_rel={np.max(f_rel):.2e} "
                      f"mean_rel={mean_f_rel:.2e}  "
                      f"max_abs={np.max(np.abs(f_diff)):.2e}")

            # Thermo comparison
            if "initial_thermo" in ref.files and "initial_thermo" in test.files:
                ref_thermo = ref["initial_thermo"]
                test_thermo = test["initial_thermo"]
                for name, ri, ti in zip(["KE", "PE", "TE"],
                                        ref_thermo, test_thermo):
                    diff = ti - ri
                    rel = abs(diff / ri) if abs(ri) > 1e-30 else 0
                    print(f"      {name}: {ref_label}={ri:.8f}  "
                          f"{label}={ti:.8f}  rel_diff={rel:.2e}")

    print(f"\n{'=' * 80}")
    print("Expected relative force errors:")
    print("  mixed vs double:  ~1e-7 (float32 force accumulation)")
    print("  single vs double: ~1e-7 (float32 throughout)")
    print("=" * 80)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    import argparse

    # Detect compare subcommand
    argv = sys.argv[1:]
    if argv and argv[0] == "compare":
        p = argparse.ArgumentParser(
            description="Compare force accuracy across builds.",
        )
        p.add_argument(
            "dirs", nargs="+",
            help="Output directories to compare (first is reference)",
        )
        args = p.parse_args(argv[1:])
        cmd_compare(args.dirs)
        return

    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_args(p)
    p.add_argument(
        "--tests", default="all",
        choices=["all", "nve", "accuracy"],
        help="Which tests to run (default: all)",
    )
    p.add_argument(
        "--out-dir", default=None, metavar="DIR",
        help="Save force .npz and results .json to this directory",
    )
    p.add_argument(
        "--nve-steps", type=int, default=NVE_STEPS,
        help=f"NVE integration steps (default: {NVE_STEPS})",
    )
    p.add_argument(
        "--tag", default="default",
        help="Tag for output filenames (default: 'default'). "
             "Useful for distinguishing workloads in the same --out-dir.",
    )
    args = p.parse_args(argv)

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
    results = {}

    # ── Force accuracy ────────────────────────────────────────────────
    if args.tests in ("all", "accuracy"):
        print("=" * 70)
        print("FORCE ACCURACY (step 0)")
        print("=" * 70)
        forces_save = None
        if args.out_dir:
            os.makedirs(args.out_dir, exist_ok=True)
            forces_save = os.path.join(
                args.out_dir, f"forces_{args.tag}.npz"
            )
        accuracy = test_force_accuracy(
            device, gsd_path, sphere_radius, force_kwargs,
            save_path=forces_save,
        )
        results["accuracy"] = accuracy
        print()

    # ── NVE stability ─────────────────────────────────────────────────
    if args.tests in ("all", "nve"):
        print("=" * 70)
        print(f"NVE ENERGY CONSERVATION (dt={args.dt}, "
              f"{args.nve_steps:,} steps)")
        print("=" * 70)
        t0 = time.perf_counter()
        nve_res = test_nve(
            device, gsd_path, args.dt, sphere_radius, force_kwargs,
            nve_steps=args.nve_steps,
        )
        elapsed = time.perf_counter() - t0
        print_nve_result(args.dt, nve_res, elapsed)
        results["nve"] = {str(args.dt): nve_res}
        print()

    # ── Save results JSON ─────────────────────────────────────────────
    if args.out_dir:
        summary_path = os.path.join(
            args.out_dir, f"results_{args.tag}.json"
        )
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                existing = json.load(f)
            existing.update(results)
            results = existing
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2, default=json_default)
        print(f"Summary saved to {summary_path}\n")


if __name__ == "__main__":
    main()
