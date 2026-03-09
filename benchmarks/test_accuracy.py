#!/usr/bin/env python3
"""Accuracy comparison between double, mixed, and single precision HOOMD builds.

Usage:
    # Step 1: Generate equilibrated snapshot (run with any build)
    python test_accuracy.py prepare

    # Step 2: Compute forces with each build (run via wrapper)
    PYTHONPATH=.../install_double/... python test_accuracy.py compute double
    PYTHONPATH=.../install_mixed/...  python test_accuracy.py compute mixed
    PYTHONPATH=.../install_single/... python test_accuracy.py compute single

    # Step 3: Compare results
    python test_accuracy.py compare

    # Or do everything in one go:
    python test_accuracy.py all
"""

import json
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

# ────────── Paths ──────────
BASE = "/groups/goloborodko/user/anton.goloborodko/src/hoomd-blue/build"
INSTALL_PATHS = {
    "double": f"{BASE}/install_double/lib/python3.12/site-packages",
    "mixed": f"{BASE}/install_mixed/lib/python3.12/site-packages",
    "single": f"{BASE}/install_single/lib/python3.12/site-packages",
}
WORK_DIR = "/tmp/accuracy_test"
GSD_PATH = os.path.join(WORK_DIR, "accuracy_state.gsd")
META_PATH = os.path.join(WORK_DIR, "accuracy_meta.json")
GPU_ID = 7

# ────────── System parameters ──────────
N_PARTICLES = 5000
CHAIN_LENGTH = 50
DENSITY = 0.3
DPD_A = 5.0


def make_lattice_chains(n_chains, chain_length, sphere_radius):
    """Place particles on a cubic lattice inside a sphere, linked as chains."""
    n_particles = n_chains * chain_length
    R_eff = sphere_radius - 1.5
    V_eff = 4.0 / 3.0 * math.pi * R_eff**3
    spacing = (V_eff / n_particles) ** (1.0 / 3.0)

    for _ in range(30):
        half_n = int(np.ceil(R_eff / spacing)) + 1
        grid_range = np.arange(-half_n, half_n + 1)
        ix, iy, iz = np.meshgrid(grid_range, grid_range, grid_range, indexing="ij")
        coords = np.column_stack([ix.ravel(), iy.ravel(), iz.ravel()]).astype(float) * spacing
        mask = np.linalg.norm(coords, axis=1) < R_eff
        if mask.sum() >= n_particles * 1.05:
            break
        spacing *= 0.97
    else:
        raise RuntimeError("Could not fit particles")

    ordered_positions = []
    for iix in range(-half_n, half_n + 1):
        y_range = range(-half_n, half_n + 1) if (iix + half_n) % 2 == 0 else range(half_n, -half_n - 1, -1)
        for iiy in y_range:
            if ((iix + half_n) + (iiy + half_n)) % 2 == 0:
                z_range = range(-half_n, half_n + 1)
            else:
                z_range = range(half_n, -half_n - 1, -1)
            for iiz in z_range:
                pos = np.array([iix, iiy, iiz], dtype=float) * spacing
                if np.linalg.norm(pos) < R_eff:
                    ordered_positions.append(pos)

    positions = np.array(ordered_positions[:n_particles])
    bond_groups, angle_groups, dihedral_groups = [], [], []
    for c in range(n_chains):
        o = c * chain_length
        for j in range(chain_length - 1):
            bond_groups.append([o + j, o + j + 1])
        for j in range(chain_length - 2):
            angle_groups.append([o + j, o + j + 1, o + j + 2])
        for j in range(chain_length - 3):
            dihedral_groups.append([o + j, o + j + 1, o + j + 2, o + j + 3])
    return positions, bond_groups, angle_groups, dihedral_groups


def make_forces(sphere_radius):
    """Create forces matching the benchmark: DPD pair + bond + angle + dihedral + wall."""
    import hoomd

    nl = hoomd.md.nlist.Cell(buffer=0.4)
    dpd = hoomd.md.pair.DPDConservative(nlist=nl, default_r_cut=1.0)
    dpd.params[("A", "A")] = dict(A=DPD_A)

    hbond = hoomd.md.bond.Harmonic()
    hbond.params["polymer"] = dict(k=30.0, r0=0.96)

    ang = hoomd.md.angle.Harmonic()
    ang.params["backbone"] = dict(k=4.0, t0=2.6)

    dih = hoomd.md.dihedral.Periodic()
    dih.params["backbone"] = dict(k=2.0, d=1, n=1, phi0=0.0)

    sw = hoomd.wall.Sphere(radius=sphere_radius)
    wg = hoomd.md.external.wall.Gaussian(walls=[sw])
    wg.params["A"] = dict(epsilon=5.0, sigma=0.5, r_cut=3.0, r_extrap=0.0)

    return [dpd, hbond, ang, dih, wg]


def make_forces_soft(sphere_radius):
    """Pair + bond + wall only (no angle/dihedral) for initial minimization."""
    import hoomd

    nl = hoomd.md.nlist.Cell(buffer=0.4)
    dpd = hoomd.md.pair.DPDConservative(nlist=nl, default_r_cut=1.0)
    dpd.params[("A", "A")] = dict(A=DPD_A)

    hbond = hoomd.md.bond.Harmonic()
    hbond.params["polymer"] = dict(k=30.0, r0=0.96)

    sw = hoomd.wall.Sphere(radius=sphere_radius)
    wg = hoomd.md.external.wall.Gaussian(walls=[sw])
    wg.params["A"] = dict(epsilon=5.0, sigma=0.5, r_cut=3.0, r_extrap=0.0)

    return [dpd, hbond, wg]


# ────────────────────────────────────────────────────────────────
# Step 1: PREPARE — build & equilibrate a snapshot, save to GSD
# ────────────────────────────────────────────────────────────────
def cmd_prepare():
    """Equilibrate with double-precision build, save GSD."""
    os.makedirs(WORK_DIR, exist_ok=True)

    # Use double build for preparing the reference state
    sys.path.insert(0, INSTALL_PATHS["double"])
    import hoomd

    n_chains = N_PARTICLES // CHAIN_LENGTH
    n_particles = n_chains * CHAIN_LENGTH
    volume = n_particles / DENSITY
    sphere_radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)

    positions, bond_groups, angle_groups, dihedral_groups = make_lattice_chains(
        n_chains, CHAIN_LENGTH, sphere_radius
    )
    box_L = 2.0 * sphere_radius + 10.0

    device = hoomd.device.GPU()
    snap = hoomd.Snapshot(device.communicator)
    if snap.communicator.rank == 0:
        snap.configuration.box = [box_L, box_L, box_L, 0, 0, 0]
        snap.particles.N = n_particles
        snap.particles.types = ["A"]
        snap.particles.position[:] = positions
        snap.particles.typeid[:] = 0
        snap.particles.body[:] = np.full(n_particles, -1, dtype=np.int32)
        snap.particles.mass[:] = 1.0
        snap.particles.velocity[:] = 0.0
        snap.bonds.N = len(bond_groups)
        snap.bonds.types = ["polymer"]
        snap.bonds.group[:] = bond_groups
        snap.bonds.typeid[:] = 0
        snap.angles.N = len(angle_groups)
        snap.angles.types = ["backbone"]
        snap.angles.group[:] = angle_groups
        snap.angles.typeid[:] = 0
        snap.dihedrals.N = len(dihedral_groups)
        snap.dihedrals.types = ["backbone"]
        snap.dihedrals.group[:] = dihedral_groups
        snap.dihedrals.typeid[:] = 0

    sim = hoomd.Simulation(device=device, seed=42)
    sim.create_state_from_snapshot(snap)

    # Start with pair+bond+wall only (like benchmark does)
    import hoomd
    forces_soft = make_forces_soft(sphere_radius)

    # FIRE minimize with soft forces only
    fire = hoomd.md.minimize.FIRE(dt=0.001, force_tol=1e-1, angmom_tol=1e-1, energy_tol=1e-5)
    fire.methods.append(hoomd.md.methods.ConstantVolume(hoomd.filter.All()))
    fire.forces = forces_soft
    sim.operations.integrator = fire
    for step in range(500):
        sim.run(100)
        if fire.converged:
            print(f"  FIRE converged after {(step + 1) * 100} steps")
            break
    else:
        print("  FIRE did not converge (continuing anyway)")

    # Langevin with soft forces
    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0, default_gamma=10.0)
    integrator = hoomd.md.Integrator(dt=0.002, methods=[langevin], forces=forces_soft)
    sim.operations.integrator = integrator
    sim.run(20_000)
    print("  Soft equilibration done (20k steps, dt=0.002)")

    # Now switch to full forces
    forces_full = make_forces(sphere_radius)
    integrator.forces = forces_full
    integrator.dt = 0.001
    sim.run(20_000)
    print("  Full force equilibration done (20k steps, dt=0.001)")

    # Final ramp
    integrator.dt = 0.005
    langevin.gamma.default = 1.0
    sim.run(5000)
    print("  Production parameters set (dt=0.005, gamma=1)")

    # Save
    hoomd.write.GSD.write(state=sim.state, filename=GSD_PATH, mode="wb")
    with open(META_PATH, "w") as f:
        json.dump(dict(sphere_radius=sphere_radius, n_particles=n_particles, chain_length=CHAIN_LENGTH), f)
    print(f"  Saved to {GSD_PATH}")


# ────────────────────────────────────────────────────────────────
# Step 2: COMPUTE — load GSD, compute forces, run short NVE,
#                    save forces/energies
# ────────────────────────────────────────────────────────────────
def cmd_compute(precision_label):
    """Load GSD, compute forces & energies, run 1000 NVE steps, save results."""
    import hoomd

    with open(META_PATH) as f:
        meta = json.load(f)
    sphere_radius = meta["sphere_radius"]

    device = hoomd.device.GPU()
    sim = hoomd.Simulation(device=device, seed=42)
    sim.create_state_from_gsd(filename=GSD_PATH)

    forces = make_forces(sphere_radius)

    # Use NVE so energy should be conserved (no thermostat noise)
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator = hoomd.md.Integrator(dt=0.005, methods=[nve], forces=forces)
    sim.operations.integrator = integrator

    # Run 0 steps to trigger force computation
    sim.run(0)

    # Grab per-particle forces and energies
    snap = sim.state.get_snapshot()
    positions = np.array(snap.particles.position)

    # Forces from each force object
    all_forces = np.zeros((len(positions), 3), dtype=np.float64)
    all_energies = np.zeros(len(positions), dtype=np.float64)
    all_virials = np.zeros((len(positions), 6), dtype=np.float64)
    force_details = {}

    for i, f in enumerate(forces):
        fname = type(f).__name__
        fi = np.array(f.forces, dtype=np.float64)  # (N, 3)
        ei = np.array(f.energies, dtype=np.float64)  # (N,)
        vi = np.array(f.virials, dtype=np.float64)  # (N, 6)
        all_forces += fi
        all_energies += ei
        all_virials += vi
        force_details[f"force_{fname}"] = fi
        force_details[f"energy_{fname}"] = ei
        force_details[f"virial_{fname}"] = vi

    # Compute total thermo quantities
    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)
    sim.run(0)
    KE0 = thermo.kinetic_energy
    PE0 = thermo.potential_energy
    TE0 = KE0 + PE0
    print(f"  [{precision_label}] Initial: KE={KE0:.10f}, PE={PE0:.10f}, TE={TE0:.10f}")

    # Run short NVE trajectory to measure energy drift
    n_nve_steps = 1000
    energies_ts = []
    for step_i in range(n_nve_steps // 10):
        sim.run(10)
        ke = thermo.kinetic_energy
        pe = thermo.potential_energy
        energies_ts.append([sim.timestep, ke, pe, ke + pe])
    energies_ts = np.array(energies_ts)

    KE_final = energies_ts[-1, 1]
    PE_final = energies_ts[-1, 2]
    TE_final = energies_ts[-1, 3]
    drift = (TE_final - TE0) / abs(TE0) if TE0 != 0 else 0
    print(f"  [{precision_label}] Final:   KE={KE_final:.10f}, PE={PE_final:.10f}, TE={TE_final:.10f}")
    print(f"  [{precision_label}] Relative energy drift over {n_nve_steps} steps: {drift:.2e}")

    # Save
    out_path = os.path.join(WORK_DIR, f"results_{precision_label}.npz")
    np.savez(
        out_path,
        positions=positions,
        total_forces=all_forces,
        total_energies=all_energies,
        total_virials=all_virials,
        energy_timeseries=energies_ts,
        initial_thermo=np.array([KE0, PE0, TE0]),
        **force_details,
    )
    print(f"  [{precision_label}] Saved to {out_path}")


# ────────────────────────────────────────────────────────────────
# Step 3: COMPARE — load all results and report errors
# ────────────────────────────────────────────────────────────────
def cmd_compare():
    """Compare mixed and single results against double reference."""
    ref_path = os.path.join(WORK_DIR, "results_double.npz")
    ref = np.load(ref_path)

    print("=" * 72)
    print("ACCURACY COMPARISON vs DOUBLE PRECISION REFERENCE")
    print("=" * 72)

    for label in ["mixed", "single"]:
        test_path = os.path.join(WORK_DIR, f"results_{label}.npz")
        if not os.path.exists(test_path):
            print(f"\n  [{label}] No results found, skipping.")
            continue

        test = np.load(test_path)
        print(f"\n{'─' * 72}")
        print(f"  {label.upper()} vs DOUBLE")
        print(f"{'─' * 72}")

        # --- Position check (should be identical from same GSD) ---
        pos_diff = np.max(np.abs(test["positions"] - ref["positions"]))
        print(f"  Max position difference (from GSD load): {pos_diff:.2e}")

        # --- Per-force-type comparison ---
        force_names = sorted(set(
            k.replace("force_", "").replace("energy_", "").replace("virial_", "")
            for k in ref.files if k.startswith(("force_", "energy_", "virial_"))
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

            # Force comparison
            f_diff = f_test - f_ref
            f_mag_ref = np.linalg.norm(f_ref, axis=1)
            # Avoid division by zero for particles with zero force
            nonzero = f_mag_ref > 1e-30
            f_rel = np.zeros(len(f_mag_ref))
            f_rel[nonzero] = np.linalg.norm(f_diff[nonzero], axis=1) / f_mag_ref[nonzero]

            # Energy comparison
            e_diff = e_test - e_ref
            nonzero_e = np.abs(e_ref) > 1e-30
            e_rel = np.zeros(len(e_ref))
            e_rel[nonzero_e] = np.abs(e_diff[nonzero_e]) / np.abs(e_ref[nonzero_e])

            print(f"\n  {fname}:")
            print(f"    Forces:   max_abs={np.max(np.abs(f_diff)):.2e}  "
                  f"max_rel={np.max(f_rel):.2e}  "
                  f"mean_rel={np.mean(f_rel[nonzero]):.2e}")
            print(f"    Energies: max_abs={np.max(np.abs(e_diff)):.2e}  "
                  f"max_rel={np.max(e_rel):.2e}  "
                  f"mean_rel={np.mean(e_rel[nonzero_e]):.2e}" if nonzero_e.any()
                  else f"    Energies: all zero")

        # --- Total forces ---
        f_ref = ref["total_forces"]
        f_test = test["total_forces"]
        f_diff = f_test - f_ref
        f_mag_ref = np.linalg.norm(f_ref, axis=1)
        nonzero = f_mag_ref > 1e-30
        f_rel = np.zeros(len(f_mag_ref))
        f_rel[nonzero] = np.linalg.norm(f_diff[nonzero], axis=1) / f_mag_ref[nonzero]
        print(f"\n  TOTAL FORCES:")
        print(f"    max_abs_error  = {np.max(np.abs(f_diff)):.6e}")
        print(f"    max_rel_error  = {np.max(f_rel):.6e}")
        print(f"    mean_rel_error = {np.mean(f_rel[nonzero]):.6e}")
        print(f"    median_rel_err = {np.median(f_rel[nonzero]):.6e}")

        # --- Energy comparison ---
        e_ref = ref["total_energies"]
        e_test = test["total_energies"]
        e_diff = e_test - e_ref
        e_abs_ref = np.abs(e_ref)
        nonzero_e = e_abs_ref > 1e-30
        e_rel = np.zeros(len(e_ref))
        e_rel[nonzero_e] = np.abs(e_diff[nonzero_e]) / e_abs_ref[nonzero_e]
        print(f"\n  TOTAL PER-PARTICLE ENERGIES:")
        print(f"    max_abs_error  = {np.max(np.abs(e_diff)):.6e}")
        print(f"    max_rel_error  = {np.max(e_rel):.6e}")
        print(f"    mean_rel_error = {np.mean(e_rel[nonzero_e]):.6e}" if nonzero_e.any() else "    all zero")

        # --- Thermodynamic totals ---
        ref_thermo = ref["initial_thermo"]  # [KE, PE, TE]
        test_thermo = test["initial_thermo"]
        print(f"\n  THERMODYNAMIC QUANTITIES (initial):")
        for name, ri, ti in zip(["KE", "PE", "TE"], ref_thermo, test_thermo):
            diff = ti - ri
            rel = abs(diff / ri) if abs(ri) > 1e-30 else 0
            print(f"    {name}: ref={ri:.10f}  test={ti:.10f}  "
                  f"diff={diff:.6e}  rel={rel:.6e}")

        # --- Energy drift comparison ---
        ref_ts = ref["energy_timeseries"]
        test_ts = test["energy_timeseries"]
        ref_TE0 = ref_thermo[2]
        test_TE0 = test_thermo[2]

        ref_drift = (ref_ts[-1, 3] - ref_TE0) / abs(ref_TE0) if abs(ref_TE0) > 1e-30 else 0
        test_drift = (test_ts[-1, 3] - test_TE0) / abs(test_TE0) if abs(test_TE0) > 1e-30 else 0
        print(f"\n  NVE ENERGY DRIFT (1000 steps, dt=0.005):")
        print(f"    double: {ref_drift:+.6e}")
        print(f"    {label}:  {test_drift:+.6e}")

        # Energy conservation over time
        ref_TE_fluct = np.std(ref_ts[:, 3]) / abs(ref_TE0) if abs(ref_TE0) > 1e-30 else 0
        test_TE_fluct = np.std(test_ts[:, 3]) / abs(test_TE0) if abs(test_TE0) > 1e-30 else 0
        print(f"    double TE fluctuation (std/|TE0|): {ref_TE_fluct:.6e}")
        print(f"    {label}  TE fluctuation (std/|TE0|): {test_TE_fluct:.6e}")

    print(f"\n{'=' * 72}")
    print("Expected: mixed ~1e-7 relative errors (float32 precision)")
    print("          single ~1e-7 relative errors (float32 throughout)")
    print("=" * 72)


# ────────────────────────────────────────────────────────────────
# ALL — run everything via subprocess
# ────────────────────────────────────────────────────────────────
def cmd_all():
    """Run prepare, compute for all precisions, then compare."""
    os.makedirs(WORK_DIR, exist_ok=True)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
    script = os.path.abspath(__file__)
    conda_prefix = os.path.expanduser("~/miniforge3/envs/main")

    # Step 1: Prepare
    print("\n" + "=" * 72)
    print("STEP 1: PREPARING EQUILIBRATED STATE (using double build)")
    print("=" * 72)
    run_env = env.copy()
    run_env["PYTHONPATH"] = INSTALL_PATHS["double"]
    result = subprocess.run(
        [sys.executable, script, "prepare"],
        env=run_env, capture_output=False
    )
    if result.returncode != 0:
        print("PREPARE failed!")
        return

    # Step 2: Compute for each precision
    for label in ["double", "mixed", "single"]:
        print(f"\n{'=' * 72}")
        print(f"STEP 2: COMPUTING FORCES ({label.upper()})")
        print(f"{'=' * 72}")
        run_env = env.copy()
        run_env["PYTHONPATH"] = INSTALL_PATHS[label]
        result = subprocess.run(
            [sys.executable, script, "compute", label],
            env=run_env, capture_output=False,
        )
        if result.returncode != 0:
            print(f"COMPUTE {label} failed!")

    # Step 3: Compare
    print(f"\n{'=' * 72}")
    print("STEP 3: COMPARING RESULTS")
    print(f"{'=' * 72}")
    # Compare doesn't need hoomd, just numpy
    subprocess.run([sys.executable, script, "compare"], env=env, capture_output=False)


# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "prepare":
        cmd_prepare()
    elif cmd == "compute":
        if len(sys.argv) < 3:
            print("Usage: test_accuracy.py compute <double|mixed|single>")
            sys.exit(1)
        cmd_compute(sys.argv[2])
    elif cmd == "compare":
        cmd_compare()
    elif cmd == "all":
        cmd_all()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)
