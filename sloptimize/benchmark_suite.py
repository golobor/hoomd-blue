#!/usr/bin/env python3
"""Comprehensive benchmark suite: stability, accuracy, and performance.

Tests three workloads across multiple dt values:
  chains  — DPD pair + bonds + angles + periodic dihedrals + wall
  nodih   — same without dihedrals
  patchy  — DPD pair + bonds + angles + PatchyGaussian (anisotropic) + wall

For each workload measures:
  1. Force accuracy  — per-force forces/energies at step 0 (saved to .npz
     for cross-build comparison)
  2. NVE stability   — total energy drift and fluctuation at each dt
  3. Langevin perf   — temperature stability and TPS at each dt

Usage
-----
    # Run single workload:
    python benchmark_suite.py 0 --workload chains --out-dir /tmp/bench/mixed

    # Run all workloads:
    python benchmark_suite.py 0 --workload all --out-dir /tmp/bench/mixed

    # Compare across builds:
    python benchmark_suite.py compare /tmp/bench/double /tmp/bench/mixed

    # Use saved equilibrated state (skips equilibration):
    python benchmark_suite.py 0 --workload chains --out-dir /tmp/bench/mixed \
        --load-dir /tmp/states

    # Equilibrate only (generates states for reuse):
    python benchmark_suite.py 0 --workload all --out-dir /tmp/bench/mixed \
        --equilibrate-only

Designed for use with run_benchmarks.py:
    python run_benchmarks.py benchmark_suite.py \\
        --lib mixed=.../install_mixed/lib/python3.12/site-packages \\
        --lib double=.../install_double/lib/python3.12/site-packages \\
        --no-dt -- --workload all --out-dir /tmp/bench/{label}

Defaults: gpu_id=0, 64K particles, chains of 200.
"""

import gc
import json
import math
import os
import sys
import tempfile
import time

import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────

N_PARTICLES = 64000
CHAIN_LENGTH = 200
DENSITY = 0.3
DPD_A = 5.0
EQUIL_STEPS = 100_000

DT_VALUES = [0.0003, 0.001, 0.003, 0.005, 0.01]
NVE_STEPS = 50_000
NVE_LOG_PERIOD = 1000
LANGEVIN_WARMUP = 10_000
LANGEVIN_BENCH = 50_000
LANGEVIN_LOG_PERIOD = 10_000

PATCHY_PARAMS = (1.0, 0.5, 0.6, 20.0, 1.5, 2)  # eps, sigma, alpha, omega, rcut, npatches

WORKLOAD_DEFS = {
    "chains": dict(include_angle=True, include_dihedral=True, patchy=None),
    "nodih":  dict(include_angle=True, include_dihedral=False, patchy=None),
    "patchy": dict(include_angle=True, include_dihedral=False, patchy=PATCHY_PARAMS),
}


# ── Output helper ─────────────────────────────────────────────────────────

class _TeeWriter:
    """Write to both the original stream and a file."""
    def __init__(self, stream, path):
        self._stream = stream
        self._file = open(path, "w")
    def write(self, s):
        self._stream.write(s)
        self._stream.flush()
        self._file.write(s)
        self._file.flush()
    def flush(self):
        self._stream.flush()
        self._file.flush()
    def close(self):
        self._file.close()


# ── Lattice chain builder ────────────────────────────────────────────────

def make_lattice_chains(n_chains, chain_length, sphere_radius):
    """Place particles on a cubic lattice inside a sphere, linked as chains."""
    n_particles = n_chains * chain_length
    R_eff = sphere_radius - 1.5
    V_eff = 4.0 / 3.0 * math.pi * R_eff**3
    spacing = (V_eff / n_particles) ** (1.0 / 3.0)

    for _ in range(30):
        half_n = int(np.ceil(R_eff / spacing)) + 1
        grid = np.arange(-half_n, half_n + 1)
        ix, iy, iz = np.meshgrid(grid, grid, grid, indexing="ij")
        coords = (
            np.column_stack([ix.ravel(), iy.ravel(), iz.ravel()]).astype(float)
            * spacing
        )
        n_sites = (np.linalg.norm(coords, axis=1) < R_eff).sum()
        if n_sites >= n_particles * 1.05:
            break
        spacing *= 0.97
    else:
        raise RuntimeError(f"Cannot fit {n_particles} sites (got {n_sites})")

    ordered = []
    for iix in range(-half_n, half_n + 1):
        yr = (
            range(-half_n, half_n + 1)
            if (iix + half_n) % 2 == 0
            else range(half_n, -half_n - 1, -1)
        )
        for iiy in yr:
            zr = (
                range(-half_n, half_n + 1)
                if ((iix + half_n) + (iiy + half_n)) % 2 == 0
                else range(half_n, -half_n - 1, -1)
            )
            for iiz in zr:
                pos = np.array([iix, iiy, iiz], dtype=float) * spacing
                if np.linalg.norm(pos) < R_eff:
                    ordered.append(pos)
    ordered = np.array(ordered)
    if len(ordered) < n_particles:
        raise RuntimeError(f"Not enough sites: {len(ordered)} < {n_particles}")

    positions = ordered[:n_particles]
    bond_groups, angle_groups, dihedral_groups = [], [], []
    for c in range(n_chains):
        off = c * chain_length
        for j in range(chain_length - 1):
            bond_groups.append([off + j, off + j + 1])
        for j in range(chain_length - 2):
            angle_groups.append([off + j, off + j + 1, off + j + 2])
        for j in range(chain_length - 3):
            dihedral_groups.append(
                [off + j, off + j + 1, off + j + 2, off + j + 3]
            )
    return positions, bond_groups, angle_groups, dihedral_groups


# ── Patch directors ──────────────────────────────────────────────────────

def _make_patch_directors(n_patches):
    """Generate n_patches evenly-spaced directors on a sphere."""
    if n_patches == 1:
        return [(0, 0, 1)]
    elif n_patches == 2:
        return [(0, 0, 1), (0, 0, -1)]
    elif n_patches == 3:
        return [
            (1, 0, 0),
            (-0.5, math.sqrt(3) / 2, 0),
            (-0.5, -math.sqrt(3) / 2, 0),
        ]
    elif n_patches == 4:
        s = 1 / math.sqrt(3)
        return [(s, s, s), (s, -s, -s), (-s, s, -s), (-s, -s, s)]
    else:
        dirs = []
        golden = (1 + math.sqrt(5)) / 2
        for i in range(n_patches):
            theta = math.acos(1 - 2 * (i + 0.5) / n_patches)
            phi = 2 * math.pi * i / golden
            dirs.append((
                math.sin(theta) * math.cos(phi),
                math.sin(theta) * math.sin(phi),
                math.cos(theta),
            ))
        return dirs


# ── Force factory ────────────────────────────────────────────────────────

def _make_forces(sphere_radius, include_angle=True, include_dihedral=True,
                 dpd_A=5.0, patchy=None):
    """Create force objects for a simulation.

    patchy : tuple or None
        (eps, sigma, alpha, omega, rcut, n_patches) for PatchyGaussian.
    """
    import hoomd

    nlist = hoomd.md.nlist.Cell(buffer=0.4)

    # DPD conservative pair (soft repulsion)
    dpdc = hoomd.md.pair.DPDConservative(nlist=nlist, default_r_cut=1.0)
    dpdc.params[("A", "A")] = dict(A=dpd_A)

    # Harmonic bonds
    harmonic = hoomd.md.bond.Harmonic()
    harmonic.params["polymer"] = dict(k=30.0, r0=0.96)

    forces = [dpdc]

    # PatchyGaussian (anisotropic pair)
    if patchy is not None:
        eps, sig, alpha, omega, rcut, n_patches = patchy
        nl_patchy = hoomd.md.nlist.Cell(buffer=0.4)
        plj = hoomd.md.pair.aniso.PatchyGaussian(
            nlist=nl_patchy, default_r_cut=rcut,
        )
        plj.params[("A", "A")] = dict(
            pair_params=dict(epsilon=eps, sigma=sig),
            envelope_params=dict(alpha=alpha, omega=omega),
        )
        plj.directors["A"] = _make_patch_directors(n_patches)
        forces.append(plj)

    forces.append(harmonic)

    if include_angle:
        angle = hoomd.md.angle.Harmonic()
        angle.params["backbone"] = dict(k=4.0, t0=2.6)
        forces.append(angle)

    if include_dihedral:
        dihedral = hoomd.md.dihedral.Periodic()
        dihedral.params["backbone"] = dict(k=2.0, d=1, n=1, phi0=0.0)
        forces.append(dihedral)

    # Gaussian wall confinement
    wall = hoomd.wall.Sphere(radius=sphere_radius)
    wall_gauss = hoomd.md.external.wall.Gaussian(walls=[wall])
    wall_gauss.params["A"] = dict(
        epsilon=5.0, sigma=0.5, r_cut=3.0, r_extrap=0.0,
    )
    forces.append(wall_gauss)
    return forces


# ── Equilibrate ──────────────────────────────────────────────────────────

def equilibrate_and_save(device, n_particles, chain_length, equil_steps,
                         include_angle=True, include_dihedral=True,
                         dpd_A=5.0, patchy=None, save_path=None):
    """Build, FIRE-minimize, Langevin-equilibrate, save GSD.

    Returns (gsd_path, sphere_radius).
    """
    import hoomd

    n_chains = n_particles // chain_length
    n_particles = n_chains * chain_length
    density = DENSITY
    volume = n_particles / density
    sphere_radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    box_L = 2.0 * sphere_radius + 10.0
    has_patchy = patchy is not None

    print(f"System: {n_particles} particles, {n_chains} chains of {chain_length}")
    print(f"Sphere R = {sphere_radius:.2f},  density = {density:.2f}")
    if has_patchy:
        eps, sig, alpha, omega, rcut, npatch = patchy
        print(f"Patchy: eps={eps} sig={sig} alpha={alpha:.2f} "
              f"omega={omega} rcut={rcut} npatches={npatch}")
    print()

    positions, bond_groups, angle_groups, dihedral_groups = make_lattice_chains(
        n_chains, chain_length, sphere_radius,
    )

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
        if has_patchy:
            snap.particles.moment_inertia[:] = [1.0, 1.0, 1.0]
            rng = np.random.default_rng(seed=123)
            q = rng.standard_normal((n_particles, 4))
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            snap.particles.orientation[:] = q
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

    # --- Stage 1: FIRE with pair+bond+wall only ---
    print("FIRE minimization (pair+bond+wall)...")
    soft_forces = _make_forces(sphere_radius, include_angle=False,
                                include_dihedral=False, dpd_A=dpd_A, patchy=None)
    fire = hoomd.md.minimize.FIRE(
        dt=0.005, force_tol=1e-1, angmom_tol=1e-1, energy_tol=1e-5,
    )
    fire.methods.append(hoomd.md.methods.ConstantVolume(hoomd.filter.All()))
    fire.forces = soft_forces
    sim.operations.integrator = fire
    for s in range(200):
        sim.run(100)
        if fire.converged:
            print(f"  Converged after {(s + 1) * 100} steps")
            break
    else:
        print("  Did not fully converge (continuing)")

    # --- Stage 2: Langevin with soft forces ---
    print("Langevin pre-equilibration (pair+bond+wall, 20k steps)...")
    langevin = hoomd.md.methods.Langevin(
        filter=hoomd.filter.All(), kT=1.0, default_gamma=10.0,
    )
    integrator = hoomd.md.Integrator(
        dt=0.002, methods=[langevin], forces=soft_forces,
    )
    sim.operations.integrator = integrator
    sim.run(20_000)

    # --- Stage 3: add angle + dihedral + patchy ---
    need_ramp = include_angle or include_dihedral or has_patchy
    if need_ramp:
        print("Adding full forces, gentle equilibration...")
        full_forces = _make_forces(sphere_radius, include_angle=include_angle,
                                    include_dihedral=include_dihedral,
                                    dpd_A=dpd_A, patchy=patchy)
        integrator.forces = full_forces
        if has_patchy:
            integrator.integrate_rotational_dof = True
            # Gentle ramp for patchy (strong orientational forces)
            integrator.dt = 0.0001
            langevin.gamma.default = 50.0
            sim.run(5_000)
            print("  sub-phase: dt=0.0001, gamma=50 (5k)")
            integrator.dt = 0.0005
            langevin.gamma.default = 20.0
            sim.run(5_000)
            print("  sub-phase: dt=0.0005, gamma=20 (5k)")
            integrator.dt = 0.001
            langevin.gamma.default = 10.0
            sim.run(10_000)
            print("  sub-phase: dt=0.001, gamma=10 (10k)")
        else:
            integrator.dt = 0.001
            sim.run(50_000)
    else:
        full_forces = soft_forces

    # --- Stage 4: production equilibration ---
    print(f"Production equilibration ({equil_steps:,} steps)...")
    integrator.dt = 0.002
    sim.run(5000)
    integrator.dt = 0.005
    langevin.gamma.default = 1.0
    t0 = time.perf_counter()
    sim.run(equil_steps - 5000)
    elapsed = time.perf_counter() - t0
    print(f"  Done ({equil_steps:,} steps in {elapsed:.1f}s)")

    # Save
    if save_path is None:
        gsd_path = os.path.join(tempfile.gettempdir(),
                                f"equil_{os.getpid()}.gsd")
    else:
        gsd_path = save_path
        os.makedirs(os.path.dirname(os.path.abspath(gsd_path)), exist_ok=True)
    hoomd.write.GSD.write(state=sim.state, filename=gsd_path, mode="wb")
    meta_path = gsd_path + ".json"
    with open(meta_path, "w") as f:
        json.dump(dict(sphere_radius=sphere_radius, n_particles=n_particles,
                       chain_length=chain_length, density=density), f)
    print(f"  Saved to {gsd_path}")
    print(f"  Metadata to {meta_path}\n")
    del sim, soft_forces
    if need_ramp:
        del full_forces
    gc.collect()
    return gsd_path, sphere_radius


# ── Test 1: Force accuracy ───────────────────────────────────────────────

def test_force_accuracy(device, gsd_path, sphere_radius,
                        include_angle=True, include_dihedral=True,
                        dpd_A=5.0, patchy=None, save_path=None):
    """Compute forces at step 0, save per-force-type data to .npz.

    This enables cross-build comparison: run the same function with each
    build's HOOMD, then compare the .npz files.
    """
    import hoomd

    print("=" * 80)
    print("FORCE ACCURACY: computing forces at step 0")
    print("=" * 80)

    sim = hoomd.Simulation(device=device, seed=42)
    sim.create_state_from_gsd(filename=gsd_path)

    forces = _make_forces(sphere_radius, include_angle=include_angle,
                          include_dihedral=include_dihedral,
                          dpd_A=dpd_A, patchy=patchy)
    has_patchy = patchy is not None

    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator = hoomd.md.Integrator(dt=0.005, methods=[nve], forces=forces)
    if has_patchy:
        integrator.integrate_rotational_dof = True
    sim.operations.integrator = integrator

    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)

    sim.run(0)

    # Grab per-particle forces and energies from each force object
    snap = sim.state.get_snapshot()
    positions = np.array(snap.particles.position)

    all_forces = np.zeros((len(positions), 3), dtype=np.float64)
    all_energies = np.zeros(len(positions), dtype=np.float64)
    force_details = {}
    name_counts = {}

    for f in forces:
        fname = type(f).__name__
        # Disambiguate duplicate class names (e.g. two Harmonics: bond + angle)
        name_counts[fname] = name_counts.get(fname, 0) + 1
        if name_counts[fname] > 1:
            fname = f"{fname}_{name_counts[fname]}"
        fi = np.array(f.forces, dtype=np.float64)
        ei = np.array(f.energies, dtype=np.float64)
        vi = np.array(f.virials, dtype=np.float64)
        all_forces += fi
        all_energies += ei
        force_details[f"force_{fname}"] = fi
        force_details[f"energy_{fname}"] = ei
        force_details[f"virial_{fname}"] = vi

        fmag = np.linalg.norm(fi, axis=1)
        print(f"  {fname:30s}  |F|: mean={np.mean(fmag):.4e}  "
              f"max={np.max(fmag):.4e}  E: sum={np.sum(ei):.6f}")

    KE0 = thermo.kinetic_energy
    PE0 = thermo.potential_energy
    TE0 = KE0 + PE0
    print(f"\n  Thermo: KE={KE0:.10f}  PE={PE0:.10f}  TE={TE0:.10f}")

    total_force_mag = np.linalg.norm(all_forces, axis=1)
    print(f"  Total |F|: mean={np.mean(total_force_mag):.4e}  "
          f"max={np.max(total_force_mag):.4e}")

    # Save
    if save_path is not None:
        np.savez(
            save_path,
            positions=positions,
            total_forces=all_forces,
            total_energies=all_energies,
            initial_thermo=np.array([KE0, PE0, TE0]),
            **force_details,
        )
        print(f"  Saved to {save_path}")

    del sim
    gc.collect()
    print()
    return dict(KE=KE0, PE=PE0, TE=TE0)


# ── Test 2: NVE energy conservation ─────────────────────────────────────

def _test_nve_at_dt(device, gsd_path, dt, sphere_radius, nve_steps,
                    log_period, include_angle=True, include_dihedral=True,
                    dpd_A=5.0, patchy=None):
    """Run NVE at *dt*, return energy diagnostics."""
    import hoomd

    sim = hoomd.Simulation(device=device, seed=42)
    sim.create_state_from_gsd(filename=gsd_path)

    forces = _make_forces(sphere_radius, include_angle=include_angle,
                          include_dihedral=include_dihedral,
                          dpd_A=dpd_A, patchy=patchy)
    has_patchy = patchy is not None

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
    rms = np.std(E_arr) / abs(np.mean(E_arr)) if np.mean(E_arr) != 0 else float("inf")
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
        status=status, drift=drift, rms_fluctuation=rms, max_deviation=max_dev,
        T_mean=T_mean, T_std=T_std, E0=E0,
    )


def run_nve_scan(device, gsd_path, sphere_radius, dt_values,
                 nve_steps=NVE_STEPS, log_period=NVE_LOG_PERIOD,
                 include_angle=True, include_dihedral=True,
                 dpd_A=5.0, patchy=None):
    """Run NVE tests at each dt and print table."""
    print("=" * 80)
    print("NVE ENERGY CONSERVATION")
    print(f"  {nve_steps:,} steps per dt,  log every {log_period}")
    print("=" * 80)

    hdr = (
        f"{'dt':>8s} | {'ΔE/E drift':>12s} | {'RMS ΔE/E':>12s} | "
        f"{'max|ΔE/E|':>12s} | {'<T>':>6s} {'±':>1s} {'σ(T)':>6s} | "
        f"{'time':>5s} | Status"
    )
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for dt in dt_values:
        t0 = time.perf_counter()
        res = _test_nve_at_dt(
            device, gsd_path, dt, sphere_radius, nve_steps, log_period,
            include_angle=include_angle, include_dihedral=include_dihedral,
            dpd_A=dpd_A, patchy=patchy,
        )
        elapsed = time.perf_counter() - t0

        if res["status"] == "CRASHED":
            steps = res.get("steps", 0)
            print(f"{dt:8.4f} | {'---':>12s} | {'---':>12s} | "
                  f"{'---':>12s} | {'---':>6s}   {'---':>6s} | "
                  f"{elapsed:5.1f}s | CRASHED @ step {steps}")
        else:
            print(f"{dt:8.4f} | {res['drift']:>12.4e} | "
                  f"{res['rms_fluctuation']:>12.4e} | "
                  f"{res['max_deviation']:>12.4e} | "
                  f"{res['T_mean']:6.3f} ± {res['T_std']:6.3f} | "
                  f"{elapsed:5.1f}s | {res['status']}")
        results[dt] = res
    print()
    return results


# ── Test 3: Langevin stability + performance ─────────────────────────────

def _test_langevin_at_dt(device, gsd_path, dt, sphere_radius,
                         warmup_steps, bench_steps, log_period,
                         include_angle=True, include_dihedral=True,
                         dpd_A=5.0, patchy=None):
    """Run Langevin at *dt*, return T stability and TPS."""
    import hoomd

    sim = hoomd.Simulation(device=device, seed=42)
    sim.create_state_from_gsd(filename=gsd_path)

    forces = _make_forces(sphere_radius, include_angle=include_angle,
                          include_dihedral=include_dihedral,
                          dpd_A=dpd_A, patchy=patchy)
    has_patchy = patchy is not None

    langevin = hoomd.md.methods.Langevin(
        filter=hoomd.filter.All(), kT=1.0, default_gamma=1.0,
    )
    integrator = hoomd.md.Integrator(dt=dt, methods=[langevin], forces=forces)
    if has_patchy:
        integrator.integrate_rotational_dof = True
    sim.operations.integrator = integrator

    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)
    device.gpu_error_checking = False

    # Warmup
    try:
        sim.run(warmup_steps)
    except Exception:
        del sim; gc.collect()
        return dict(status="CRASHED", steps=0)

    # Benchmark
    temps = []
    tps_list = []
    steps_done = 0

    while steps_done < bench_steps:
        chunk = min(log_period, bench_steps - steps_done)
        t0 = time.perf_counter()
        try:
            sim.run(chunk)
        except Exception:
            del sim; gc.collect()
            return dict(status="CRASHED", steps=steps_done)
        wall = time.perf_counter() - t0

        T = thermo.kinetic_temperature
        if not np.isfinite(T) or T > 100:
            del sim; gc.collect()
            return dict(status="CRASHED", steps=steps_done)

        temps.append(T)
        tps_list.append(chunk / wall)
        steps_done += chunk

    del sim; gc.collect()

    T_arr = np.array(temps)
    tps_arr = np.array(tps_list)
    T_mean, T_std = np.mean(T_arr), np.std(T_arr)
    tps_mean, tps_std = np.mean(tps_arr), np.std(tps_arr)

    # With rotational DOF (patchy), HOOMD's kinetic_temperature divides KE
    # by all DOF (translational + rotational), so reported T ≈ kT/2 for
    # 3 translational + 3 rotational DOF.  Adjust expected T accordingly.
    T_expected = 0.5 if has_patchy else 1.0
    if abs(T_mean - T_expected) > 0.5 * T_expected or T_std > 0.5 * T_expected:
        status = "UNSTABLE"
    elif abs(T_mean - T_expected) > 0.1 * T_expected or T_std > 0.1 * T_expected:
        status = "WARN"
    else:
        status = "OK"

    return dict(
        status=status, T_mean=T_mean, T_std=T_std,
        tps_mean=tps_mean, tps_std=tps_std,
    )


def run_langevin_scan(device, gsd_path, sphere_radius, dt_values,
                      warmup_steps=LANGEVIN_WARMUP,
                      bench_steps=LANGEVIN_BENCH,
                      log_period=LANGEVIN_LOG_PERIOD,
                      include_angle=True, include_dihedral=True,
                      dpd_A=5.0, patchy=None):
    """Run Langevin tests at each dt and print table."""
    print("=" * 80)
    print("LANGEVIN STABILITY + PERFORMANCE")
    print(f"  {warmup_steps:,} warmup + {bench_steps:,} bench steps per dt")
    print("=" * 80)

    hdr = (
        f"{'dt':>8s} | {'<T>':>6s} {'±':>1s} {'σ(T)':>6s} | "
        f"{'TPS':>10s} {'±':>1s} {'σ':>8s} | "
        f"{'ns/day':>8s} | {'time':>5s} | Status"
    )
    print(hdr)
    print("-" * len(hdr))

    results = {}
    for dt in dt_values:
        t0 = time.perf_counter()
        res = _test_langevin_at_dt(
            device, gsd_path, dt, sphere_radius,
            warmup_steps, bench_steps, log_period,
            include_angle=include_angle, include_dihedral=include_dihedral,
            dpd_A=dpd_A, patchy=patchy,
        )
        elapsed = time.perf_counter() - t0

        if res["status"] == "CRASHED":
            steps = res.get("steps", 0)
            print(f"{dt:8.4f} | {'---':>6s}   {'---':>6s} | "
                  f"{'---':>10s}   {'---':>8s} | "
                  f"{'---':>8s} | {elapsed:5.1f}s | CRASHED @ step {steps}")
        else:
            ns_day = res["tps_mean"] * dt * 86400 / 1e6
            print(f"{dt:8.4f} | {res['T_mean']:6.3f} ± {res['T_std']:6.3f} | "
                  f"{res['tps_mean']:10.0f} ± {res['tps_std']:8.0f} | "
                  f"{ns_day:8.3f} | {elapsed:5.1f}s | {res['status']}")
        results[dt] = res
    print()
    return results


# ── Run one workload ─────────────────────────────────────────────────────

def run_workload(workload_name, device, dt_values, out_dir=None,
                 load_dir=None, equilibrate_only=False,
                 tests="all",
                 n_particles=N_PARTICLES, chain_length=CHAIN_LENGTH,
                 equil_steps=EQUIL_STEPS):
    """Run all tests for a single workload.

    Returns dict with nve, langevin, and accuracy results.
    """
    import hoomd

    wdef = WORKLOAD_DEFS[workload_name]
    include_angle = wdef["include_angle"]
    include_dihedral = wdef["include_dihedral"]
    patchy = wdef["patchy"]

    forces_str = "pair+bond+wall"
    if include_angle:
        forces_str += "+angle"
    if include_dihedral:
        forces_str += "+dihedral"
    if patchy is not None:
        eps, sig, alpha, omega, rcut, npatch = patchy
        forces_str += f"+patchyGauss({npatch}p)"

    print("\n" + "#" * 80)
    print(f"# WORKLOAD: {workload_name}")
    print(f"#   Forces: {forces_str}")
    print(f"#   dt values: {dt_values}")
    print("#" * 80 + "\n")

    # Determine paths
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        gsd_save = os.path.join(out_dir, f"state_{workload_name}.gsd")
        forces_save = os.path.join(out_dir, f"forces_{workload_name}.npz")
    else:
        gsd_save = None
        forces_save = None

    # Equilibrate or load
    if load_dir is not None:
        gsd_path = os.path.join(load_dir, f"state_{workload_name}.gsd")
        meta_path = gsd_path + ".json"
        if not os.path.exists(gsd_path):
            print(f"  State file not found at {gsd_path}, equilibrating...")
            gsd_path, sphere_radius = equilibrate_and_save(
                device, n_particles, chain_length, equil_steps,
                include_angle=include_angle, include_dihedral=include_dihedral,
                dpd_A=DPD_A, patchy=patchy, save_path=gsd_save,
            )
        else:
            with open(meta_path) as f:
                meta = json.load(f)
            sphere_radius = meta["sphere_radius"]
            print(f"  Loaded state from {gsd_path}")
            print(f"  sphere_radius = {sphere_radius:.2f}\n")
    else:
        gsd_path, sphere_radius = equilibrate_and_save(
            device, n_particles, chain_length, equil_steps,
            include_angle=include_angle, include_dihedral=include_dihedral,
            dpd_A=DPD_A, patchy=patchy, save_path=gsd_save,
        )

    if equilibrate_only:
        print(f"  Equilibration complete for {workload_name}.\n")
        return {}

    results = {}

    # Test 1: force accuracy
    if tests in ("all", "accuracy"):
        accuracy = test_force_accuracy(
            device, gsd_path, sphere_radius,
            include_angle=include_angle, include_dihedral=include_dihedral,
            dpd_A=DPD_A, patchy=patchy, save_path=forces_save,
        )
        results["accuracy"] = accuracy

    # Test 2: NVE stability
    if tests in ("all", "nve"):
        nve = run_nve_scan(
            device, gsd_path, sphere_radius, dt_values,
            include_angle=include_angle, include_dihedral=include_dihedral,
            dpd_A=DPD_A, patchy=patchy,
        )
        results["nve"] = {str(k): v for k, v in nve.items()}

    # Test 3: Langevin performance
    if tests in ("all", "langevin"):
        langevin = run_langevin_scan(
            device, gsd_path, sphere_radius, dt_values,
            include_angle=include_angle, include_dihedral=include_dihedral,
            dpd_A=DPD_A, patchy=patchy,
        )
        results["langevin"] = {str(k): v for k, v in langevin.items()}

    # Save summary JSON (merge with existing data from prior runs)
    if out_dir is not None:
        summary_path = os.path.join(out_dir, f"results_{workload_name}.json")
        if os.path.exists(summary_path):
            with open(summary_path) as f:
                existing = json.load(f)
            existing.update(results)
            results = existing
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2, default=_json_default)
        print(f"  Summary saved to {summary_path}\n")

    return results


def _json_default(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


# ── Compare mode ─────────────────────────────────────────────────────────

def cmd_compare(dirs):
    """Compare force accuracy and stability across builds.

    dirs: list of output directories, one per build.
    Labels are inferred from directory names.
    """
    print("=" * 80)
    print("CROSS-BUILD COMPARISON")
    print("=" * 80)

    labels = [os.path.basename(d.rstrip("/")) for d in dirs]
    ref_label = labels[0]
    ref_dir = dirs[0]

    print(f"  Reference: {ref_label}")
    print(f"  Compare:   {', '.join(labels[1:])}")
    print()

    # Find available workloads
    workloads = []
    for wname in WORKLOAD_DEFS:
        ref_forces = os.path.join(ref_dir, f"forces_{wname}.npz")
        if os.path.exists(ref_forces):
            workloads.append(wname)
    if not workloads:
        print("  No force data found! Run benchmarks first.")
        return

    for wname in workloads:
        ref_forces_path = os.path.join(ref_dir, f"forces_{wname}.npz")
        ref = np.load(ref_forces_path)

        print(f"\n{'─' * 80}")
        print(f"  WORKLOAD: {wname}")
        print(f"{'─' * 80}")

        for i, (label, d) in enumerate(zip(labels[1:], dirs[1:])):
            test_path = os.path.join(d, f"forces_{wname}.npz")
            if not os.path.exists(test_path):
                print(f"\n    [{label}] No force data found, skipping.")
                continue

            test = np.load(test_path)
            print(f"\n    {label.upper()} vs {ref_label.upper()}")

            # Per-force-type comparison
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
                nonzero = f_mag_ref > 1e-30
                f_rel = np.zeros(len(f_mag_ref))
                f_rel[nonzero] = (
                    np.linalg.norm(f_diff[nonzero], axis=1) / f_mag_ref[nonzero]
                )

                # Energy comparison
                e_diff = e_test - e_ref
                nonzero_e = np.abs(e_ref) > 1e-30
                e_rel = np.zeros(len(e_ref))
                e_rel[nonzero_e] = (
                    np.abs(e_diff[nonzero_e]) / np.abs(e_ref[nonzero_e])
                )

                mean_f_rel = np.mean(f_rel[nonzero]) if nonzero.any() else 0
                mean_e_rel = np.mean(e_rel[nonzero_e]) if nonzero_e.any() else 0

                print(f"      {fname:30s}  "
                      f"F: max_rel={np.max(f_rel):.2e} mean_rel={mean_f_rel:.2e}  "
                      f"E: max_rel={np.max(e_rel):.2e} mean_rel={mean_e_rel:.2e}")

            # Total forces
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
                  f"F: max_rel={np.max(f_rel):.2e} mean_rel={mean_f_rel:.2e}  "
                  f"max_abs={np.max(np.abs(f_diff)):.2e}")

            # Thermo comparison
            ref_thermo = ref["initial_thermo"]
            test_thermo = test["initial_thermo"]
            for name, ri, ti in zip(["KE", "PE", "TE"], ref_thermo, test_thermo):
                diff = ti - ri
                rel = abs(diff / ri) if abs(ri) > 1e-30 else 0
                print(f"      {name}: {ref_label}={ri:.8f}  "
                      f"{label}={ti:.8f}  rel_diff={rel:.2e}")

        # NVE/Langevin comparison across builds
        print(f"\n    NVE stability (max|ΔE/E|) — {wname}:")
        hdr = f"      {'dt':>8s}"
        for label in labels:
            hdr += f" | {label:>14s}"
        print(hdr)
        print("      " + "-" * (len(hdr) - 6))

        for dt in DT_VALUES:
            row = f"      {dt:8.4f}"
            for label, d in zip(labels, dirs):
                rpath = os.path.join(d, f"results_{wname}.json")
                if os.path.exists(rpath):
                    with open(rpath) as f:
                        rdata = json.load(f)
                    nve_data = rdata.get("nve", {}).get(str(dt), {})
                    if nve_data.get("status") == "CRASHED":
                        row += f" | {'CRASHED':>14s}"
                    elif "max_deviation" in nve_data:
                        row += f" | {nve_data['max_deviation']:>14.4e}"
                    else:
                        row += f" | {'---':>14s}"
                else:
                    row += f" | {'---':>14s}"
            print(row)

        print(f"\n    Langevin TPS — {wname}:")
        print(hdr)
        print("      " + "-" * (len(hdr) - 6))

        for dt in DT_VALUES:
            row = f"      {dt:8.4f}"
            for label, d in zip(labels, dirs):
                rpath = os.path.join(d, f"results_{wname}.json")
                if os.path.exists(rpath):
                    with open(rpath) as f:
                        rdata = json.load(f)
                    lang_data = rdata.get("langevin", {}).get(str(dt), {})
                    if lang_data.get("status") == "CRASHED":
                        row += f" | {'CRASHED':>14s}"
                    elif "tps_mean" in lang_data:
                        tps = lang_data["tps_mean"]
                        row += f" | {tps:>14.0f}"
                    else:
                        row += f" | {'---':>14s}"
                else:
                    row += f" | {'---':>14s}"
            print(row)

    print(f"\n{'=' * 80}")
    print("Expected relative force errors:")
    print("  mixed vs double:  ~1e-7 (float32 force accumulation)")
    print("  single vs double: ~1e-7 (float32 throughout)")
    print("=" * 80)


# ── Main driver ──────────────────────────────────────────────────────────

def main():
    import argparse as _ap

    # If first arg is a number (gpu_id from run_benchmarks.py), prepend 'run'
    argv = sys.argv[1:]
    if argv and argv[0] not in ("run", "compare", "-h", "--help"):
        argv = ["run"] + argv

    p = _ap.ArgumentParser(
        description=__doc__,
        formatter_class=_ap.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── run subcommand ──
    run_p = sub.add_parser("run", help="Run benchmarks")
    run_p.add_argument("gpu_id", nargs="?", type=int, default=0)
    run_p.add_argument("--workload", default="all",
                       choices=["chains", "nodih", "patchy", "all"],
                       help="Workload to benchmark (default: all)")
    run_p.add_argument("--dt", default=None,
                       help="Comma-separated dt values")
    run_p.add_argument("--out-dir", default=None,
                       help="Output directory for results")
    run_p.add_argument("--load-dir", default=None,
                       help="Load equilibrated states from this directory")
    run_p.add_argument("--equilibrate-only", action="store_true",
                       help="Only equilibrate and save states")
    run_p.add_argument("--tests", default="all",
                       choices=["all", "accuracy", "nve", "langevin"],
                       help="Which tests to run (default: all)")
    run_p.add_argument("--log", default=None, metavar="FILE",
                       help="Tee output to FILE")

    # ── compare subcommand ──
    cmp_p = sub.add_parser("compare", help="Compare results across builds")
    cmp_p.add_argument("dirs", nargs="+",
                       help="Output directories to compare "
                            "(first is reference)")

    args = p.parse_args(argv)

    if args.command == "compare":
        cmd_compare(args.dirs)
        return

    # ── run ──
    if hasattr(args, "log") and args.log:
        sys.stdout = _TeeWriter(sys.stdout, args.log)

    import hoomd

    dt_values = (
        [float(x) for x in args.dt.split(",")]
        if args.dt else DT_VALUES
    )

    print(f"HOOMD: {hoomd.__file__}")
    prec = hoomd.version.floating_point_precision
    print(f"Precision: {prec}")
    print(f"Version: {hoomd.version.version}")
    print()

    device = hoomd.device.GPU(gpu_id=args.gpu_id)

    workloads = (
        list(WORKLOAD_DEFS.keys()) if args.workload == "all"
        else [args.workload]
    )

    all_results = {}
    for wname in workloads:
        res = run_workload(
            wname, device, dt_values,
            out_dir=args.out_dir,
            load_dir=args.load_dir,
            equilibrate_only=args.equilibrate_only,
            tests=args.tests,
        )
        all_results[wname] = res

    # Print combined summary
    if not args.equilibrate_only:
        print("\n" + "=" * 80)
        print("COMBINED SUMMARY")
        print("=" * 80)
        for wname in workloads:
            res = all_results[wname]
            print(f"\n  {wname}:")
            lang = res.get("langevin", {})
            for dt_str, lr in sorted(lang.items()):
                if lr.get("status") == "CRASHED":
                    print(f"    dt={dt_str:>8s}  CRASHED")
                elif "tps_mean" in lr:
                    print(f"    dt={dt_str:>8s}  "
                          f"TPS={lr['tps_mean']:8.0f} ± {lr['tps_std']:5.0f}  "
                          f"T={lr['T_mean']:.3f}±{lr['T_std']:.3f}  "
                          f"{lr['status']}")
        print()


if __name__ == "__main__":
    main()
