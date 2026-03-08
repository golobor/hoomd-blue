#!/usr/bin/env python3
"""Test simulation stability and performance vs timestep (dt).

Part 1 — NVE energy conservation
---------------------------------
After thorough equilibration, run NVE (microcanonical) at each dt.
Total energy E = KE + PE must be conserved; drift/fluctuations reveal
integrator error and floating-point precision limits.

Part 2 — Langevin stability + performance
------------------------------------------
Run Langevin dynamics at each dt.  Measure:
  • Temperature stability — <T> should stay near kT=1.0
  • Performance — TPS at each dt

Usage
-----
    python benchmark_dt_stability.py [gpu_id] [n_particles] [chain_length] [options]

Options:
    --dt DT_VALUES     Comma-separated dt values (default: 0.0003,0.001,0.003,0.005,0.01)
    --no-angle         Disable angle forces
    --no-dihedral      Disable dihedral forces

Defaults: gpu_id=0, n_particles=64000, chain_length=200
"""

import gc
import json
import math
import os
import sys
import tempfile
import time

import numpy as np


class _TeeWriter:
    """Write to both the original stream and a file, flushing after every write."""
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


# ── lattice chain builder ─────────────────────────────────────────────────

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
    bond_groups = []
    angle_groups = []
    dihedral_groups = []
    for c in range(n_chains):
        off = c * chain_length
        for j in range(chain_length - 1):
            bond_groups.append([off + j, off + j + 1])
        for j in range(chain_length - 2):
            angle_groups.append([off + j, off + j + 1, off + j + 2])
        for j in range(chain_length - 3):
            dihedral_groups.append([off + j, off + j + 1, off + j + 2, off + j + 3])
    return positions, bond_groups, angle_groups, dihedral_groups


# ── force factory ─────────────────────────────────────────────────────────

def _make_forces(sphere_radius, include_angle=True, include_dihedral=True,
                 dpd_A=5.0, attract=None):
    """Create a fresh set of force objects (each sim needs its own).

    attract: if not None, a (strength, rcut) tuple.  Adds a second DPD
    Conservative pair with A=-strength and the given rcut (attraction).
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

    # Optional: DPD conservative attraction at longer range
    if attract is not None:
        attr_strength, attr_rcut = attract
        nlist_attr = hoomd.md.nlist.Cell(buffer=0.4)
        dpd_attr = hoomd.md.pair.DPDConservative(
            nlist=nlist_attr, default_r_cut=attr_rcut,
        )
        dpd_attr.params[("A", "A")] = dict(A=-attr_strength)
        forces.append(dpd_attr)

    forces.append(harmonic)

    if include_angle:
        # Harmonic angle (stiffness, curling)
        angle = hoomd.md.angle.Harmonic()
        angle.params["backbone"] = dict(k=4.0, t0=2.6)
        forces.append(angle)

    if include_dihedral:
        # Periodic dihedral (torsional stiffness)
        dihedral = hoomd.md.dihedral.Periodic()
        dihedral.params["backbone"] = dict(k=2.0, d=1, n=1, phi0=0.0)
        forces.append(dihedral)

    # Gaussian wall confinement (smooth)
    wall = hoomd.wall.Sphere(radius=sphere_radius)
    wall_gauss = hoomd.md.external.wall.Gaussian(walls=[wall])
    wall_gauss.params["A"] = dict(
        epsilon=5.0, sigma=0.5, r_cut=3.0, r_extrap=0.0,
    )
    forces.append(wall_gauss)
    return forces


# ── equilibrate & save ────────────────────────────────────────────────────

def equilibrate_and_save(device, n_particles, chain_length, equil_steps=100_000,
                         include_angle=True, include_dihedral=True,
                         dpd_A=5.0, attract=None, save_path=None):
    """Build, FIRE-minimize, Langevin-equilibrate, save to GSD.  Return path + geometry.

    If *save_path* is given, the GSD (and a .json sidecar with geometry) are
    written there; otherwise a temp file is used.
    """
    import hoomd

    n_chains = n_particles // chain_length
    n_particles = n_chains * chain_length
    n_bonds = n_chains * (chain_length - 1)
    density = 0.3
    volume = n_particles / density
    sphere_radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    box_L = 2.0 * sphere_radius + 10.0

    print(f"System: {n_particles} particles, {n_chains} chains of {chain_length}")
    print(f"Sphere R = {sphere_radius:.2f},  density = {density:.2f}")
    print()

    positions, bond_groups, angle_groups, dihedral_groups = make_lattice_chains(
        n_chains, chain_length, sphere_radius,
    )
    n_angles = len(angle_groups)
    n_dihedrals = len(dihedral_groups)

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
        snap.bonds.N = n_bonds
        snap.bonds.types = ["polymer"]
        snap.bonds.group[:] = bond_groups
        snap.bonds.typeid[:] = 0
        snap.angles.N = n_angles
        snap.angles.types = ["backbone"]
        snap.angles.group[:] = angle_groups
        snap.angles.typeid[:] = 0
        snap.dihedrals.N = n_dihedrals
        snap.dihedrals.types = ["backbone"]
        snap.dihedrals.group[:] = dihedral_groups
        snap.dihedrals.typeid[:] = 0

    sim = hoomd.Simulation(device=device, seed=42)
    sim.create_state_from_snapshot(snap)

    # --- Stage 1: FIRE with pair + bond + wall only (no angle/dihedral) ---
    print("FIRE minimization (pair+bond+wall)...")
    soft_forces = _make_forces(sphere_radius, include_angle=False, include_dihedral=False,
                               dpd_A=dpd_A, attract=attract)
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

    # --- Stage 2: Langevin with soft forces to randomize angles ---
    print("Langevin pre-equilibration (pair+bond+wall, 20k steps)...")
    langevin = hoomd.md.methods.Langevin(
        filter=hoomd.filter.All(), kT=1.0, default_gamma=10.0,
    )
    integrator = hoomd.md.Integrator(
        dt=0.002, methods=[langevin], forces=soft_forces,
    )
    sim.operations.integrator = integrator
    sim.run(20_000)

    # --- Stage 3: add angle + dihedral, gentle Langevin ---
    if include_angle or include_dihedral:
        print("Adding angle+dihedral forces, gentle equilibration (50k steps)...")
        full_forces = _make_forces(sphere_radius, include_angle=include_angle,
                                   include_dihedral=include_dihedral,
                                   dpd_A=dpd_A, attract=attract)
        integrator.forces = full_forces
        integrator.dt = 0.001
        sim.run(50_000)
    else:
        full_forces = soft_forces

    # --- Stage 4: full production equilibration ---
    print(f"Production equilibration ({equil_steps:,} steps)...")
    integrator.dt = 0.002
    sim.run(5000)
    integrator.dt = 0.005
    langevin.gamma.default = 1.0
    remaining = equil_steps - 5000
    t0 = time.perf_counter()
    sim.run(remaining)
    elapsed = time.perf_counter() - t0
    print(f"  Done ({equil_steps:,} steps in {elapsed:.1f}s)")

    # Save
    if save_path is None:
        gsd_path = os.path.join(tempfile.gettempdir(), f"equil_{os.getpid()}.gsd")
    else:
        gsd_path = save_path
        os.makedirs(os.path.dirname(os.path.abspath(gsd_path)), exist_ok=True)
    hoomd.write.GSD.write(state=sim.state, filename=gsd_path, mode="wb")
    # Save geometry sidecar so --load-state can recover sphere_radius
    meta_path = gsd_path + ".json"
    with open(meta_path, "w") as f:
        json.dump(dict(sphere_radius=sphere_radius, n_particles=n_particles,
                       chain_length=chain_length, density=density), f)
    print(f"  Saved to {gsd_path}")
    print(f"  Metadata to {meta_path}\n")
    del sim, soft_forces, full_forces
    gc.collect()

    return gsd_path, sphere_radius


# ── Part 1: NVE energy conservation test ─────────────────────────────────

def test_nve_at_dt(device, gsd_path, dt, sphere_radius, nve_steps, log_period,
                   include_angle=True, include_dihedral=True,
                   dpd_A=5.0, attract=None):
    """Run NVE at *dt*, return energy diagnostics."""
    import hoomd

    sim = hoomd.Simulation(device=device, seed=42)
    sim.create_state_from_gsd(filename=gsd_path)

    forces = _make_forces(sphere_radius, include_angle=include_angle,
                          include_dihedral=include_dihedral,
                          dpd_A=dpd_A, attract=attract)
    nve = hoomd.md.methods.ConstantVolume(filter=hoomd.filter.All())
    integrator = hoomd.md.Integrator(dt=dt, methods=[nve], forces=forces)
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
        T_mean=T_mean, T_std=T_std,
    )


def run_nve_scan(device, gsd_path, sphere_radius, dt_values, nve_steps, log_period,
                 include_angle=True, include_dihedral=True,
                 dpd_A=5.0, attract=None):
    """Run NVE part and print table."""
    print("=" * 80)
    print("PART 1: NVE Energy Conservation")
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
        res = test_nve_at_dt(device, gsd_path, dt, sphere_radius, nve_steps, log_period,
                             include_angle=include_angle, include_dihedral=include_dihedral,
                             dpd_A=dpd_A, attract=attract)
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


# ── Part 2: Langevin stability + performance ─────────────────────────────

def test_langevin_at_dt(device, gsd_path, dt, sphere_radius,
                        warmup_steps, bench_steps, log_period,
                        include_angle=True, include_dihedral=True,
                        dpd_A=5.0, attract=None):
    """Run Langevin at *dt*, return T stability and TPS."""
    import hoomd

    sim = hoomd.Simulation(device=device, seed=42)
    sim.create_state_from_gsd(filename=gsd_path)

    forces = _make_forces(sphere_radius, include_angle=include_angle,
                          include_dihedral=include_dihedral,
                          dpd_A=dpd_A, attract=attract)
    langevin = hoomd.md.methods.Langevin(
        filter=hoomd.filter.All(), kT=1.0, default_gamma=1.0,
    )
    integrator = hoomd.md.Integrator(dt=dt, methods=[langevin], forces=forces)
    sim.operations.integrator = integrator

    thermo = hoomd.md.compute.ThermodynamicQuantities(filter=hoomd.filter.All())
    sim.operations.computes.append(thermo)
    device.gpu_error_checking = False

    # ── warmup ────────────────────────────────────────────────────────
    try:
        sim.run(warmup_steps)
    except Exception:
        del sim; gc.collect()
        return dict(status="CRASHED", steps=0)

    # ── benchmark with T + TPS logging ────────────────────────────────
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

    # Temperature should be within ≈5% of kT=1.0 for a stable simulation
    if abs(T_mean - 1.0) > 0.5 or T_std > 0.5:
        status = "UNSTABLE"
    elif abs(T_mean - 1.0) > 0.1 or T_std > 0.1:
        status = "WARN"
    else:
        status = "OK"

    return dict(
        status=status,
        T_mean=T_mean, T_std=T_std,
        tps_mean=tps_mean, tps_std=tps_std,
    )


def run_langevin_scan(device, gsd_path, sphere_radius, dt_values,
                      warmup_steps, bench_steps, log_period,
                      include_angle=True, include_dihedral=True,
                      dpd_A=5.0, attract=None):
    """Run Langevin part and print table."""
    print("=" * 80)
    print("PART 2: Langevin Stability + Performance")
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
        res = test_langevin_at_dt(
            device, gsd_path, dt, sphere_radius,
            warmup_steps, bench_steps, log_period,
            include_angle=include_angle, include_dihedral=include_dihedral,
            dpd_A=dpd_A, attract=attract,
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


# ── main driver ───────────────────────────────────────────────────────────

def run_dt_stability(
    n_particles=64000,
    chain_length=200,
    gpu_id=0,
    dt_values=None,
    equil_steps=100_000,
    nve_steps=50_000,
    nve_log_period=1000,
    langevin_warmup=10_000,
    langevin_bench=50_000,
    langevin_log_period=10_000,
    include_angle=True,
    include_dihedral=True,
    dpd_A=5.0,
    attract=None,
    save_state=None,
    load_state=None,
    equilibrate_only=False,
):
    import hoomd

    if dt_values is None:
        dt_values = [0.0003, 0.001, 0.003, 0.005, 0.01]

    print(f"HOOMD: {hoomd.__file__}")
    forces_str = f"pair(A={dpd_A})+bond+wall"
    if attract is not None:
        forces_str += f"+attract({attract[0]},r={attract[1]})"
    if include_angle:
        forces_str += "+angle"
    if include_dihedral:
        forces_str += "+dihedral"
    print(f"Forces: {forces_str}")
    if not equilibrate_only:
        print(f"dt values: {dt_values}")
    print()

    device = hoomd.device.GPU(gpu_id=gpu_id)

    # ── equilibrate or load ───────────────────────────────────────────
    if load_state is not None:
        # Load pre-equilibrated state
        meta_path = load_state + ".json"
        with open(meta_path) as f:
            meta = json.load(f)
        sphere_radius = meta["sphere_radius"]
        gsd_path = load_state
        print(f"Loaded state from {gsd_path}")
        print(f"  sphere_radius = {sphere_radius:.2f}\n")
        owns_gsd = False
    else:
        gsd_path, sphere_radius = equilibrate_and_save(
            device, n_particles, chain_length, equil_steps,
            include_angle=include_angle, include_dihedral=include_dihedral,
            dpd_A=dpd_A, attract=attract, save_path=save_state,
        )
        owns_gsd = save_state is None  # only clean up temp files

    if equilibrate_only:
        print("Equilibration complete (--equilibrate-only).")
        return {}

    # ── Part 1: NVE ──────────────────────────────────────────────────
    nve_results = run_nve_scan(
        device, gsd_path, sphere_radius, dt_values,
        nve_steps, nve_log_period,
        include_angle=include_angle, include_dihedral=include_dihedral,
        dpd_A=dpd_A, attract=attract,
    )

    # ── Part 2: Langevin ──────────────────────────────────────────────
    langevin_results = run_langevin_scan(
        device, gsd_path, sphere_radius, dt_values,
        langevin_warmup, langevin_bench, langevin_log_period,
        include_angle=include_angle, include_dihedral=include_dihedral,
        dpd_A=dpd_A, attract=attract,
    )

    # Cleanup temp files only
    if owns_gsd:
        try:
            os.unlink(gsd_path)
        except OSError:
            pass

    return dict(nve=nve_results, langevin=langevin_results)


# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse as _ap

    p = _ap.ArgumentParser(description=__doc__,
                           formatter_class=_ap.RawDescriptionHelpFormatter)
    p.add_argument("gpu_id", nargs="?", type=int, default=0)
    p.add_argument("n_particles", nargs="?", type=int, default=64000)
    p.add_argument("chain_length", nargs="?", type=int, default=200)
    p.add_argument("--dt", default=None,
                   help="Comma-separated dt values "
                        "(default: 0.0003,0.001,0.003,0.005,0.01)")
    p.add_argument("--no-angle", action="store_true",
                   help="Disable angle forces")
    p.add_argument("--no-dihedral", action="store_true",
                   help="Disable dihedral forces")
    p.add_argument("--dpd-A", type=float, default=5.0,
                   help="DPD conservative repulsion strength (default: 5.0)")
    p.add_argument("--attract", default=None,
                   help="Add DPD attraction: STRENGTH,RCUT (e.g. 3,1.5)")
    p.add_argument("--save-state", default=None, metavar="PATH",
                   help="Save equilibrated state to this GSD path")
    p.add_argument("--load-state", default=None, metavar="PATH",
                   help="Load equilibrated state from this GSD path "
                        "(skip equilibration)")
    p.add_argument("--equilibrate-only", action="store_true",
                   help="Only equilibrate and save state, then exit")
    p.add_argument("--log", default=None, metavar="FILE",
                   help="Tee all output to FILE (line-buffered, tail -f friendly)")
    a = p.parse_args()

    if a.log:
        sys.stdout = _TeeWriter(sys.stdout, a.log)

    dt_values = [float(x) for x in a.dt.split(",")] if a.dt else None
    attract = tuple(float(x) for x in a.attract.split(",")) if a.attract else None

    run_dt_stability(
        n_particles=a.n_particles,
        chain_length=a.chain_length,
        gpu_id=a.gpu_id,
        dt_values=dt_values,
        include_angle=not a.no_angle,
        include_dihedral=not a.no_dihedral,
        dpd_A=a.dpd_A,
        attract=attract,
        save_state=a.save_state,
        load_state=a.load_state,
        equilibrate_only=a.equilibrate_only,
    )
