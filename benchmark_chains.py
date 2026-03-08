#!/usr/bin/env python3
"""Benchmark HOOMD-blue: polymer chains in spherical confinement.

Runs a simulation of Harmonic-bonded polymer chains confined in a sphere
with LJ wall interactions. Reports TPS over time to verify convergence.

Equilibration strategy:
  1. Place particles on a cubic lattice inside the sphere (no overlaps).
  2. Connect consecutive lattice sites as chains with Harmonic bonds.
  3. FIRE-minimize to resolve any strain.
  4. Gentle Langevin warmup (small dt, high gamma) → production parameters.

Usage:
    python benchmark_chains.py [gpu_id] [n_particles] [chain_length] [options]

Options:
    --no-angle         Disable angle forces
    --no-dihedral      Disable dihedral forces

Defaults: gpu_id=0, n_particles=64000, chain_length=200
"""

import gc
import json
import sys
import os
import tempfile
import time
import math

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


def make_lattice_chains(n_chains, chain_length, sphere_radius):
    """Place particles on a cubic lattice inside a sphere, linked as chains.

    Uses a snake-like traversal of the lattice to ensure consecutive
    particles in each chain are nearest-neighbors on the lattice, giving
    bond lengths equal to the lattice spacing.

    Returns (positions, bond_groups) with guaranteed:
      - No overlaps (lattice spacing > WCA diameter)
      - All bond lengths = lattice spacing
      - All particles inside sphere with margin
    """
    n_particles = n_chains * chain_length

    # Choose spacing so we get enough sites.
    R_eff = sphere_radius - 1.5
    V_eff = 4.0 / 3.0 * math.pi * R_eff**3

    spacing = (V_eff / n_particles) ** (1.0 / 3.0)

    # Find lattice sites inside sphere
    for _ in range(30):
        half_n = int(np.ceil(R_eff / spacing)) + 1
        # Generate all integer grid points
        grid_range = np.arange(-half_n, half_n + 1)
        ix, iy, iz = np.meshgrid(grid_range, grid_range, grid_range, indexing='ij')
        ix = ix.ravel()
        iy = iy.ravel()
        iz = iz.ravel()
        coords = np.column_stack([ix, iy, iz]).astype(float) * spacing
        dists = np.linalg.norm(coords, axis=1)
        mask = dists < R_eff
        n_sites = mask.sum()

        if n_sites >= n_particles * 1.05:  # need 5% margin for path-finding
            break
        spacing *= 0.97
    else:
        raise RuntimeError(
            f"Could not fit {n_particles} lattice sites in sphere "
            f"(got {n_sites}, R_eff={R_eff:.1f}, spacing={spacing:.3f})"
        )

    print(f"  Lattice: {n_sites} sites available, need {n_particles}, "
          f"spacing={spacing:.3f}")

    # Build snake-order traversal: walk through the 3D lattice in a zigzag
    # pattern so that consecutive sites are always nearest-neighbors.
    # Sort by (ix, iy, iz) with alternating directions.
    side = 2 * half_n + 1

    # We'll walk through the grid in a Hamiltonian-like fashion on the
    # integer grid, then filter only sites inside the sphere.
    # Use the "snake" or "boustrophedon" ordering.
    ordered_positions = []
    for iix in range(-half_n, half_n + 1):
        # Alternate y-direction based on x parity
        if (iix + half_n) % 2 == 0:
            y_range = range(-half_n, half_n + 1)
        else:
            y_range = range(half_n, -half_n - 1, -1)
        for iiy in y_range:
            # Alternate z-direction based on (x+y) parity
            if ((iix + half_n) + (iiy + half_n)) % 2 == 0:
                z_range = range(-half_n, half_n + 1)
            else:
                z_range = range(half_n, -half_n - 1, -1)
            for iiz in z_range:
                pos = np.array([iix, iiy, iiz], dtype=float) * spacing
                if np.linalg.norm(pos) < R_eff:
                    ordered_positions.append(pos)

    ordered_positions = np.array(ordered_positions)
    print(f"  Snake-ordered sites: {len(ordered_positions)}")

    if len(ordered_positions) < n_particles:
        raise RuntimeError(
            f"Not enough snake-ordered sites: {len(ordered_positions)} < {n_particles}"
        )

    # Take first n_particles positions from the ordered list
    positions = ordered_positions[:n_particles]

    # Build bond list: consecutive particles in each chain
    bond_groups = []
    angle_groups = []
    dihedral_groups = []
    for c in range(n_chains):
        offset = c * chain_length
        for j in range(chain_length - 1):
            bond_groups.append([offset + j, offset + j + 1])
        for j in range(chain_length - 2):
            angle_groups.append([offset + j, offset + j + 1, offset + j + 2])
        for j in range(chain_length - 3):
            dihedral_groups.append([offset + j, offset + j + 1, offset + j + 2, offset + j + 3])

    # Report bond length statistics
    bond_lengths = []
    for a, b in bond_groups:
        bond_lengths.append(np.linalg.norm(positions[a] - positions[b]))
    bl = np.array(bond_lengths)
    print(f"  Bond lengths: mean={bl.mean():.3f}, "
          f"min={bl.min():.3f}, max={bl.max():.3f}")

    # Check for inter-chain bonds that span a chain boundary
    # (these should NOT exist since bonds only connect within chains)
    boundary_bonds = 0
    for c in range(n_chains - 1):
        last_of_chain = c * chain_length + chain_length - 1
        first_of_next = (c + 1) * chain_length
        d = np.linalg.norm(positions[last_of_chain] - positions[first_of_next])
        if d > spacing * 1.5:
            boundary_bonds += 1
    # Note: boundary_bonds counts chain-to-chain gaps, which is expected
    # since consecutive chains pick up from where the snake path continues.
    # The bond list only has intra-chain bonds, so this is fine.

    return positions, bond_groups, angle_groups, dihedral_groups


def _make_forces(sphere_radius, with_angle=True, with_dihedral=True,
                 dpd_A=5.0, attract=None, patchy=None):
    """Create a fresh set of force objects.

    Parameters
    ----------
    patchy : tuple or None
        If not None, (epsilon, sigma, alpha, omega, rcut, n_patches).
        Adds a PatchyLJ (Kern-Frenkel-like) anisotropic pair force.
    """
    import hoomd

    nl = hoomd.md.nlist.Cell(buffer=0.4)
    dpd = hoomd.md.pair.DPDConservative(nlist=nl, default_r_cut=1.0)
    dpd.params[("A", "A")] = dict(A=dpd_A)
    hbond = hoomd.md.bond.Harmonic()
    hbond.params["polymer"] = dict(k=30.0, r0=0.96)
    flist = [dpd]
    if attract is not None:
        attr_strength, attr_rcut = attract
        nl_attr = hoomd.md.nlist.Cell(buffer=0.4)
        dpd_attr = hoomd.md.pair.DPDConservative(
            nlist=nl_attr, default_r_cut=attr_rcut,
        )
        dpd_attr.params[("A", "A")] = dict(A=-attr_strength)
        flist.append(dpd_attr)
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
        # Build patch directors: evenly distributed on sphere
        directors = _make_patch_directors(n_patches)
        plj.directors["A"] = directors
        flist.append(plj)
    flist.append(hbond)
    if with_angle:
        ang = hoomd.md.angle.Harmonic()
        ang.params["backbone"] = dict(k=4.0, t0=2.6)
        flist.append(ang)
    if with_dihedral:
        dih = hoomd.md.dihedral.Periodic()
        dih.params["backbone"] = dict(k=2.0, d=1, n=1, phi0=0.0)
        flist.append(dih)
    sw = hoomd.wall.Sphere(radius=sphere_radius)
    wg = hoomd.md.external.wall.Gaussian(walls=[sw])
    wg.params["A"] = dict(epsilon=5.0, sigma=0.5, r_cut=3.0, r_extrap=0.0)
    flist.append(wg)
    return flist


def _make_patch_directors(n_patches):
    """Generate n_patches evenly-spaced directors on a sphere."""
    if n_patches == 1:
        return [(0, 0, 1)]
    elif n_patches == 2:
        return [(0, 0, 1), (0, 0, -1)]
    elif n_patches == 3:
        # Equatorial triangle
        return [
            (1, 0, 0),
            (-0.5, math.sqrt(3)/2, 0),
            (-0.5, -math.sqrt(3)/2, 0),
        ]
    elif n_patches == 4:
        # Tetrahedron
        s = 1 / math.sqrt(3)
        return [(s, s, s), (s, -s, -s), (-s, s, -s), (-s, -s, s)]
    else:
        # Fibonacci sphere
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


def equilibrate_and_save(device, n_particles, chain_length,
                        with_angle=True, with_dihedral=True,
                        dpd_A=5.0, attract=None, patchy=None,
                        save_path=None):
    """Build system, FIRE-minimize, Langevin-equilibrate, save to GSD.

    Returns (gsd_path, sphere_radius).
    """
    import hoomd

    n_chains = n_particles // chain_length
    n_particles = n_chains * chain_length
    n_bonds = n_chains * (chain_length - 1)
    density = 0.3
    volume = n_particles / density
    sphere_radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    box_L = 2.0 * sphere_radius + 10.0

    print(f"System: {n_particles} particles in {n_chains} chains of {chain_length}")
    print(f"Sphere radius: {sphere_radius:.2f}")
    print(f"Box L: {box_L:.2f}")
    print(f"Density: {density:.2f}")
    forces_str = f"pair(A={dpd_A})+bond+wall"
    if attract is not None:
        forces_str += f"+attract({attract[0]},r={attract[1]})"
    if patchy is not None:
        eps, sig, alpha, omega, rcut, npatch = patchy
        forces_str += (f"+patchyGauss(e={eps},s={sig},a={alpha:.2f},"
                       f"w={omega},r={rcut},{npatch}p)")
    if with_angle:
        forces_str += "+angle"
    if with_dihedral:
        forces_str += "+dihedral"
    print(f"Bonds: {n_bonds}")
    print(f"Forces: {forces_str}")
    print()

    # ----- Build snapshot -----
    print("Generating initial configuration on lattice...")
    positions, bond_groups, angle_groups, dihedral_groups = make_lattice_chains(
        n_chains, chain_length, sphere_radius
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
        if patchy is not None:
            # Set moment of inertia for rotational DOF (unit sphere)
            snap.particles.moment_inertia[:] = [1.0, 1.0, 1.0]
            # Random initial orientations (unit quaternions)
            rng = np.random.default_rng(seed=123)
            q = rng.standard_normal((n_particles, 4))
            q /= np.linalg.norm(q, axis=1, keepdims=True)
            snap.particles.orientation[:] = q
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

    # ===== Phase 0: FIRE with pair+bond+wall only (no patchy!) =====
    print("\nPhase 0: FIRE energy minimization (pair+bond+wall)...")
    soft_forces = _make_forces(sphere_radius, with_angle=False, with_dihedral=False,
                               dpd_A=dpd_A, attract=attract, patchy=None)
    fire = hoomd.md.minimize.FIRE(
        dt=0.005, force_tol=1e-1, angmom_tol=1e-1, energy_tol=1e-5,
    )
    fire.methods.append(hoomd.md.methods.ConstantVolume(hoomd.filter.All()))
    fire.forces = soft_forces
    sim.operations.integrator = fire

    for step in range(200):
        sim.run(100)
        if fire.converged:
            print(f"  FIRE converged after {(step + 1) * 100} steps")
            break
    else:
        print("  FIRE did not fully converge (continuing)")

    # ===== Phase 1: Langevin with soft forces (no patchy) =====
    print("Phase 1: Langevin pre-equilibration (pair+bond+wall, 20k steps)...")
    langevin = hoomd.md.methods.Langevin(
        filter=hoomd.filter.All(), kT=1.0, default_gamma=10.0,
    )
    integrator = hoomd.md.Integrator(
        dt=0.002, methods=[langevin], forces=soft_forces,
    )
    sim.operations.integrator = integrator
    sim.run(20_000)
    print("  done")

    # ===== Phase 2: add angle+dihedral+patchy, gentle equilibration =====
    need_phase2 = with_angle or with_dihedral or (patchy is not None)
    if need_phase2:
        print("Phase 2: adding angle+dihedral+patchy, gentle equilibration...")
        full_forces = _make_forces(sphere_radius, with_angle=with_angle,
                                   with_dihedral=with_dihedral,
                                   dpd_A=dpd_A, attract=attract,
                                   patchy=patchy)
        integrator.forces = full_forces
        if patchy is not None:
            integrator.integrate_rotational_dof = True
            # Patchy forces can be very strong with random orientations;
            # start with tiny dt and high damping, ramp up gradually.
            integrator.dt = 0.0001
            langevin.gamma.default = 50.0
            sim.run(5_000)
            print("  sub-phase 2a: dt=0.0001, gamma=50 (5k steps)")
            integrator.dt = 0.0005
            langevin.gamma.default = 20.0
            sim.run(5_000)
            print("  sub-phase 2b: dt=0.0005, gamma=20 (5k steps)")
            integrator.dt = 0.001
            langevin.gamma.default = 10.0
            sim.run(10_000)
            print("  sub-phase 2c: dt=0.001, gamma=10 (10k steps)")
        else:
            integrator.dt = 0.001
            sim.run(20_000)
        print("  done")

    # ===== Phase 3: ramp to production parameters =====
    print("Phase 3: ramping to production parameters...")
    integrator.dt = 0.002
    sim.run(5000)
    integrator.dt = 0.005
    langevin.gamma.default = 1.0
    sim.run(2000)
    print("  done (dt=0.005, gamma=1)")

    # Save
    if save_path is None:
        gsd_path = os.path.join(tempfile.gettempdir(), f"equil_{os.getpid()}.gsd")
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
    del sim
    gc.collect()

    return gsd_path, sphere_radius


def run_benchmark(
    n_particles=64000,
    chain_length=200,
    warmup_steps=10000,
    bench_steps=100000,
    gpu_id=0,
    report_interval=10000,
    with_angle=True,
    with_dihedral=True,
    dpd_A=5.0,
    attract=None,
    patchy=None,
    save_state=None,
    load_state=None,
    equilibrate_only=False,
):
    """Run polymer chain benchmark and return TPS data."""
    import hoomd

    n_chains = n_particles // chain_length
    n_particles = n_chains * chain_length
    density = 0.3
    volume = n_particles / density
    sphere_radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)

    print(f"HOOMD loaded from: {hoomd.__file__}")
    device = hoomd.device.GPU(gpu_id=gpu_id)

    # ── equilibrate or load ───────────────────────────────────────────
    if load_state is not None:
        meta_path = load_state + ".json"
        with open(meta_path) as f:
            meta = json.load(f)
        sphere_radius = meta["sphere_radius"]
        gsd_path = load_state
        owns_gsd = False
        print(f"Loaded state from {gsd_path}")
        print(f"  sphere_radius = {sphere_radius:.2f}\n")
    else:
        gsd_path, sphere_radius = equilibrate_and_save(
            device, n_particles, chain_length,
            with_angle=with_angle, with_dihedral=with_dihedral,
            dpd_A=dpd_A, attract=attract, patchy=patchy,
            save_path=save_state,
        )
        owns_gsd = save_state is None

    if equilibrate_only:
        print("Equilibration complete (--equilibrate-only).")
        return {}

    # ── benchmark from saved state ────────────────────────────────────
    sim = hoomd.Simulation(device=device, seed=42)
    sim.create_state_from_gsd(filename=gsd_path)

    forces = _make_forces(sphere_radius, with_angle=with_angle,
                          with_dihedral=with_dihedral,
                          dpd_A=dpd_A, attract=attract, patchy=patchy)
    langevin = hoomd.md.methods.Langevin(
        filter=hoomd.filter.All(), kT=1.0, default_gamma=1.0,
    )
    integrator = hoomd.md.Integrator(
        dt=0.005, methods=[langevin], forces=forces,
    )
    if patchy is not None:
        integrator.integrate_rotational_dof = True
    sim.operations.integrator = integrator
    device.gpu_error_checking = False

    # ===== Warmup with TPS monitoring =====
    print(f"\nWarming up ({warmup_steps} steps, reporting every "
          f"{report_interval})...")
    warmup_tps = []
    for chunk_start in range(0, warmup_steps, report_interval):
        chunk = min(report_interval, warmup_steps - chunk_start)
        t0 = time.perf_counter()
        sim.run(chunk)
        t1 = time.perf_counter()
        tps = chunk / (t1 - t0)
        warmup_tps.append(tps)
        step = chunk_start + chunk
        print(f"  warmup step {step:>6d}/{warmup_steps}: {tps:.1f} TPS")

    print()

    # ===== Benchmark with TPS monitoring =====
    print(f"Benchmarking ({bench_steps} steps, reporting every "
          f"{report_interval})...")
    bench_tps = []
    for chunk_start in range(0, bench_steps, report_interval):
        chunk = min(report_interval, bench_steps - chunk_start)
        t0 = time.perf_counter()
        sim.run(chunk)
        t1 = time.perf_counter()
        tps = chunk / (t1 - t0)
        bench_tps.append(tps)
        step = chunk_start + chunk
        print(f"  bench  step {step:>6d}/{bench_steps}: {tps:.1f} TPS")

    # Summary
    avg_tps = np.mean(bench_tps)
    std_tps = np.std(bench_tps)
    print(f"\nBenchmark summary:")
    print(f"  Mean TPS: {avg_tps:.1f} +/- {std_tps:.1f}")
    print(f"  ns/day:   {avg_tps * 0.005 * 86400 / 1e6:.3f}")
    print(f"  TPS range: [{min(bench_tps):.1f}, {max(bench_tps):.1f}]")
    print(f"  TPS std/mean: {std_tps / avg_tps * 100:.1f}%")

    # Cleanup temp files only
    if owns_gsd:
        try:
            os.unlink(gsd_path)
            os.unlink(gsd_path + ".json")
        except OSError:
            pass

    return {
        "warmup_tps": warmup_tps,
        "bench_tps": bench_tps,
        "avg_tps": avg_tps,
        "std_tps": std_tps,
    }


if __name__ == "__main__":
    import argparse as _ap

    p = _ap.ArgumentParser(description=__doc__,
                           formatter_class=_ap.RawDescriptionHelpFormatter)
    p.add_argument("gpu_id", nargs="?", type=int, default=0)
    p.add_argument("n_particles", nargs="?", type=int, default=64000)
    p.add_argument("chain_length", nargs="?", type=int, default=200)
    p.add_argument("--no-angle", action="store_true",
                   help="Disable angle forces")
    p.add_argument("--no-dihedral", action="store_true",
                   help="Disable dihedral forces")
    p.add_argument("--dpd-A", type=float, default=5.0,
                   help="DPD conservative repulsion strength (default: 5.0)")
    p.add_argument("--attract", default=None,
                   help="Add DPD attraction: STRENGTH,RCUT (e.g. 3,1.5)")
    p.add_argument("--patchy", default=None,
                   help="Add PatchyGaussian (Kern-Frenkel-like) force with "
                        "rotational DOF. Format: EPS,SIGMA,ALPHA,OMEGA,RCUT,NPATCHES "
                        "(e.g. 1.0,0.5,0.6,20,1.5,2). "
                        "Uses Gaussian pair potential (no hard core). "
                        "ALPHA is patch half-angle in radians.")
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

    attract = tuple(float(x) for x in a.attract.split(",")) if a.attract else None
    if a.patchy:
        parts = a.patchy.split(",")
        patchy = (float(parts[0]), float(parts[1]), float(parts[2]),
                  float(parts[3]), float(parts[4]), int(parts[5]))
    else:
        patchy = None

    results = run_benchmark(
        n_particles=a.n_particles,
        chain_length=a.chain_length,
        gpu_id=a.gpu_id,
        with_angle=not a.no_angle,
        with_dihedral=not a.no_dihedral,
        dpd_A=a.dpd_A,
        attract=attract,
        patchy=patchy,
        save_state=a.save_state,
        load_state=a.load_state,
        equilibrate_only=a.equilibrate_only,
    )
