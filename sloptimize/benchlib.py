"""Shared benchmark library for HOOMD-blue mixed-precision benchmarks.

Provides the simulation setup functions used by benchmark_tps.py,
benchmark_stability.py, and profile_kernels.py:

- make_lattice_chains() — build polymer chains on a lattice inside a sphere
- make_forces()         — create all HOOMD force objects
- equilibrate_and_save() — FIRE + Langevin equilibration pipeline
- add_common_args()     — add shared CLI flags to any argparse parser
- load_or_equilibrate() — convenience: load saved state or equilibrate
- parse_force_kwargs()  — extract force-related kwargs from parsed args
"""

import gc
import json
import math
import os
import sys
import tempfile
import time

import numpy as np


# ── Output helper ─────────────────────────────────────────────────────────

class TeeWriter:
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


# ── Lattice chain builder ────────────────────────────────────────────────

def make_lattice_chains(n_chains, chain_length, sphere_radius):
    """Place particles on a cubic lattice inside a sphere, linked as chains.

    Uses a snake-like (boustrophedon) traversal so consecutive particles
    in each chain are nearest-neighbors on the lattice, giving bond lengths
    equal to the lattice spacing.

    Returns (positions, bond_groups, angle_groups, dihedral_groups).
    """
    n_particles = n_chains * chain_length

    R_eff = sphere_radius - 1.5
    V_eff = 4.0 / 3.0 * math.pi * R_eff ** 3
    spacing = (V_eff / n_particles) ** (1.0 / 3.0)

    # Find lattice sites inside sphere
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
        raise RuntimeError(
            f"Cannot fit {n_particles} lattice sites (got {n_sites}, "
            f"R_eff={R_eff:.1f}, spacing={spacing:.3f})"
        )

    print(f"  Lattice: {n_sites} sites available, need {n_particles}, "
          f"spacing={spacing:.3f}")

    # Build snake-order traversal
    ordered_positions = []
    for iix in range(-half_n, half_n + 1):
        if (iix + half_n) % 2 == 0:
            y_range = range(-half_n, half_n + 1)
        else:
            y_range = range(half_n, -half_n - 1, -1)
        for iiy in y_range:
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
            f"Not enough snake-ordered sites: "
            f"{len(ordered_positions)} < {n_particles}"
        )

    positions = ordered_positions[:n_particles]

    # Build topology
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
            dihedral_groups.append(
                [offset + j, offset + j + 1, offset + j + 2, offset + j + 3]
            )

    # Report bond length statistics
    bl = np.array(
        [np.linalg.norm(positions[a] - positions[b]) for a, b in bond_groups]
    )
    print(f"  Bond lengths: mean={bl.mean():.3f}, "
          f"min={bl.min():.3f}, max={bl.max():.3f}")

    return positions, bond_groups, angle_groups, dihedral_groups


# ── Patch directors ──────────────────────────────────────────────────────

def make_patch_directors(n_patches):
    """Generate *n_patches* evenly-spaced directors on a sphere."""
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


# ── Force factory ────────────────────────────────────────────────────────

def make_forces(sphere_radius, include_angle=True, include_dihedral=True,
                dpd_A=5.0, attract=None, patchy=None):
    """Create a fresh set of HOOMD force objects.

    Parameters
    ----------
    sphere_radius : float
        Radius of the confining sphere (for wall force).
    include_angle : bool
        Include harmonic angle forces.
    include_dihedral : bool
        Include periodic dihedral forces.
    dpd_A : float
        DPD conservative repulsion strength.
    attract : tuple or None
        If not None, ``(strength, rcut)`` — adds a second DPDConservative
        pair with negative ``A`` for attraction.
    patchy : tuple or None
        If not None, ``(eps, sigma, alpha, omega, rcut, n_patches)`` — adds
        a PatchyGaussian anisotropic pair force with Fibonacci-sphere patches.

    Returns
    -------
    list
        HOOMD force objects ready for ``Integrator(forces=...)``.
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
        directors = make_patch_directors(n_patches)
        plj.directors["A"] = directors
        flist.append(plj)

    flist.append(hbond)

    if include_angle:
        ang = hoomd.md.angle.Harmonic()
        ang.params["backbone"] = dict(k=4.0, t0=2.6)
        flist.append(ang)

    if include_dihedral:
        dih = hoomd.md.dihedral.Periodic()
        dih.params["backbone"] = dict(k=2.0, d=1, n=1, phi0=0.0)
        flist.append(dih)

    sw = hoomd.wall.Sphere(radius=sphere_radius)
    wg = hoomd.md.external.wall.Gaussian(walls=[sw])
    wg.params["A"] = dict(epsilon=5.0, sigma=0.5, r_cut=3.0, r_extrap=0.0)
    flist.append(wg)

    return flist


# ── Equilibration ────────────────────────────────────────────────────────

def equilibrate_and_save(device, n_particles, chain_length,
                         include_angle=True, include_dihedral=True,
                         dpd_A=5.0, attract=None, patchy=None,
                         save_path=None):
    """Build system, FIRE-minimize, Langevin-equilibrate, save to GSD.

    Returns ``(gsd_path, sphere_radius)``.
    """
    import hoomd

    n_chains = n_particles // chain_length
    n_particles = n_chains * chain_length
    n_bonds = n_chains * (chain_length - 1)
    density = 0.3
    volume = n_particles / density
    sphere_radius = (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0)
    box_L = 2.0 * sphere_radius + 10.0

    print(f"System: {n_particles} particles in {n_chains} chains "
          f"of {chain_length}")
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
    if include_angle:
        forces_str += "+angle"
    if include_dihedral:
        forces_str += "+dihedral"
    print(f"Bonds: {n_bonds}")
    print(f"Forces: {forces_str}")
    print()

    # ── Build snapshot ────────────────────────────────────────────────
    print("Generating initial configuration on lattice...")
    positions, bond_groups, angle_groups, dihedral_groups = \
        make_lattice_chains(n_chains, chain_length, sphere_radius)
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
            snap.particles.moment_inertia[:] = [1.0, 1.0, 1.0]
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

    # ── Phase 0: FIRE with pair+bond+wall only (no patchy!) ──────────
    print("\nPhase 0: FIRE energy minimization (pair+bond+wall)...")
    soft_forces = make_forces(
        sphere_radius, include_angle=False, include_dihedral=False,
        dpd_A=dpd_A, attract=attract, patchy=None,
    )
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

    # ── Phase 1: Langevin with soft forces (no patchy) ───────────────
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

    # ── Phase 2: add angle+dihedral+patchy, gentle equilibration ─────
    need_phase2 = include_angle or include_dihedral or (patchy is not None)
    if need_phase2:
        print("Phase 2: adding angle+dihedral+patchy, gentle equilibration...")
        full_forces = make_forces(
            sphere_radius, include_angle=include_angle,
            include_dihedral=include_dihedral,
            dpd_A=dpd_A, attract=attract, patchy=patchy,
        )
        integrator.forces = full_forces
        if patchy is not None:
            integrator.integrate_rotational_dof = True
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

    # ── Phase 3: ramp to production parameters ───────────────────────
    print("Phase 3: ramping to production parameters...")
    integrator.dt = 0.002
    sim.run(5000)
    integrator.dt = 0.005
    langevin.gamma.default = 1.0
    sim.run(2000)
    print("  done (dt=0.005, gamma=1)")

    # ── Save ─────────────────────────────────────────────────────────
    if save_path is None:
        gsd_path = os.path.join(
            tempfile.gettempdir(), f"equil_{os.getpid()}.gsd"
        )
    else:
        gsd_path = save_path
        os.makedirs(os.path.dirname(os.path.abspath(gsd_path)), exist_ok=True)
    hoomd.write.GSD.write(state=sim.state, filename=gsd_path, mode="wb")
    meta_path = gsd_path + ".json"
    with open(meta_path, "w") as f:
        json.dump(dict(
            sphere_radius=sphere_radius, n_particles=n_particles,
            chain_length=chain_length, density=density,
        ), f)
    print(f"  Saved to {gsd_path}")
    print(f"  Metadata to {meta_path}\n")
    del sim
    gc.collect()

    return gsd_path, sphere_radius


# ── CLI helpers ──────────────────────────────────────────────────────────

def add_common_args(parser):
    """Add shared CLI arguments to *parser*.

    System: -g/--gpu, -N/--particles, -L/--chain-length.
    Forces: --no-angle, --no-dihedral, --dpd-A, --attract, --patchy.
    Run:    --dt, --save-state, --load-state, --equilibrate-only, --log.
    """
    parser.add_argument(
        "-g", "--gpu", type=int, default=0,
        help="GPU device index (default: 0)",
    )
    parser.add_argument(
        "-N", "--particles", type=int, default=64000,
        help="Number of particles (default: 64000)",
    )
    parser.add_argument(
        "-L", "--chain-length", type=int, default=200,
        help="Monomers per chain (default: 200)",
    )
    parser.add_argument(
        "--no-angle", action="store_true",
        help="Disable harmonic angle forces",
    )
    parser.add_argument(
        "--no-dihedral", action="store_true",
        help="Disable periodic dihedral forces",
    )
    parser.add_argument(
        "--dpd-A", type=float, default=5.0,
        help="DPD conservative repulsion strength (default: 5.0)",
    )
    parser.add_argument(
        "--attract", default=None,
        help="Add DPD attraction: STRENGTH,RCUT (e.g. 3,1.5)",
    )
    parser.add_argument(
        "--patchy", default=None,
        help="Add PatchyGaussian: EPS,SIGMA,ALPHA,OMEGA,RCUT,NPATCHES "
             "(e.g. 1.0,0.5,0.6,20,1.5,2)",
    )
    parser.add_argument(
        "--dt", type=float, default=0.005,
        help="Integration timestep (default: 0.005)",
    )
    parser.add_argument(
        "--save-state", default=None, metavar="PATH",
        help="Save equilibrated state to this GSD path",
    )
    parser.add_argument(
        "--load-state", default=None, metavar="PATH",
        help="Load equilibrated state from this GSD path "
             "(skip equilibration)",
    )
    parser.add_argument(
        "--equilibrate-only", action="store_true",
        help="Only equilibrate and save state, then exit",
    )
    parser.add_argument(
        "--log", default=None, metavar="FILE",
        help="Tee all output to FILE (line-buffered, tail -f friendly)",
    )


def parse_force_kwargs(args):
    """Extract force-related keyword arguments from parsed *args*.

    Returns a dict suitable for ``make_forces(**kwargs)`` or
    ``equilibrate_and_save(..., **kwargs)``.
    """
    attract = None
    if args.attract:
        attract = tuple(float(x) for x in args.attract.split(","))

    patchy = None
    if args.patchy:
        parts = args.patchy.split(",")
        patchy = (float(parts[0]), float(parts[1]), float(parts[2]),
                  float(parts[3]), float(parts[4]), int(parts[5]))

    return dict(
        include_angle=not args.no_angle,
        include_dihedral=not args.no_dihedral,
        dpd_A=args.dpd_A,
        attract=attract,
        patchy=patchy,
    )


def load_or_equilibrate(args, device):
    """Load saved state or run equilibration.

    Uses ``args.load_state``, ``args.save_state``, ``args.particles``,
    ``args.chain_length``, and force kwargs from ``parse_force_kwargs(args)``.

    Returns ``(gsd_path, sphere_radius)``.
    """
    fkw = parse_force_kwargs(args)

    if args.load_state is not None:
        meta_path = args.load_state + ".json"
        with open(meta_path) as f:
            meta = json.load(f)
        sphere_radius = meta["sphere_radius"]
        gsd_path = args.load_state
        print(f"Loaded state from {gsd_path}")
        print(f"  sphere_radius = {sphere_radius:.2f}\n")
        return gsd_path, sphere_radius

    return equilibrate_and_save(
        device, args.particles, args.chain_length,
        save_path=args.save_state, **fkw,
    )


def json_default(obj):
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)
