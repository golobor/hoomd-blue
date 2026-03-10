# Benchmarking

How to build comparison configurations, run benchmarks, and the full results.

---

## Comparison Builds

All three builds — **mixed**, **double**, **single** — are compiled from **this same
codebase**. Precision is controlled entirely by CMake flags:

| Config | CMake flags | What it does |
|--------|------------|--------------|
| mixed | `-DHOOMD_LONGREAL_SIZE=64 -DHOOMD_SHORTREAL_SIZE=32` | Forces in float, integration in double |
| double | `-DHOOMD_LONGREAL_SIZE=64 -DHOOMD_SHORTREAL_SIZE=64` | Original upstream behavior |
| single | `-DHOOMD_LONGREAL_SIZE=32 -DHOOMD_SHORTREAL_SIZE=32` | Everything float (theoretical max speed) |

`HOOMD_MIXED_PRECISION` is **not** a CMake variable — it is a C preprocessor macro
auto-defined when `SHORTREAL_SIZE != LONGREAL_SIZE`. The only CMake knobs are
`HOOMD_LONGREAL_SIZE` and `HOOMD_SHORTREAL_SIZE`.

The double and single builds exist only for benchmark comparison. Regular users only
need the mixed build.

### Building all three

Each configuration needs its own build tree and install prefix.
**Always pass the precision flags explicitly** — CMake caches variables, so a stale
cache can silently produce the wrong build.

```bash
# Mixed (the default build/ directory)
cd build
cmake .. -DHOOMD_LONGREAL_SIZE=64 -DHOOMD_SHORTREAL_SIZE=32 \
         -DENABLE_GPU=ON
make -j8
cmake --install . --prefix install_mixed

# Double (separate build tree)
mkdir -p build_double && cd build_double
cmake .. -DHOOMD_LONGREAL_SIZE=64 -DHOOMD_SHORTREAL_SIZE=64 \
         -DENABLE_GPU=ON -DBUILD_TESTING=OFF -DBUILD_MPCD=OFF
make -j8
cmake --install . --prefix ../build/install_double

# Single (separate build tree)
mkdir -p build_single && cd build_single
cmake .. -DHOOMD_LONGREAL_SIZE=32 -DHOOMD_SHORTREAL_SIZE=32 \
         -DENABLE_GPU=ON -DBUILD_TESTING=OFF -DBUILD_MPCD=OFF
make -j8
cmake --install . --prefix ../build/install_single
```

**Verify** each build reports the expected precision:

```bash
cd /tmp && PYTHONPATH=<install>/lib/python3.12/site-packages \
  python3 -c "import hoomd; print(hoomd.version.floating_point_precision)"
# mixed  → (64, 32)    compile_flags: DOUBLE[SINGLE]
# double → (64, 64)    compile_flags: DOUBLE[DOUBLE]
# single → (32, 32)    compile_flags: SINGLE[SINGLE]
```

### Switching between builds

```bash
source sloptimize/use_hoomd.sh mixed    # forces=float, integration=double
source sloptimize/use_hoomd.sh double   # original, everything double
source sloptimize/use_hoomd.sh single   # everything float
```

Auto-detects repo root and Python version from the script location.

---

## Running Benchmarks

```bash
cd sloptimize

# Single dt, all three builds in parallel (one per GPU):
python run_benchmarks.py benchmark_chains.py \
  --lib mixed=../build/install_mixed/lib/python3.12/site-packages \
  --lib double=../build/install_double/lib/python3.12/site-packages \
  --lib single=../build/install_single/lib/python3.12/site-packages \
  --no-dt \
  -- 64000 200

# dt sweep (equilibrate once, benchmark each dt):
python run_benchmarks.py benchmark_chains.py \
  --lib mixed=... --lib double=... --lib single=... \
  -- 64000 200

# Without dihedrals:
python run_benchmarks.py benchmark_chains.py \
  --lib mixed=... --lib double=... --lib single=... \
  --no-dt \
  -- 64000 200 --no-dihedral
```

The runner auto-detects free GPUs and uses a queue-based GPU pool — each worker
acquires a GPU before starting and releases it when done, guaranteeing no two jobs
share a GPU simultaneously. Use `--gpus 0,1,2` to restrict to specific devices.

### Workload

`benchmark_chains.py` — 64K particles (or configurable), 320 chains × 200 monomers,
pair(Gaussian A=5) + harmonic bonds + wall(Gaussian) + angle + dihedral, Langevin
integrator, dt=0.005. Protocol: 10K warmup + 100K benchmark steps, report every 10K.

---

## Results

### Headline (64K polymers + dihedrals, dt=0.005, RTX 4090)

| Build | TPS | vs Double |
|-------|-----|-----------|
| double | 2,389 ± 51 | 1.0× |
| **mixed** | **7,563 ± 328** | **3.17×** |
| single | 12,135 ± 351 | 5.08× |

### Progression Through Phases

**With dihedrals (64K particles, dt=0.005):**

| Phase | Mixed TPS | vs Double | vs Single |
|-------|-----------|-----------|-----------|
| Phase 2A (external evaluators only) | 3,059 | +16.5% | 0.25× |
| Phase 2B (float4 positions) | 4,262 | +62.3% | 0.35× |
| Phase 2C (dihedral fix) | 7,474 | +199% | 0.62× |

---

## Accuracy

Measured on 64K-particle polymer system (benchmark_chains.py configuration, 100 steps).

### Force Accuracy (vs Double Reference)

| Metric | Mixed | Single |
|--------|-------|--------|
| Max relative error | 1.82×10⁻⁵ | 1.82×10⁻⁵ |
| Mean relative error | 3.64×10⁻⁷ | 3.64×10⁻⁷ |
| Mean absolute error | 2.87×10⁻⁶ | 2.87×10⁻⁶ |

Mixed and single produce identical force errors — confirming all force computation now
uses float32. The ~10⁻⁷ mean relative error is consistent with float32 machine epsilon.

### Energy Accuracy (vs Double Reference)

| Metric | Mixed | Single |
|--------|-------|--------|
| PE relative difference | 4.85×10⁻⁸ | 4.85×10⁻⁸ |
| KE relative difference | 2.30×10⁻⁷ | 2.30×10⁻⁷ |

### Energy Conservation (100-step drift)

| Build | ΔE/E₀ |
|-------|--------|
| Double | -6.30×10⁻⁵ |
| Mixed | -6.23×10⁻⁵ |
| Single | -6.23×10⁻⁵ |

---

## dt Sweep

### With Dihedrals (64K particles)

| dt | Double | Mixed | Single |
|----|--------|-------|--------|
| 0.005 | 2,389 ± 51 | 7,563 ± 328 | 12,135 ± 351 |
| 0.01 | 1,989 ± 49 | 5,642 ± 101 | 9,400 ± 253 |
| 0.03 | — | 4,916 ± 86 | — |

dt=0.05 and dt=0.1 crash (Langevin dynamics with dihedrals becomes unstable).
Mixed delivers 2.8–3.2× over double across stable dt values.

### Without Dihedrals (64K particles)

| dt | Double | Mixed | Single |
|----|--------|-------|--------|
| 0.005 | 4,229 ± 69 | 11,583 ± 200 | 18,316 ± 41 |
| 0.01 | 3,955 ± 143 | 9,903 ± 225 | 15,735 ± 104 |
| 0.03 | 3,394 ± 125 | 7,553 ± 93 | 11,853 ± 171 |
| 0.05 | 2,627 ± 73 | 5,042 ± 48 | 8,135 ± 85 |
| 0.1 | 2,554 ± 21 | 5,040 ± 16 | 8,229 ± 6 |

All builds stable at all dt values. Mixed consistently 1.9–2.7× double.
At large dt (0.05–0.1) all builds' TPS plateaus — the overhead of more
frequent neighbor list rebuilds dominates.

### Without Dihedrals (256K particles)

| dt | Double | Mixed | Single |
|----|--------|-------|--------|
| 0.005 | 1,140 ± 11 | 3,725 ± 67 | 5,464 ± 166 |
| 0.01 | 1,060 ± 22 | 3,215 ± 61 | 4,741 ± 98 |
| 0.03 | 934 ± 40 | 2,558 ± 29 | 3,728 ± 63 |
| 0.05 | 735 ± 31 | 1,702 ± 22 | 2,514 ± 10 |
| 0.1 | 730 ± 19 | 1,724 ± 12 | 2,450 ± 4 |

All builds stable. Mixed achieves ~3.3× double at dt=0.005 — bandwidth-bound
workloads benefit more at larger system sizes. Mixed-to-single gap narrows to
1.47× (from 1.58× at 64K), consistent with memory bandwidth becoming the
dominant bottleneck.

### With Attraction, No Dihedrals (64K particles, A=-0.5, r_cut=1.5)

Adds a second DPDConservative pair force (attractive, separate neighbor list).

| dt | Double | Mixed | Single | Mixed/Double |
|----|--------|-------|--------|-------------|
| 0.005 | 1,935 ± 83 | 6,917 ± 170 | 10,200 ± 226 | 3.57× |
| 0.01 | 1,796 ± 122 | 5,685 ± 202 | 8,720 ± 173 | 3.17× |
| 0.03 | 1,450 ± 95 | 3,958 ± 90 | 5,835 ± 147 | 2.73× |
| 0.05 | 1,034 ± 53 | 2,393 ± 51 | 3,751 ± 35 | 2.31× |
| 0.1 | 1,012 ± 24 | 2,502 ± 29 | 3,646 ± 17 | 2.47× |

All stable at every dt. The second pair force increases compute intensity,
giving mixed a higher speedup (3.6× vs 2.7× without attraction at dt=0.005) —
more pair-force compute means more float savings to harvest.

### Patchy Particles, No Dihedrals (64K particles, PatchyGaussian)

Uses `AnisoPotentialPairPatchyGauss` with parameters `eps=1.0, sigma=0.5,
alpha=0.6, omega=20, r_cut=1.5, npatches=2`. This exercises the anisotropic
pair kernel which evaluates orientational (quaternion) math.

Includes the cancellation-free `rotmat3(quat)` constructor (`1 − 2c² − 2d²`
diagonals) and PatchEnvelope rotation unified to `rotmat3<ForceReal>` on both
CPU and GPU.

| dt | Double | Mixed | Single | Mixed/Double |
|----|--------|-------|--------|-------------|
| 0.005 | 329 ± 19 | 2,902 ± 42 | 4,892 ± 135 | 8.82× |
| 0.01 | 357 ± 31 | 2,613 ± 62 | 4,507 ± 161 | 7.32× |
| 0.03 | 366 ± 20 | 2,232 ± 56 | 3,756 ± 79 | 6.10× |
| 0.05 | 352 ± 12 | 1,674 ± 16 | 2,709 ± 29 | 4.76× |
| 0.1 | 338 ± 2 | 1,694 ± 9 | 2,702 ± 6 | 5.01× |

**Mixed 5–9× faster than double.** The anisotropic pair evaluator internally
uses `Scalar` (double on mixed, float on single), so mixed does NOT reach
single-precision speed. But the isotropic portions of the kernel (pair force
I/O, minimum-image, neighbor list traversal) all use `ForceReal` (float),
giving a massive speedup over pure double. Double is extremely slow because
the RTX 4090 has a 64:1 FP32:FP64 throughput ratio.

---

## Remaining Mixed→Single Performance Gap

The ~1.6× gap between mixed (~7,500 TPS) and single (~12,000 TPS) for isotropic
pair forces is structural:

1. **Integrator I/O**: Read/write `double4` positions for integration accuracy
2. **Position sync**: `syncPositionsForceReal()` reads double4, writes float4
   (one extra kernel per timestep)
3. **CellList**: Stores `Scalar4` positions — nlist reads double4 neighbors

These are fundamental to the mixed-precision design (double integration is the point)
and cannot be further optimized without sacrificing the precision guarantees.

For anisotropic potentials (patchy), mixed is ~1.7× slower than single because the
evaluator internals (`PairModulator`, `PatchEnvelope` distance/angle math) still use
`Scalar` which is double in mixed but float in single.
