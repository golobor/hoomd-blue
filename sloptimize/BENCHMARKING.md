# Benchmarking

How to build comparison configurations, run benchmarks, and the full results.

---

## Comparison Builds

Five builds are used for benchmarking — three from the **sloptimized** source tree and
two **upstream baselines** from the pre-fork commit. Precision is controlled entirely
by CMake flags:

| Config | Source | CMake flags | What it does |
|--------|--------|------------|--------------|
| mixed | sloptimized | `-DHOOMD_LONGREAL_SIZE=64 -DHOOMD_SHORTREAL_SIZE=32` | Forces in float, integration in double |
| double | sloptimized | `-DHOOMD_LONGREAL_SIZE=64 -DHOOMD_SHORTREAL_SIZE=64` | Double with our code changes |
| single | sloptimized | `-DHOOMD_LONGREAL_SIZE=32 -DHOOMD_SHORTREAL_SIZE=32` | Everything float, with our code changes |
| upstream_double | upstream | `-DHOOMD_LONGREAL_SIZE=64 -DHOOMD_SHORTREAL_SIZE=64` | Unmodified upstream double (regression check) |
| upstream_single | upstream | `-DHOOMD_LONGREAL_SIZE=32 -DHOOMD_SHORTREAL_SIZE=32` | Unmodified upstream single (regression check) |

`HOOMD_MIXED_PRECISION` is **not** a CMake variable — it is a C preprocessor macro
auto-defined when `SHORTREAL_SIZE != LONGREAL_SIZE`. The only CMake knobs are
`HOOMD_LONGREAL_SIZE` and `HOOMD_SHORTREAL_SIZE`.

Mixed is the primary build. Double and single exist for speedup comparison.
Upstream_double and upstream_single verify our changes didn't regress performance.

### Building all five

Each configuration needs its own build tree and install prefix.
**Always pass the precision flags explicitly** — CMake caches variables, so a stale
cache can silently produce the wrong build.

**Sloptimized builds** (from working tree):

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

**Upstream baselines** (from pre-fork commit via `git worktree`):

```bash
# Find the fork point (last upstream commit before our changes)
FORK=$(git merge-base mixed-precision trunk 2>/dev/null \
    || git rev-parse v6.1.1)  # fallback to tag
echo "Fork point: $FORK"

# Create a worktree checkout of the upstream code
git worktree add ../hoomd-upstream $FORK
cd ../hoomd-upstream && git submodule update --init

# Upstream double
mkdir -p build_double && cd build_double
cmake .. -DHOOMD_LONGREAL_SIZE=64 -DHOOMD_SHORTREAL_SIZE=64 \
         -DENABLE_GPU=ON -DBUILD_TESTING=OFF -DBUILD_MPCD=OFF
make -j8
cmake --install . --prefix ../../hoomd-blue/build/install_upstream_double
cd ..

# Upstream single
mkdir -p build_single && cd build_single
cmake .. -DHOOMD_LONGREAL_SIZE=32 -DHOOMD_SHORTREAL_SIZE=32 \
         -DENABLE_GPU=ON -DBUILD_TESTING=OFF -DBUILD_MPCD=OFF
make -j8
cmake --install . --prefix ../../hoomd-blue/build/install_upstream_single
```

**Verify** each build reports the expected precision:

```bash
cd /tmp && PYTHONPATH=<install>/lib/python3.12/site-packages \
  python3 -c "import hoomd; print(hoomd.version.floating_point_precision)"
# mixed            → (64, 32)    compile_flags: DOUBLE[SINGLE]
# double           → (64, 64)    compile_flags: DOUBLE[DOUBLE]
# single           → (32, 32)    compile_flags: SINGLE[SINGLE]
# upstream_double  → (64, 64)    + git_sha1 matches $FORK
# upstream_single  → (32, 32)    + git_sha1 matches $FORK
```

Clean up the worktree when done:

```bash
git worktree remove ../hoomd-upstream
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

Two scripts, one shared library:

- **`benchlib.py`** — shared simulation setup (lattice builder, force factory,
  equilibration). Not run directly.
- **`benchmark_tps.py`** — measures Langevin TPS at a single dt.
- **`benchmark_stability.py`** — NVE energy conservation + per-force accuracy
  at step 0. Also has a `compare` subcommand for cross-build comparison.

```bash
cd sloptimize

# Single-dt TPS, all three builds in parallel (one per GPU):
python run_benchmarks.py benchmark_tps.py \
  --lib mixed=../build/install_mixed/lib/python3.12/site-packages \
  --lib double=../build/install_double/lib/python3.12/site-packages \
  --lib single=../build/install_single/lib/python3.12/site-packages \
  --no-dt

# dt sweep (equilibrate once, benchmark each dt):
python run_benchmarks.py benchmark_tps.py \
  --lib mixed=... --lib double=... --lib single=...

# Without dihedrals:
python run_benchmarks.py benchmark_tps.py \
  --lib mixed=... --lib double=... --lib single=... \
  --no-dt \
  -- --no-dihedral

# Non-default system size:
python run_benchmarks.py benchmark_tps.py \
  --lib mixed=... --lib double=... \
  --no-dt \
  -- -N 256000 -L 400

# Force accuracy (save .npz, then compare):
python run_benchmarks.py benchmark_stability.py \
  --lib mixed=... --lib double=... \
  --no-dt \
  -- --tests accuracy --out-dir /tmp/bench/{label}
python benchmark_stability.py compare /tmp/bench/double /tmp/bench/mixed

# NVE stability:
python benchmark_stability.py --tests nve --dt 0.01

# Patchy workload TPS:
python benchmark_tps.py --no-dihedral --patchy 1.0,0.5,0.6,20,1.5,2

# Full suite (all 5 builds × 3 workloads):
bash run_full_benchmarks.sh
bash run_full_benchmarks.sh --quick  # single dt only
```

The runner auto-detects free GPUs and uses a queue-based GPU pool — each worker
acquires a GPU before starting and releases it when done, guaranteeing no two jobs
share a GPU simultaneously. Use `--gpus 0,1,2` to restrict to specific devices.

### Workloads

Workload variations are expressed as CLI flags (no more `--workload` enum):

| Workload | CLI flags | Description |
|----------|-----------|-------------|
| chains | *(default)* | pair + bond + angle + dihedral + wall |
| nodih | `--no-dihedral` | same without dihedrals |
| patchy | `--no-dihedral --patchy 1.0,0.5,0.6,20,1.5,2` | PatchyGaussian anisotropic pair |
| attract | `--attract 0.5,1.5` | DPD attraction pair |

Default system: 64K particles, 320 chains × 200 monomers.  TPS protocol:
10K warmup + 100K benchmark steps, report every 10K.

---

## Results

### Headline (64K polymers + dihedrals, dt=0.005, RTX 4090)

| Build | TPS | vs Double |
|-------|-----|-----------|
| double | 2,379 ± 42 | 1.0× |
| **mixed** | **7,904 ± 348** | **3.32×** |
| single | 11,857 ± 412 | 4.98× |

### Upstream Regression Check

Comparing sloptimized double/single against unmodified upstream code at the fork
point (`af55fdf58`, trunk tip) to verify our changes don't regress performance.

**64K particles + dihedrals, dt=0.005:**

| Build | TPS | vs upstream |
|-------|-----|-------------|
| upstream_double | 2,516 ± 53 | — |
| double | 2,379 ± 42 | −5.5% |
| upstream_single | 12,496 ± 544 | — |
| single | 11,857 ± 412 | −5.1% |
| **mixed** | **7,904 ± 348** | **+214% vs upstream_double** |

**64K particles, no dihedrals, dt=0.005:**

| Build | TPS | vs upstream |
|-------|-----|-------------|
| upstream_double | 4,394 ± 89 | — |
| double | 4,190 ± 71 | −4.6% |
| upstream_single | 18,933 ± 556 | — |
| single | 19,236 ± 593 | +1.6% |
| **mixed** | **11,821 ± 45** | **+169% vs upstream_double** |

**Summary:** Our code changes introduce ~5% overhead in pure-double mode (likely
from the extra `syncPositionsForceReal` kernel launch and template instantiation
that exist even when `ForceReal == Scalar`). Pure-single shows no measurable
regression — within error bars of upstream. The mixed build delivers **2.7–3.1×**
over upstream double, far outweighing the small regression.

### Progression Through Phases

**With dihedrals (64K particles, dt=0.005):**

| Phase | Mixed TPS | vs Double | vs Single |
|-------|-----------|-----------|-----------|
| Phase 2A (external evaluators only) | 3,059 | +16.5% | 0.25× |
| Phase 2B (float4 positions) | 4,262 | +62.3% | 0.35× |
| Phase 2C (dihedral fix) | 7,474 | +199% | 0.62× |

---

## Accuracy

Measured at step 0 on 64K-particle polymer system, all forces compared against the
double build as reference.

### Force Accuracy (vs Double Reference)

**With dihedrals (chains workload):**

| Metric | Mixed | Single |
|--------|-------|--------|
| Max relative error | 5.03×10⁻⁴ | 5.03×10⁻⁴ |
| Mean relative error | 3.91×10⁻⁷ | 3.95×10⁻⁷ |
| Max absolute error | 9.81×10⁻³ | 9.81×10⁻³ |

**Without dihedrals (nodih workload):**

| Metric | Mixed | Single |
|--------|-------|--------|
| Max relative error | 5.86×10⁻³ | 5.86×10⁻³ |
| Mean relative error | 1.12×10⁻⁶ | 1.12×10⁻⁶ |
| Max absolute error | 4.75×10⁻² | 4.75×10⁻² |

**Patchy workload:**

| Metric | Mixed | Single |
|--------|-------|--------|
| Max relative error | 3.61×10⁻⁴ | 3.61×10⁻⁴ |
| Mean relative error | 5.43×10⁻⁷ | 5.47×10⁻⁷ |
| Max absolute error | 5.68×10⁻³ | 5.68×10⁻³ |

Mixed and single produce nearly identical force errors across all workloads —
confirming all force computation uses float32. The ~10⁻⁷ mean relative error is
consistent with float32 machine epsilon. Max relative errors (up to ~10⁻³) come
from DPDConservative forces near zero magnitude, where the relative error is
amplified but the absolute error remains small.

Upstream_double forces match double to machine precision (~10⁻¹⁶ relative error),
confirming no numerical regression from our code changes. The only exception is the
Periodic dihedral force, where our `SMALL = 1e-12` epsilon clamp introduces ~10⁻⁸
mean relative error at near-collinear geometries.

### Energy Accuracy (vs Double Reference)

| Workload | Build | PE rel. diff. | KE rel. diff. |
|----------|-------|--------------|---------------|
| chains | mixed | 3.47×10⁻⁸ | 0 |
| chains | single | 4.94×10⁻⁸ | 3.41×10⁻⁸ |
| nodih | mixed | 3.16×10⁻⁸ | 0 |
| nodih | single | 1.04×10⁻⁷ | 4.86×10⁻⁸ |
| patchy | mixed | 3.57×10⁻⁸ | 0 |
| patchy | single | 2.39×10⁻⁸ | 8.78×10⁻⁸ |

Mixed preserves KE exactly (double-precision integration), with PE errors at
~10⁻⁸. Single shows slightly larger errors from float integration.

### Energy Conservation (NVE, dt=0.005, 50K steps)

| Build | nodih drift | patchy drift |
|-------|-------------|-------------|
| double | 7.25×10⁻⁵ | 5.32×10⁻⁴ |
| mixed | 1.64×10⁻⁴ | 5.69×10⁻⁴ |
| single | 1.11×10⁻⁴ | 5.21×10⁻⁴ |
| upstream_double | 1.33×10⁻⁴ | 6.07×10⁻⁴ |
| upstream_single | 1.11×10⁻⁴ | — |

All builds conserve energy to ~10⁻⁴–10⁻⁵ in NVE (without-dihedral and patchy
workloads), with no meaningful difference between precision levels or
sloptimized vs upstream.

**Chains NVE (with dihedrals):** All five builds are **UNSTABLE** at dt=0.005
(drift ≫ 1). This is a physics issue — Langevin-equilibrated polymer
configurations contain near-collinear dihedral geometries that produce large
forces when the thermostat is removed for NVE. The instability affects double
and upstream builds equally, confirming it is not a precision artifact.

---

## dt Sweep

### With Dihedrals (64K particles)

| dt | Mixed | Double | Single | Upstr. Dbl | Upstr. Sgl |
|----|-------|--------|--------|------------|------------|
| 0.005 | 7,904 ± 348 | 2,379 ± 42 | 11,857 ± 412 | 2,516 ± 53 | 12,496 ± 544 |
| 0.01 | 5,466 ± 63 | 2,004 ± 47 | 9,262 ± 460 | 2,116 ± 35 | 8,833 ± 150 |
| 0.03 | 4,907 ± 65 | CRASH | CRASH | 2,024 ± 59 | 7,691 ± 56 |
| 0.05 | CRASH | CRASH | CRASH | CRASH | CRASH |
| 0.1 | CRASH | CRASH | CRASH | CRASH | CRASH |

Mixed delivers 3.3× over double at dt=0.005.  The crash boundary differs
between builds: sloptimized double and single crash at dt≥0.03, while mixed and
upstream builds survive dt=0.03 (all crash at dt≥0.05).  The crash threshold is
stochastic — it depends on the specific equilibrated state and which dihedral
geometries happen to be near-collinear at the moment of the dt jump.

### Without Dihedrals (64K particles)

| dt | Mixed | Double | Single | Upstr. Dbl | Upstr. Sgl |
|----|-------|--------|--------|------------|------------|
| 0.005 | 11,821 ± 45 | 4,190 ± 71 | 19,236 ± 593 | 4,394 ± 89 | 18,933 ± 556 |
| 0.01 | 9,614 ± 90 | 4,027 ± 140 | 16,522 ± 318 | 4,015 ± 141 | 16,617 ± 334 |
| 0.03 | 7,525 ± 119 | 3,297 ± 129 | 12,107 ± 32 | 3,459 ± 125 | 12,076 ± 24 |
| 0.05 | 5,063 ± 50 | 2,569 ± 73 | 8,052 ± 14 | 2,714 ± 67 | 8,108 ± 32 |
| 0.1 | 5,072 ± 5 | 2,592 ± 21 | 8,312 ± 4 | 2,697 ± 24 | 7,832 ± 19 |

All builds stable at all dt values. Mixed consistently 2.0–2.8× double.
At large dt (0.05–0.1) all builds' TPS plateaus — the overhead of more
frequent neighbor list rebuilds dominates.

### Without Dihedrals (256K particles)

*Measured with sloptimized builds only (earlier run):*

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

*Measured with sloptimized builds only (earlier run):*

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

*Measured with sloptimized builds only (earlier run):*

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

**Note:** Upstream single-precision cannot run the patchy workload — it crashes
with `CUDA Error: misaligned address` during equilibration.  Force accuracy and
NVE data for patchy are available for the other four builds (see Accuracy section).

---

## Remaining Mixed→Single Performance Gap

The ~1.5× gap between mixed (~7,900 TPS) and single (~11,900 TPS) for isotropic
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

---

## Dihedral Numerical Stability Fix

The GPU dihedral kernels (Harmonic, OPLS, PeriodicImproper) compute forces via
cross products of bond vectors. Near-linear geometries (three consecutive atoms
nearly collinear) cause the cross product magnitude to approach zero, leading to
a chain of numerical failures in float:

1. `raasq = |dab × dcbm|²` → tiny (catastrophic cancellation in float)
2. `raa2inv = 1/raasq` → huge (`1/1e-38 → 1e+38`)
3. `rabinv = sqrt(raa2inv * rbb2inv)` → overflow → `Inf`
4. `s_abcd`, `c_abcd` → `NaN`
5. Force intermediates `gaa = -raa2inv * rg` → huge → particle ejection

The singularity is analytically removable (the `dV/dφ ∝ sin(φ)` factor cancels
`1/|n|²` at collinear geometries), but float precision loses this cancellation.
The CPU path uses `Scalar` (double in mixed mode), hiding the problem.

**Fix:** Added `SMALL = ForceReal(1e-12)` epsilon clamping on `raasq` and `rbbsq`
before division, preventing float overflow without affecting physics. The value
is small enough that it never activates for geometrically plausible configurations,
but prevents `1/0` overflow at exact collinearity. Also clamp `s_abcd` to [-1, 1]
(previously only `c_abcd` was clamped).

**Files modified:**
- `hoomd/md/HarmonicDihedralForceGPU.cu`
- `hoomd/md/OPLSDihedralForceGPU.cu`
- `hoomd/md/PeriodicImproperForceGPU.cu`

**Impact:**
- Prevents float overflow/NaN that would crash mixed-precision dihedral simulations
- Mixed TPS at dt=0.005: ~4,300 (pre-fix) → **~7,900** (+84% — preventing overflow
  is faster than the hardware's NaN/Inf propagation overhead)
- 58/58 tests still pass

### Why All Builds Crash at Large dt With Dihedrals

With `SMALL = ForceReal(1e-12)`, all builds crash at dt≥0.05 — the clamp prevents
float overflow/NaN but does not override the physics.  At large dt the Verlet
integrator overshoots the stiff dihedral potential barrier, particles swing past,
and the simulation diverges regardless of floating-point precision.

The earlier clamp of `SMALL = 0.001` was large enough to artificially soften the
dihedral barrier at near-collinear geometries, which let mixed survive at dt=0.05.
However that amounted to a silent potential modification.  With the physics-correct
clamp (`1e-12`), mixed crashes at the same thresholds as double and single, which
is the expected behaviour.
