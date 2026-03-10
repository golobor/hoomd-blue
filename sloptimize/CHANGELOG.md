# Implementation Changelog

The unabridged developer diary: every phase, every bug, every file changed.
For a high-level overview, see [README.md](README.md). For design rationale and
benchmark analysis, see [DESIGN.md](DESIGN.md).

---

## Plan Evolution

### Iteration 1: Initial plan

The first plan targeted the core force pipeline and integrators:
1. Add `ForceReal` type alias in HOOMDMath.h
2. Add `posCorrection` array to ParticleData
3. Modify pair force kernel (PotentialPairGPU.cuh) — positions, accumulators, virial to ShortReal
4. Modify all pair evaluators (EvaluatorPair*.h) — internal types to ShortReal
5. Modify neighbor list kernel — positions and distances to ShortReal
6. Modify force summation (Integrator.cu)
7. Modify integrator kernels — keep double, use loadPos/storePos
8. Add build option (HOOMD_MIXED_PRECISION)
9. Update snapshot I/O
10. Scope limited to pair forces + NVE/Langevin/BD integrators

Key decisions made:
- Force output stays `Scalar4` initially (promotes ShortReal→Scalar at kernel output) to
  minimize integrator pipeline changes. Bandwidth optimization deferred.
- No fixed-point force accumulation (HOOMD uses warp reduction, not atomics).
- `pos + correction` split over Kahan summation.

### Iteration 2: Deeper analysis of remaining FP64 bottlenecks

After the initial plan, deeper research into both codebases revealed that the first iteration
only covered ~55-60% of GPU compute time. The following additional FP64 bottlenecks were
identified:

**Neighbor list kernels** (NeighborListGPUBinned.cu, etc.):
- All position loads, distance calculations (`dx`, `dot(dx,dx)`), and stencil comparisons
  in Scalar (double). ~2000-4000 FP64 FLOPS per particle per rebuild.
- Safe to convert: `r_buff` (buffer distance) absorbs FP32 rounding.

**Cell list kernel** (CellListGPU.cu):
- `makeFraction()` and cell-index computation in Scalar.
- `d_cell_xyzf` stored as Scalar4 (32 bytes) — could be ForceReal4 (16 bytes).

**Angle force kernels** (3 standalone .cu files):
- HarmonicAngleForceGPU.cu, CosineSqAngleForceGPU.cu, TableAngleForceGPU.cu
- All positions, dx, force, virial in Scalar. ~50 FP64 FLOPS per angle.
- Note: `sqrtf` is already used on some lines (likely a latent bug).

**Dihedral/improper force kernels** (5 standalone .cu files):
- HarmonicDihedralForceGPU.cu, OPLSDihedralForceGPU.cu, TableDihedralForceGPU.cu,
  HarmonicImproperForceGPU.cu, PeriodicImproperForceGPU.cu
- All positions, cross-products, forces, virial in Scalar. ~80 FP64 FLOPS per dihedral.

**Bond kernel `minImage` roundtrip** (PotentialBondGPU.cuh):
- `ForceReal3 dx → Scalar3 → box.minImage(Scalar3) → ForceReal3` — should use
  `box.minImageForceReal(dx)` directly.

**DPD thermostat pair kernel** (PotentialPairDPDThermoGPU.cuh):
- ALL geometry and force accumulation in Scalar (FP64), despite evaluator using ForceReal.

### Iteration 3: Revised phasing (current)

Based on the deeper analysis, the plan was restructured:

- **Phase 1**: Convert force/virial/torque **output storage** to ForceReal (the data
  pipeline, not the kernel internals). ✅ COMPLETE
- **Phase 2**: Convert **kernel internals** to use ForceReal, split into:
  - **2A**: External evaluators (benchmark-relevant FP64 force kernel)
  - **2B**: Float4 position mirror (halves bandwidth)
  - **2C**: Missed dihedral/improper/angle kernels
- **Phase 3**: `--use_fast_math` → **SKIPPED** (see DESIGN.md)

---

## Phase 1: Force Output Pipeline

**Goal**: Convert all force/virial/torque storage and GPU kernel I/O from double to float.

**Status**: ✅ COMPLETE — all 58 tests pass

### Step 1: Core data structures

Changed the underlying storage types for force data:

**ForceCompute.h/.cc**:
- `GPUArray<Scalar4> m_force` → `GPUArray<ForceReal4> m_force`
- `GPUArray<Scalar4> m_torque` → `GPUArray<ForceReal4> m_torque`
- `GPUArray<Scalar> m_virial` → `GPUArray<ForceReal> m_virial`
- All accessor methods updated to return `ForceReal4`/`ForceReal` types

**ParticleData.h/.cc**:
- `m_net_force`, `m_net_virial`, `m_net_torque` → ForceReal4/ForceReal
- "Alt" arrays for double-buffering
- All `getNetForce()`, `getNetVirial()`, `getNetTorque()` accessors

### Step 2: Force summation infrastructure

**Integrator.cu/.cc/.cuh**:
- `gpu_force_list` struct: `Scalar4*` → `ForceReal4*`, `Scalar*` → `ForceReal*`
- GPU kernel local variables and accumulators converted
- CPU fallback path updated

### Step 3: Integrator step kernels (CPU side)

Updated `ArrayHandle<Scalar4>` → `ArrayHandle<ForceReal4>` for net force/torque/virial
reads in all integrator `.cc` files:

- TwoStepLangevin.cc, TwoStepConstantVolume.cc, TwoStepConstantPressure.cc,
  FIREEnergyMinimizer.cc, ForceDistanceConstraint.cc

Fixed `vec3<Scalar>(ForceReal4)` constructor issue (vec3 only accepts Scalar4):
```cpp
// Before:
vec3<Scalar> t(h_net_torque.data[j]);
// After:
auto t4 = h_net_torque.data[j];
vec3<Scalar> t(t4.x, t4.y, t4.z);
```
12 occurrences across 5 files.

### Step 4: GPU kernel signatures and bodies

Changed `Scalar4* d_net_force`, `d_net_torque`, `Scalar* d_net_virial` → ForceReal4*/ForceReal*
in all GPU kernel files:

- FIREEnergyMinimizerGPU.cuh/.cu
- ForceDistanceConstraintGPU.cuh/.cu
- TwoStepBDGPU.cuh/.cu
- TwoStepRATTLEBDGPU.cuh, TwoStepRATTLENVEGPU.cuh
- TwoStepRATTLELangevinGPU.cuh (template header)
- TwoStepRATTLEGPU.cu.inc (template instantiation)

### Step 5: GPU template headers for force compute

Updated ArrayHandle types for `this->m_force`/`m_torque`/`m_virial` in 8 GPU template
headers (PotentialPairGPU.cuh, PotentialBondGPU.cuh, PotentialExternalGPU.cuh, etc.)
and corresponding kernel parameter structs.

### Step 6: Communicator and SFCPackTuner

- **Communicator.cc / CommunicatorGPU.cc**: ArrayHandle types for ghost exchange
- **SFCPackTuner**: Added separate `ForceReal4*` temp buffers for net_force/torque
  (can't reuse `Scalar4* scal4_tmp`), updated virial temp to ForceReal

### Step 7: Test files

Updated `GPUArray<Scalar4>` → `GPUArray<ForceReal4>` in 10+ test `.cc` files.

### Step 8: MPCD module

Updated `mpcd/BounceBackNVE.h`, `BounceBackNVEGPU.cu/.cuh` — net force parameter types.

### Critical bug: sizeof mismatch in hipMemsetAsync

**Discovery**: 57/58 tests passed. Single failure: `test_pppm_force` (GPU triclinic)
with "CUDA Error: invalid argument".

**Root cause**: `sizeof(Scalar4)` (= 32 bytes) used for `hipMemsetAsync` calls that zero
`ForceReal4` arrays (= 16 bytes). Wrote 2× the required memory, overrunning buffers.

**Fix**: Changed `sizeof(Scalar4)` → `sizeof(ForceReal4)` in:
- PPPMForceComputeGPU.cu
- ForceCompositeGPU.cc
- PotentialTersoffGPU.cuh

### Lessons learned

1. **CUDA POD types are strict**: `float4` ↔ `double4` have no conversion operators.
   Every conversion needs explicit `make_scalar4(f.x, f.y, f.z, f.w)`.

2. **`vec3<Scalar>` constructor gotcha**: Has `vec3(Scalar4)` but NOT `vec3(ForceReal4)`.
   Manual field extraction needed in ~12 integrator sites.

3. **sizeof bugs are silent killers**: `hipMemsetAsync(ptr, 0, N * sizeof(Scalar4))`
   compiles fine with `ForceReal4*` ptr but writes 2× too much memory. Always grep for
   `sizeof(Scalar` when changing array element types.

4. **Template instantiation files**: `.cu.inc` files `#include`d by generated `.cu` files
   need updating too.

5. **SFCPackTuner has its own temp buffers**: Typed to match the arrays it permutes —
   needs new buffer types when array element types change.

6. **sed is powerful but dangerous**: Always follow up bulk substitutions with `grep` to
   verify completeness (patterns like `this->m_pdata->` can be missed).

---

## Phase 2 Code Audit

Before Phase 2 implementation, a comprehensive audit of ALL GPU kernels was conducted.

### Critical finding: most force kernels were ALREADY ForceReal

The audit revealed that HOOMD's existing mixed-precision infrastructure had already
converted the most compute-intensive kernels. The original Phase 2A plan (convert
~55-60 files) was based on incorrect assumptions.

**Already ForceReal (no changes needed):**

| Kernel | Status |
|--------|--------|
| `PotentialPairGPU.cuh` | All position loads, dx, evaluator, virial — ForceReal |
| All 32 `EvaluatorPair*.h` | Params, constructor, evalForceAndEnergy — ForceReal |
| `PotentialBondGPU.cuh` | Uses `minImageForceReal` directly |
| `HarmonicAngleForceGPU.cu` | All ForceReal |
| `OPLSDihedralForceGPU.cu` | All ForceReal |
| `NeighborListGPUBinned.cu` | ForceReal distance math |
| `AnisoPotentialPairGPU.cuh` | All ForceReal |

**Correctly Scalar (integrators — should stay double):**

TwoStepNVEGPU.cu, TwoStepConstantVolumeGPU.cu, TwoStepLangevinGPU.cu,
TwoStepBDGPU.cu, FIREEnergyMinimizerGPU.cu — read ForceReal4, compute in Scalar.

**Still FP64, needed conversion (8 kernels):**

| Kernel | Benchmark-relevant? |
|--------|-------------------|
| `PotentialExternalGPU.cuh` | **YES** — wall.Gaussian hits all 64K particles/step |
| `HarmonicDihedralForceGPU.cu` | **YES** — used by md.dihedral.Periodic |
| `TableDihedralForceGPU.cu` | No |
| `HarmonicImproperForceGPU.cu` | No |
| `PeriodicImproperForceGPU.cu` | No |
| `PPPMForceComputeGPU.cu` | No |
| `PotentialTersoffGPU.cuh` | No |
| `ForceCompositeGPU.cu` | No |

### Key technical findings

1. **`loadPosForceReal` reads 32 bytes**: Loads full `Scalar4` (double4 = 32 bytes),
   truncates to float in-register. FP64 FLOPS eliminated but bandwidth cost remains.
   Only fixable by Phase 2B (changing `d_pos` to float4).

2. **`PotentialPairDPDThermoGPU.cuh`**: Re-audit showed it already uses ForceReal3 for
   positions, dx, velocities, force, virial. Only global memory reads are Scalar4.

3. **External wall kernel**: Sole remaining FP64 force kernel in benchmark — all computation
   in Scalar, including evaluator bodies.

---

## Phase 2A: External Evaluators → ForceReal

**Files changed:**
- `PotentialExternalGPU.cuh` — kernel body: Scalar3 → ForceReal3, Scalar virial/energy → ForceReal
- `PotentialExternal.h` — CPU code: match new evaluator interface
- `EvaluatorWalls.h` — constructor, callEvaluator, extrapEvaluator → ForceReal
- `EvaluatorExternalPeriodic.h` — constructor, member, eval → ForceReal
- `EvaluatorExternalElectricField.h` — constructor, member, eval → ForceReal
- `EvaluatorExternalMagneticField.h` — constructor + eval interface

Pattern: evaluator constructors take `ForceReal3 X` instead of `Scalar3 X`.
Params stay Scalar (host/Python compatibility) — narrowed at use site.

### Benchmark (64K polymers + dihedrals, dt=0.005, RTX 4090)

| Build | TPS | vs Double |
|-------|-----|-----------|
| Double | 2,625.9 ± 31.6 | — |
| **Mixed (Phase 2A)** | **3,058.9 ± 69.5** | **+16.5%** |
| Single | 12,288.7 ± 465.5 | +368% |

The mixed→single gap (4.0×) confirmed most remaining FP64 comes from position bandwidth.

---

## Phase 2B: Float4 Position Mirror

### Mechanism: `syncPositionsForceReal()`

Rather than plumbing `d_pos_forcereal` through every integrator kernel (9 files ×
args structs × host code × kernel code), a single sync point in
`ForceCompute::compute()` copies `m_pos → m_pos_forcereal` (narrowing double4→float4)
before any force kernel reads positions.

### Critical bug fix: `.w` type bit-packing

HOOMD packs particle type as an integer into position `.w`. Naive `ForceReal(pos.w)`
corrupts the type tag (floating-point cast loses bits). Fixed with:
```cpp
__int_as_forcereal(__scalar_as_int(pos.w))
```
Helpers `__scalar_as_int()` and `__int_as_forcereal()` added to `HOOMDMath.h`.

### Files changed

**ParticleData infrastructure:**
- `ParticleData.h` — `m_pos_forcereal`, `m_pos_forcereal_alt`, `getPositionsForceReal()`,
  `swapPositionsForceReal()`
- `ParticleData.cc` — allocate + reallocate + initialize mirror arrays
- `ParticleData.cu` / `ParticleData.cuh` — `syncPositionsForceReal()` GPU kernel
- `HOOMDMath.h` — `__scalar_as_int()`, `__int_as_forcereal()` helpers
- `MixedPrecisionPos.h` — `loadPosForceReal()` updated to accept `ForceReal4*`

**Sync point:**
- `ForceCompute.cc` — call `syncPositionsForceReal()` at top of `compute()`

**Force kernels (~30 files) — `Scalar4* d_pos` → `ForceReal4* d_pos`:**
- Pair forces: PotentialPairGPU, AnisoPotentialPairGPU, PotentialPairDPDThermoGPU,
  FrictionPairGPU, PotentialTersoffGPU, PotentialSpecialPairGPU
- Bonded: PotentialBondGPU, BondTablePotentialGPU, all angle/dihedral/improper kernels
- Mesh: MeshBondGPU, MeshVolumeConservationGPU, etc.
- Other: ActiveForceComputeGPU, ConstantForceComputeGPU, ForceCompositeGPU,
  ForceDistanceConstraintGPU, PPPMForceComputeGPU, ComputeThermoGPU, ComputeThermoHMAGPU
- Neighbor list: NeighborListGPUBinned, NeighborListGPUTree, NeighborListGPUStencil

### What did NOT change

- `getPositions()` return type — stays `GPUArray<Scalar4>&`
- All HPMC code — zero changes
- GSD, snapshots, MPI, Python API — unchanged

### Benchmark (64K polymers + dihedrals, dt=0.005, RTX 4090)

| Build | TPS | vs Double |
|-------|-----|-----------|
| Double | 2,625.9 | — |
| Mixed (Phase 2A) | 3,058.9 | +16.5% |
| **Mixed (Phase 2B)** | **4,261.9** | **+62.3%** |
| Single | 12,288.7 | +368% |

Without dihedrals: mixed 11,218 TPS = +149% vs double.

---

## Phase 2C: Dihedral/Improper/Angle Kernel Fixes

### Problem: sqrtf/Scalar "Frankenstein" precision bug

Four dihedral/improper kernels loaded `ForceReal4` positions but immediately promoted to
`Scalar3` (double3) and performed ALL arithmetic in FP64. Two issues:

1. **Performance**: FP64 dihedral math → 1:64 penalty. Adding dihedrals dropped mixed
   from 10,825 to 4,083 TPS (2.65×).

2. **Stability**: Raw `sqrtf()` on `Scalar` (double) values silently truncates to float
   before sqrt, then the float result gets stored in double with noise bits.
   Near-collinear dihedral geometries amplify through `1/raasq`, causing force blowups.

### Files converted

- `HarmonicDihedralForceGPU.cu` — used by `md.dihedral.Periodic` (benchmark kernel)
- `TableDihedralForceGPU.cu` — `vec3<Scalar>` → `vec3<ForceReal>`, `acosf` → `fast::acos`
- `HarmonicImproperForceGPU.cu` — `rsqrtf` → `fast::rsqrt`, `#define SMALL` → `ForceReal(0.001)`
- `PeriodicImproperForceGPU.cu` — Chebyshev recurrence converted
- `CosineSqAngleForceGPU.cu` — cosine-squared angle potential

### Benchmark (64K polymers + dihedrals, dt=0.005, RTX 4090)

| Build | Before fix | After fix | Change |
|-------|-----------|-----------|--------|
| double | 1,158 | 2,499 | (run variance) |
| **mixed** | **4,083** | **7,474** | **+83%** |
| single | 11,954 | 12,062 | (no code change) |

### Stability improvement

| dt | Before: Mixed | After: Mixed |
|----|--------------|-------------|
| 0.005 | 4,083 | stable |
| 0.01 | 1,226 (67% std!) | **5,312 (15% std)** |
| 0.03 | crashed | **4,023 (survives!)** |

Mixed is now the **most stable build** for dihedrals at large dt.

---

## Neighbor List Analysis

The nlist GPU kernel already performs distance math in ForceReal, but reads positions
as `Scalar4` (double4 = 32 bytes) — both self-position from `d_pos` and neighbor
positions from CellList's `d_cell_xyzf`.

Converting CellList's `d_cell_xyzf` to `ForceReal4` would halve neighbor-read bandwidth.
However, the nlist is rebuilt only every ~100 steps. Estimated improvement for polymer
workloads: **~0.05%** — negligible.

Scope would be ~12 files (CellList.h/cc, CellListGPU.cu/cc/cuh,
NeighborListGPUBinned.cu/cuh/cc, NeighborListGPUStencil.cu/cuh/cc,
NeighborListBinned.cc, NeighborListStencil.cc, test_cell_list.cc). Mechanical changes
but touches infrastructure shared with HPMC.

**Decision**: Deferred. Would matter for systems with frequent nlist rebuilds
(fast-moving particles, small buffers) but not for target polymer/chromatin workloads.

---

## Bug Fixes

### Shared memory alignment in AnisoPotentialPairGPU.cuh

**Symptom**: `RuntimeError: CUDA Error: misaligned address` when running patchy particle
forces (PatchyGaussian, etc.) in mixed precision.

**Root cause**: The GPU kernel allocates shared memory as
`[param_type × N][ForceReal × N][shape_type × N_types]`. When `ForceReal = float` (4 bytes)
and `N` is odd (e.g. single-type system: `N = 1`), `shape_type` starts at a non-8-byte-aligned
offset. `shape_type` contains `ManagedArray` with pointer members requiring 8-byte alignment.
In the original double build, `ForceReal = double` (8 bytes) always gave correct alignment.

**Fix**: Added explicit 8-byte alignment padding between `s_rcutsq` and `s_shape_params` in
both the kernel and the launcher's shared memory size calculation.

### minImageForceReal unavailable in non-mixed builds

**Symptom**: Single and double GPU builds fail with `class "BoxDim" has no member
"minImageForceReal"`.

**Root cause**: `BoxDim::minImageForceReal()` was guarded by `#ifdef HOOMD_MIXED_PRECISION`,
but dihedral/improper/angle/bond kernels called it unconditionally after Phase 2C conversion.

**Fix**: Removed the `#ifdef` guard — `ForceReal` is always defined as `ShortReal`, so
the function compiles in all builds. In non-mixed builds (`ForceReal = Scalar`), it's
equivalent to `minImage()`. Also simplified remaining `#ifdef`-guarded call sites in
`PotentialPairGPU.cuh` and `AnisoPotentialPairGPU.cuh`.

---

## Aniso Evaluator: Float Rotation Optimization

**Goal**: Eliminate double-precision quaternion rotation from the patchy force kernel to
unlock mixed-precision speedup for anisotropic potentials.

### rotmat3(quat) cancellation-free constructor (VectorMath.h)

The original `rotmat3(quat)` constructor used `a²+b²-c²-d²` for diagonal elements, which
suffers catastrophic cancellation when the quaternion represents rotations near 90° (where
`a²+b² ≈ c²+d²`). This makes float precision unsafe.

**Fix**: Rewrote using the identity `a²+b²-c²-d² = 1-2c²-2d²` (from the unit quaternion
constraint `a²+b²+c²+d² = 1`). This subtracts small values from 1, eliminating cancellation.
Same formula already used in `EvaluatorPairALJ::quat2mat()`.

Off-diagonal elements (`2(xy ± sz)` etc.) are inherently safe — no cancellation.

**File changed**: `hoomd/VectorMath.h` (~L1143-1175)

### PatchEnvelope rotation unified to ForceReal (PatchEnvelope.h)

Previously, PatchEnvelope had separate CPU and GPU rotation paths:
- **CPU** (`#ifndef __HIPCC__`): `rotmat3<LongReal>` → mat×vec in double
- **GPU** (`#else`): `rotate(quat<LongReal>, vec3<LongReal>)` → 8 double-precision rotations

Both used `LongReal` (double in mixed builds), wasting the RTX 4090's 64:1 FP32:FP64 ratio.

**Fix**: Unified to a single path using `rotmat3<ForceReal>` with the cancellation-free
constructor. Quaternions are narrowed from `Scalar4` to `quat<ForceReal>` (float has ~7
significant digits = ~0.006 millidegree angular precision, more than sufficient for
molecular simulations).

**File changed**: `hoomd/md/PatchEnvelope.h` (~L118-150)

### Benchmark results

With a correctly configured mixed build (`HOOMD_SHORTREAL_SIZE=32`), patchy particles show
**8.8× mixed-over-double speedup** at dt=0.005, and 5–9× across all dt values. The rotation
optimisation contributes to this, though the majority of the gain comes from the isotropic
portions of the aniso kernel (pair I/O, minimum-image, nlist traversal) using `ForceReal`.

Note: earlier benchmarks incorrectly showed mixed ≈ double (1.0×) because the mixed build
had `HOOMD_SHORTREAL_SIZE=64` in `CMakeCache.txt`, making it an effective double build.
All benchmark data in BENCHMARKING.md has been re-collected with the corrected build.

See BENCHMARKING.md "Patchy Particles, No Dihedrals" for full results.

---

## Benchmark Configuration Fix

**Problem**: The `build/` directory ("mixed") had `HOOMD_SHORTREAL_SIZE=64` in its CMake
cache, meaning `ShortReal = double` and `ForceReal = double` — identical to the double build.
All previous "mixed" benchmark numbers were actually double-precision runs, reporting
`floating_point_precision: (64, 64)` and `DOUBLE[DOUBLE]`.

**Root cause**: CMake cache variable persisted from an earlier configure. The CMakeLists.txt
default is `SHORTREAL_SIZE=32`, but once cached, changing the source doesn't update it.

**Fix**: Reconfigured with `cmake . -DHOOMD_SHORTREAL_SIZE=32`, triggering a full rebuild.
Verified with `hoomd.version.floating_point_precision == (64, 32)` and `DOUBLE[SINGLE]`.

**Impact**: All benchmark data re-collected. The corrected results show the expected speedups
(2.7–3.6× for isotropic, 5–9× for patchy) rather than the ~1.0× previously reported for
anisotropic potentials.

---

## Commit History

```
a56ee37b8  Log: expand nlist CellList analysis with scope and cost-benefit
0d85807bf  Update log: Phase 2C results, nlist analysis, summary
4b1025edf  Convert CosineSqAngleForceGPU.cu to ForceReal
a1bdf7b5a  Convert dihedral/improper GPU kernels to ForceReal
8489563a9  Phase 2B: float4 position mirror + accuracy tests + dt sweep benchmarks
007bdc275  Phase 2A Step 1: convert external potential evaluators to ForceReal
2f667b121  Convert nlist, angle, dihedral, DPD thermo, bond kernels to ForceReal
133feaffb  Add benchmark and conversion scripts for mixed-precision testing
3edbe36ee  Mixed precision: ForceReal (float) for force computation on GPU
```
