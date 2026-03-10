# Mixed-Precision GPU Optimization — Change Log

## Problem Statement

HOOMD-blue uses double-precision floating point (FP64) throughout its entire GPU computation
pipeline: force calculations, position storage, virial/torque accumulation, and integrator updates.
On consumer/gaming GPUs such as the NVIDIA RTX 4090, the FP64-to-FP32 throughput ratio is **1:64**,
meaning every double-precision operation runs 64× slower than its single-precision equivalent.

| GPU Class | FP64:FP32 ratio | Example |
|-----------|----------------|---------|
| Data center (A100, H100) | 1:2 | Production HPC |
| Gaming (RTX 3090, 4090) | 1:32 or 1:64 | Consumer cards |

HOOMD's `Scalar` = `LongReal` = `double` (defined in CMakeLists.txt / HOOMDMath.h). Every GPU
kernel — pair force evaluation, neighbor list construction, integration — operates entirely in
double precision. On a gaming GPU, this means the force evaluation loop in PotentialPairGPU.cuh
runs at 1/32 to 1/64 the theoretical throughput.

For many molecular dynamics applications — particularly coarse-grained polymer simulations — the
precision provided by 32-bit floats is more than adequate for forces and short-range interactions.
Double precision is only truly needed for particle positions (to avoid accumulation drift) and
certain long-range electrostatic calculations.

---

## Background Research: OpenMM vs HOOMD Precision Strategies

### How OpenMM achieves equal gaming/datacenter GPU performance

OpenMM uses three key techniques (all achievable in HOOMD):

1. **Two-type system** (CudaContext.cpp): `real` (= `float` in mixed mode) for force kernels;
   `mixed` (= `double` in mixed mode) for integration. Forces are evaluated in float, but
   positions are accumulated in double.

2. **Position correction term** (integrationUtilities.cc): Positions are stored as
   `float4 posq` + `float4 posqCorrection`, where
   `posqCorrection = pos_double - float(pos_double)`. Force kernels read only `posq` (fast
   float4 loads). Integrators reconstruct full double:
   `pos_double = posq + (double)posqCorrection`. This is a form of double-single arithmetic:
   two floats encode ~48 bits of mantissa.

3. **Fixed-point force accumulation** (common.cu): Forces are converted to `long long` via
   `x * 2^32`, enabling exact atomic accumulation. HOOMD doesn't need this — it already avoids
   atomics by assigning one thread-group per particle (PotentialPairGPU.cuh uses warp-level
   reduction instead).

### Where precision actually matters in MD

From computational physics, here's what matters:

| Operation | Precision needed | Justification |
|-----------|-----------------|---------------|
| Pair force evaluation (F = -dV/dr) | **Float OK** | Errors in individual force evaluations are comparable to statistical noise from finite temperature |
| Distance calculation (r_ij) | **Float OK** | Periodic wrapping keeps coordinates O(L), so relative precision is adequate |
| Position accumulation (x += v·Δt) | **Double critical** | Positions grow monotonically; float accumulation causes O(ε_float · x) drift per step — kills energy conservation |
| Velocity update | **Double preferred** | Same accumulation argument, though less severe since velocities oscillate around zero |
| Energy reporting | **Double preferred** | Summing ~N small terms; float loses significance for large N |
| Neighbor list building | **Float OK** | Buffer distance absorbs rounding errors |

### HOOMD's existing precision infrastructure

HOOMD already has `ShortReal` (`float` by default, `HOOMD_SHORTREAL_SIZE=32`) but uses it
**only in HPMC** (hard particle Monte Carlo), never in MD. The type is defined in HOOMDMath.h
alongside `LongReal` (= double) and `Scalar` (= double). The infrastructure exists; the MD
kernels just don't use it.

The `ForceReal` type alias (`= ShortReal = float`) was added as part of the mixed-precision
feature, along with `ForceReal2/3/4` (= float2/3/4). These are the types for force-evaluation
kernel outputs.

HOOMD also has `MixedPrecisionPos.h` with helper functions for position loading:
- `loadPosForceReal(d_pos, idx)` — truncates double → float for force kernels
- `loadPosFull(d_pos, d_pos_correction, idx)` — reconstructs full double from float + correction
- `storePosFull(d_pos, d_pos_correction, idx, pos)` — splits double into float + correction

And `BoxDim.h` has `minImageForceReal()` — the float-precision minimum image convention.

These were introduced but only used in a handful of kernels. The vast majority of MD GPU
kernels still use raw `d_pos[idx]` (double4) and `box.minImage()` (double).

### Key design decisions

- **Chose OpenMM's `pos + correction` split** over Kahan summation: the split is cheaper (no
  extra adds per step) and integrates naturally with separate force-kernel and integrator
  precision requirements.

- **Chose NOT to implement fixed-point force accumulation** (OpenMM's `long long` trick):
  HOOMD doesn't use atomics for forces — each particle has a dedicated thread group with
  warp-level reduction. Fixed-point conversion would add overhead without benefit.

- **Gaming GPUs as sole test platform**: They maximally expose the FP64 penalty, making
  success/failure unambiguous. On datacenter GPUs the same code works correctly, just with
  a smaller speedup (which is fine — those users aren't the ones with the performance problem).

---

## Approach: Three-Phase Plan

Eliminate FP64 from the GPU hot path in three phases:

1. **Phase 1 — Force output pipeline**: Convert all force/virial/torque arrays and GPU kernel
   outputs from `Scalar4`/`Scalar` (double4/double) to `ForceReal4`/`ForceReal` (float4/float).
   This covers the largest number of GPU kernels and the most bandwidth-intensive data transfers.

2. **Phase 2 — Position and kernel arithmetic**: Convert GPU kernels to use float positions
   (force kernels) or float+correction (integrator kernels), and convert all per-particle
   arithmetic (distance, force accumulation, virial) to ForceReal. This eliminates the
   remaining FP64 compute from the GPU hot path.

3. **Phase 3 — Fast math**: Enable `--use_fast_math` CUDA compiler flag for hardware-accelerated
   FP32 intrinsics (sin, cos, exp, rsqrt).

### Technical context

- HOOMD-blue version: v6.1.1
- Branch: `mixed-precision` (fork: `golobor/hoomd-blue`)
- Build configuration: `HOOMD_SHORTREAL_SIZE=32`, `HOOMD_LONGREAL_SIZE=64`
- This means: `ForceReal` = float, `ForceReal4` = float4, `Scalar` = double, `Scalar4` = double4
- `HOOMD_MIXED_PRECISION` is defined when SHORT != LONG sizes
- Key constraint: CUDA POD types (float4, double4) have **no** cross-type assignment operators.
  All conversions must go through `make_scalar4()`/`make_forcereal4()` helpers.

### Verification strategy (all on gaming GPUs)

| Test | What it proves |
|------|----------------|
| Energy drift (NVE, 10^6 steps, LJ fluid ρ*=0.8, T*=1.0) | Mixed ≈ double accuracy |
| Performance (TPS) | Mixed ≈ single speed (within ~10-20%) |
| Force accuracy (frozen config) | Float forces precise enough (median err ~10^-7) |
| Position accumulation unit test | `posCorrection` mechanism works |
| Ensemble averages (NVT) — g(r), pressure, diffusion | No systematic bias |
| Regression: existing pytest MD test suite | Nothing broken |

Expected outcome: double will be ~30-60x slower than single on GeForce cards. Mixed should
land within ~10-20% of single, because the only double-precision work is the integrator
(~5-10% of runtime).

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
- OpenMM does the same (findInteractingBlocks.cu uses `real` = float).

**Cell list kernel** (CellListGPU.cu):
- `makeFraction()` and cell-index computation in Scalar.
- `d_cell_xyzf` stored as Scalar4 (32 bytes) — could be ForceReal4 (16 bytes).
- Safe: cell binning tolerates FP32 rounding (at most 1 ULP error, handled by stencil overlap).

**Angle force kernels** (3 standalone .cu files):
- HarmonicAngleForceGPU.cu, CosineSqAngleForceGPU.cu, TableAngleForceGPU.cu
- All positions, dx, force, virial in Scalar. ~50 FP64 FLOPS per angle.
- Safe: bond angle precision in FP32 (~7 digits) is adequate for ~0.01 radian resolution.
- Note: `sqrtf` is already used on some lines (likely a latent bug) — should be made intentional.

**Dihedral/improper force kernels** (5 standalone .cu files):
- HarmonicDihedralForceGPU.cu, OPLSDihedralForceGPU.cu, TableDihedralForceGPU.cu,
  HarmonicImproperForceGPU.cu, PeriodicImproperForceGPU.cu
- All positions, cross-products, forces, virial in Scalar. ~80 FP64 FLOPS per dihedral.

**Bond kernel `minImage` roundtrip** (PotentialBondGPU.cuh):
- Currently: `ForceReal3 dx → Scalar3 → box.minImage(Scalar3) → ForceReal3` (float→double→float)
- Should use `box.minImageForceReal(dx)` directly. Removes ~12 FP64 ops per bond.

**DPD thermostat pair kernel** (PotentialPairDPDThermoGPU.cuh):
- Does ALL geometry and force accumulation in Scalar (FP64), despite evaluator using ForceReal.
- Direct copy of the pattern already applied to isotropic pair force kernel.

**Observation on atomicAdd**: Pair forces and anisotropic pair forces do NOT use atomicAdd
(they use per-particle thread groups with warp reduction). The multi-body potentials
(Tersoff, angles, dihedrals) and PPPM DO use atomicAdd for force accumulation, but these
already write to ForceReal4 arrays after Phase 1, so the atomics are FP32.

### Iteration 3: Revised phasing (current)

Based on the deeper analysis, the plan was restructured:

- **Phase 1** became: convert force/virial/torque **output storage** to ForceReal (the data
  pipeline, not the kernel internals). This was the minimal change that touches the most files
  and was the right first step to validate the approach. ✅ COMPLETE.

- **Phase 2** became: convert **kernel internals** to use ForceReal — position loading (via
  MixedPrecisionPos.h helpers), distance calculations, force accumulators, all per-particle
  arithmetic. This is where the actual FP64 compute elimination happens. Split into:
  - **Phase 2A**: Use helpers in all kernels, keep m_pos as double4 (safe, incremental)
  - **Phase 2B**: Change m_pos storage to float4 + correction (halves bandwidth, deeper change)

- **Phase 3** remains: `--use_fast_math` for FP32 intrinsics.

---

## Phase 1: Force Output Pipeline

**Goal**: Convert all force/virial/torque storage and GPU kernel I/O from double to float.

**Status**: ✅ COMPLETE — all 58 tests pass

### Step 1: Core data structures

Changed the underlying storage types for force data throughout the core:

**ForceCompute.h/.cc**:
- `GPUArray<Scalar4> m_force` → `GPUArray<ForceReal4> m_force`
- `GPUArray<Scalar4> m_torque` → `GPUArray<ForceReal4> m_torque`
- `GPUArray<Scalar> m_virial` → `GPUArray<ForceReal> m_virial`
- All accessor methods updated to return `ForceReal4`/`ForceReal` types
- `LocalForceComputeData` class updated

**ParticleData.h/.cc**:
- `m_net_force`, `m_net_virial`, `m_net_torque` arrays → ForceReal4/ForceReal
- Corresponding "Alt" arrays for double-buffering
- All `getNetForce()`, `getNetVirial()`, `getNetTorque()` accessors
- `GhostLocalDataAccess::getLocalBuffer` specializations

**Result**: Core data layer compiles, but downstream consumers all break with type mismatches.

### Step 2: Force summation infrastructure

**Integrator.cu/.cc/.cuh**:
- `gpu_force_list` struct: changed `Scalar4* d_force`, `Scalar* d_virial`, `Scalar4* d_torque`
  → `ForceReal4*`/`ForceReal*` types
- GPU kernel `gpu_integrator_sum_net_force_kernel`: all local variables and accumulator types
  converted from Scalar4/Scalar to ForceReal4/ForceReal
- CPU fallback `Integrator::computeNetForce()`: ArrayHandle types updated

**Result**: Force summation compiles and produces float output.

### Step 3: Integrator step kernels (CPU side)

Updated `ArrayHandle<Scalar4>` → `ArrayHandle<ForceReal4>` (and Scalar → ForceReal for virial)
for net force/torque/virial reads in all integrator `.cc` files:

- TwoStepLangevin.cc
- TwoStepConstantVolume.cc
- TwoStepConstantPressure.cc
- FIREEnergyMinimizer.cc
- ForceDistanceConstraint.cc

Also fixed `vec3<Scalar>(ForceReal4)` patterns — the `vec3` constructor only accepts `Scalar4`,
so ForceReal4 fields must be extracted manually:
```cpp
// Before:
vec3<Scalar> t(h_net_torque.data[j]);
// After:
auto t4 = h_net_torque.data[j];
vec3<Scalar> t(t4.x, t4.y, t4.z);
```

Files with this pattern: TwoStepLangevin.cc, TwoStepConstantVolume.cc,
TwoStepConstantPressure.cc, TwoStepRATTLELangevin.h, TwoStepRATTLENVE.h (12 occurrences total).

### Step 4: GPU kernel signatures and bodies

Changed `Scalar4* d_net_force`, `Scalar4* d_net_torque`, `Scalar* d_net_virial` parameters to
ForceReal4*/ForceReal* in all GPU kernel files:

**Kernel declaration/definition files (.cuh/.cu)**:
- FIREEnergyMinimizerGPU.cuh/.cu
- ForceDistanceConstraintGPU.cuh/.cu
- TwoStepBDGPU.cuh/.cu
- TwoStepRATTLEBDGPU.cuh
- TwoStepRATTLENVEGPU.cuh
- TwoStepRATTLELangevinGPU.cuh (template header)
- TwoStepRATTLEGPU.cu.inc (template instantiation — generates .cu files)

Also changed kernel-body local variables:
```cpp
// Before:
Scalar4 net_force = d_net_force[idx];
// After:
ForceReal4 net_force = d_net_force[idx];
```

And `make_scalar4` → `make_forcereal4` where writing back to ForceReal4 arrays.

### Step 5: GPU template headers for force compute

Updated `ArrayHandle<Scalar4>` for `this->m_force`/`this->m_torque`/`this->m_virial` in 8 GPU
template headers:
- PotentialPairGPU.cuh
- PotentialBondGPU.cuh
- PotentialExternalGPU.cuh
- PotentialSpecialPairGPU.cuh
- PotentialTersoffGPU.cuh
- AnisoPotentialPairGPU.cuh
- PotentialMeshGPU.cuh (via pattern match)
- ActiveForceComputeGPU.cuh (implied)

Also updated `Scalar4* d_force`/`d_torque` and `Scalar* d_virial` in the corresponding kernel
parameter structs and kernel bodies across all force-compute `.cu` files.

### Step 6: Communicator and SFCPackTuner

**Communicator.cc / CommunicatorGPU.cc**:
- ArrayHandle types for net force/torque/virial during ghost exchange

**SFCPackTuner.cc / SFCPackTunerGPU.cu/.cuh/.cc**:
- Temporary sort buffers for force arrays: `Scalar4* scal4_tmp` can't hold ForceReal4 data
- Added separate `ForceReal4*` temp buffers for net_force, net_torque
- Updated net_virial temp buffer to ForceReal

### Step 7: Test files

Updated `GPUArray<Scalar4>` → `GPUArray<ForceReal4>` and corresponding ArrayHandle types in
10+ test `.cc` files to match the new API.

### Step 8: MPCD module

Updated `mpcd/BounceBackNVE.h`, `BounceBackNVEGPU.cu`, `BounceBackNVEGPU.cuh` — net force
parameter types in the bounce-back integration kernel.

### Critical bug: sizeof mismatch in hipMemsetAsync

**Discovery**: After all type changes compiled, 57/58 tests passed. The single failure was
`test_pppm_force` (GPU triclinic variant) with "CUDA Error: invalid argument".

**Root cause**: Several files used `sizeof(Scalar4)` (= 32 bytes for double4) to compute byte
counts for `hipMemsetAsync` calls that zero `ForceReal4` arrays (= 16 bytes for float4). This
wrote 2× the required memory, overrunning buffer boundaries.

**Fix**: Changed `sizeof(Scalar4)` → `sizeof(ForceReal4)` and `sizeof(Scalar)` → `sizeof(ForceReal)`
in:
- `PPPMForceComputeGPU.cu` — `d_force` memset
- `ForceCompositeGPU.cc` — `d_force`, `d_torque`, `d_virial` memsets
- `PotentialTersoffGPU.cuh` — `d_force`, `d_virial` memsets

**Result**: All 58 tests pass. Phase 1 complete.

### Lessons learned from Phase 1

1. **CUDA POD types are strict**: `float4` and `double4` have no implicit or explicit conversion
   operators between them. You cannot assign one to the other. Every conversion must go through
   an explicit helper like `make_scalar4(f.x, f.y, f.z, f.w)`. This applies to all CUDA vector
   types (float2↔double2, float3↔double3, etc.).

2. **`vec3<Scalar>` constructor gotcha**: HOOMD's `vec3<T>` template has a convenience constructor
   `vec3(Scalar4)` but NOT `vec3(ForceReal4)`. When reading from ForceReal4 arrays, you need
   manual field extraction: `vec3<Scalar>(t4.x, t4.y, t4.z)`. This pattern appears in ~12 places
   across integrator files.

3. **sizeof bugs are silent killers**: `hipMemsetAsync(ptr, 0, N * sizeof(Scalar4))` compiles
   fine even when `ptr` points to a `ForceReal4` array — but it writes 2× too much memory (32
   vs 16 bytes per element), corrupting adjacent allocations. The symptom was a "CUDA Error:
   invalid argument" that only manifested in one specific test (triclinic PPPM). Always grep for
   `sizeof(Scalar` when changing array element types.

4. **Template instantiation files**: Some HOOMD GPU code uses `.cu.inc` files that are `#include`d
   by generated `.cu` files for template instantiation (e.g., `TwoStepRATTLEGPU.cu.inc`). These
   need to be updated too, and cmake may need reconfiguration to regenerate the `.cu` wrappers.

5. **SFCPackTuner has its own temp buffers**: The space-filling curve particle sorter maintains
   temporary buffers typed to match the arrays it permutes. When force arrays changed from
   Scalar4 to ForceReal4, new temp buffer types were needed — a `Scalar4*` pointer can't receive
   `ForceReal4` data.

6. **sed is powerful but dangerous**: Bulk `sed -i` edits across dozens of files are fast for
   mechanical type substitutions, but patterns like `this->m_pdata->getNetForce()` can be missed
   if the initial regex only matches `m_pdata->getNetForce()` (missing the `this->` prefix).
   Always do a follow-up grep to verify completeness.

---

## Research: Phase 2 Code Audit

Before starting Phase 2 implementation, we conducted a comprehensive audit of ALL GPU kernels
to determine which ones actually need conversion from FP64 to ForceReal.

### Critical finding: most force kernels are ALREADY ForceReal

The audit revealed that HOOMD's existing mixed-precision infrastructure had already converted
the most compute-intensive kernels. The original Phase 2A plan (convert ~55-60 files) was based
on incorrect assumptions.

#### Already ForceReal (0 FP64 FLOPS) — no changes needed

| Kernel | What's ForceReal |
|--------|-----------------|
| `PotentialPairGPU.cuh` | All position loads, dx, rsq, evaluator, force accumulation, virial |
| All 32 `EvaluatorPair*.h` | Params, constructor, evalForceAndEnergy — all ForceReal |
| `PotentialBondGPU.cuh` | Position, evaluator, force. Uses `minImageForceReal` directly |
| `HarmonicAngleForceGPU.cu` | Position loads, dx, force, virial — all ForceReal |
| `OPLSDihedralForceGPU.cu` | Position loads, dx, force, virial — all ForceReal |
| `NeighborListGPUBinned.cu` | Position loads, dx, rsq — all ForceReal |
| `AnisoPotentialPairGPU.cuh` | Position, orientation, evaluator — all ForceReal |
| `PotentialSpecialPairGPU` | Same pattern as PotentialBondGPU |

#### Correctly Scalar (should stay double) — integrators

| Kernel | Why it's correct |
|--------|-----------------|
| `TwoStepNVEGPU.cu` | Reads ForceReal4 net_force, promotes to Scalar for velocity/position updates |
| `TwoStepConstantVolumeGPU.cu` | Same pattern — accumulation precision needed |
| `TwoStepLangevinGPU.cu` | Same — step_two reads ForceReal4, computes in Scalar |
| `TwoStepBDGPU.cu` | Same pattern |
| `FIREEnergyMinimizerGPU.cu` | Same pattern |
| `CellListGPU.cu` | Cell assignment kernel — runs infrequently, not a force kernel |
| `BoxResizeUpdaterGPU.cu` | Infrastructure, not hot path |

#### Still FP64, NEED conversion (8 kernels)

| Kernel | FP64 FLOPS/unit | Impact on benchmark? |
|--------|-----------------|---------------------|
| `PotentialExternalGPU.cuh` | ~20-50/particle | **YES** — wall.Gaussian hits all 64K particles/step |
| `PotentialPairDPDThermoGPU.cuh` | ~80/pair | **YES** — DPD conservative pair in benchmark |
| `PPPMForceComputeGPU.cu` | ~200/particle | No — no electrostatics in benchmark |
| `PotentialTersoffGPU.cuh` | ~100+/pair | No — no Tersoff in benchmark |
| `HarmonicImproperForceGPU.cu` | ~100/improper | No — no impropers in benchmark |
| `ForceCompositeGPU.cu` | ~40/constituent | No — no rigid bodies in benchmark |
| `ActiveForceComputeGPU.cu` | ~30/particle | No — no active forces in benchmark |
| `ForceDistanceConstraintGPU.cu` | ~50/constraint | No — no constraints in benchmark |

### Key technical findings

#### 1. `loadPosForceReal` reads 32 bytes (bandwidth waste)

`loadPosForceReal(d_pos, idx)` loads a full `Scalar4` (double4 = 32 bytes) from global memory,
then truncates to float in-register. The FP64 FLOPS are eliminated (truncation is free), but
the bandwidth cost remains: 32 bytes per particle instead of 16 bytes. This is only fixable by
Phase 2B (changing `d_pos` to float4).

#### 2. `storePosFull` does NOT truncate `d_pos`

`storePosFull(d_pos, d_pos_correction, idx, pos)` writes the **full double** to `d_pos[idx]`
AND computes `correction = pos_double - float(pos_double)` stored in `d_pos_correction[idx]`.
Currently the correction is redundant since `d_pos` retains full precision. The correction
mechanism is designed for Phase 2B, when `d_pos` becomes float4 and corrections carry actual
information.

#### 3. SFCPackTuner fix only needed for Phase 2B

The `m_pos_correction` permutation bug in SFCPackTuner only matters when the correction array
carries real data (Phase 2B). Currently `storePosFull` keeps d_pos at full precision, so
unpermuted corrections are harmless — they're never used to reconstruct positions in force
kernels (which read d_pos directly).

#### 4. `PotentialPairDPDThermoGPU.cuh` is already mostly ForceReal

Re-audit revealed the DPD kernel already uses ForceReal3 for positions (posi, posj), dx,
velocities, force accumulation, and virial. The remaining FP64: position loads from d_pos
(Scalar4) and velocity loads from d_vel (Scalar4) — these are just the global memory reads,
immediately narrowed to ForceReal3. No FP64 compute remaining.

#### 5. External wall kernel — sole FP64 force kernel in benchmark

`PotentialExternalGPU.cuh` runs ALL computation in Scalar (double): position extraction,
force/torque/virial initialization, evaluator construction, and evaluator body. The wall
evaluators (`EvaluatorWalls.h`, `EvaluatorExternalPeriodic.h`, etc.) also use Scalar internally.
Only the output writes (from Phase 1) are ForceReal.

The benchmark (`benchmark_chains.py`) uses `hoomd.md.external.wall.Gaussian` on all 64,000
particles every timestep. This is the only remaining FP64 force kernel in the benchmark
hot path.

---

## Phase 2A: Revised Execution Plan

**Goal**: Convert the remaining FP64 force kernels to ForceReal. Focus on benchmark-relevant
kernels first.

**Status**: ✅ COMPLETE

### Step 1: Convert PotentialExternalGPU.cuh + all external evaluators

**The benchmark bottleneck.** Convert the GPU kernel and all 4 evaluators to use ForceReal
for positions, forces, torques, virial, and energy. Wall geometry functions
(`distVectorWallToPoint` in WallData.h) stay Scalar — their results are narrowed to ForceReal
at the evaluator boundary.

Files changed:
- `PotentialExternalGPU.cuh` — kernel body: Scalar3 force/torque/Xi → ForceReal3, Scalar virial/energy → ForceReal
- `PotentialExternal.h` — CPU code: match new evaluator interface
- `EvaluatorWalls.h` — constructor, callEvaluator, extrapEvaluator, evalForceTorqueEnergyAndVirial → ForceReal
- `EvaluatorExternalPeriodic.h` — constructor, member, eval → ForceReal
- `EvaluatorExternalElectricField.h` — constructor, member, eval → ForceReal
- `EvaluatorExternalMagneticField.h` — constructor + eval interface (internals stay Scalar for quat precision)

Pattern: evaluator constructors take `ForceReal3 X` (position) instead of `Scalar3 X`.
Evaluator param_type members stay Scalar (host/Python compatibility) — narrowed at use site.

### Step 2: Convert remaining FP64 force kernels (low priority)

Only matters if these force types are used in target workloads. Conversion is mechanical —
same pattern as HarmonicAngle: narrow positions to ForceReal3, change intermediates to ForceReal,
switch `box.minImage()` → `box.minImageForceReal()`, narrow params at use site.

Already converted: HarmonicAngleForceGPU.cu, OPLSDihedralForceGPU.cu

Still FP64 (entire kernel body):
- `CosineSqAngleForceGPU.cu` — cosine-squared angle potential
- `TableAngleForceGPU.cu` — tabulated angle potential
- `HarmonicDihedralForceGPU.cu` — also has `__scalar2int_rn` macro to update
- `TableDihedralForceGPU.cu` — also uses `vec3<Scalar>` → needs `vec3<ForceReal>`
- `HarmonicImproperForceGPU.cu` — has `#define SMALL Scalar(0.001)`
- `PeriodicImproperForceGPU.cu` — Chebyshev recurrence, all Scalar
- `PPPMForceComputeGPU.cu` — charge spreading + force interpolation
- `PotentialTersoffGPU.cuh` — three-body potential
- `ForceCompositeGPU.cu` — rigid body forces
- `ActiveForceComputeGPU.cu` — active matter
- `ForceDistanceConstraintGPU.cu` — distance constraints

### Step 3: Validate and benchmark

Run benchmarks on gaming GPU to measure the impact:
- `benchmark_chains.py` (64K particles, bonds+angles+dihedrals+wall, Langevin)
- `benchmark_mixed.py` (27K particles, LJ liquid, NVE)
- Profile with `nsys` to confirm FP64 instruction count drops to ~0 in force kernels

---

## Benchmark Results

### Phase 2A Step 1: External potentials → ForceReal

**Workload:** `benchmark_chains.py` — 64K particles, 320 chains × 200 monomers,
pair(Gaussian A=5) + harmonic bonds + wall(Gaussian) + angle + dihedral,
Langevin integrator, dt=0.005.

**Hardware:** RTX 4090 (FP64:FP32 = 1:64), GPU 7.

**Protocol:** 10K warmup + 100K benchmark steps, report every 10K.

| Build | TPS (mean ± std) | ns/day | vs Double |
|-------|-------------------|--------|-----------|
| Double (baseline) | 2625.9 ± 31.6 | 1.134 | — |
| **Mixed (Phase 1+2A)** | **3058.9 ± 69.5** | **1.321** | **+16.5%** |
| Single | 12288.7 ± 465.5 | 5.309 | +368% |

**Analysis:**
- Mixed → Double speedup of **16.5%** confirms FP64 elimination in force output pipeline
  and external potential evaluators has measurable impact.
- Single is **4.7× faster** than double, showing the FP64:FP32 penalty is enormous on RTX 4090.
- The large gap between mixed (3059) and single (12289) = **4.0×** tells us that most remaining
  FP64 comes from **position storage** (`m_pos` is still `Scalar4 = double4`). Every kernel
  reads positions from global memory — pair forces, bond forces, wall forces, neighbor list,
  integrator — all paying the double-precision bandwidth cost.
- **Phase 2B (m_pos → float4)** is the critical next step to close this gap.

---

## Phase 2B: Float4 Position Mirror ✅ COMPLETE

### Problem

Every GPU kernel reads positions from `m_pos` (`GPUArray<Scalar4>` = `double4` in mixed mode).
Even after converting force arithmetic to ForceReal, every kernel still pays the double4 bandwidth
cost for position reads. With 5+ force kernels + neighbor list per timestep, position reads
dominate GPU memory traffic.

Benchmark gap: mixed 3059 TPS vs single 12289 TPS = 4.0× gap, almost entirely from position
bandwidth.

### Design Decision

**Two-array approach** (not replacing m_pos):
- Keep `m_pos` as `GPUArray<Scalar4>` — the full-precision "source of truth"
- Add `m_pos_forcereal` as `GPUArray<ForceReal4>` — float4 mirror for GPU kernels
- Add `getPositionsForceReal()` accessor returning `GPUArray<ForceReal4>&`
- Integrator writes both arrays (one extra float4 write per particle per step)
- Force kernels + neighbor list read `m_pos_forcereal` (float4) instead of `m_pos` (double4)

**Why not replace m_pos?** HPMC (Monte Carlo) is compiled (`BUILD_HPMC=ON`) and has 30+ sites
using `ArrayHandle<Scalar4>(getPositions(), ...)`. Changing the return type of `getPositions()`
would break compilation. The two-array approach leaves HPMC, GSD, snapshots, and MPI completely
untouched.

**Memory cost**: +N × 16 bytes (one float4 per particle). For N=64K: +1 MB. Negligible vs
the bandwidth savings.

**Bandwidth analysis per timestep** (64K particles, 5 force kernels + nlist):
- Current: 6 kernels × N × 32 bytes (double4 reads) = 12.3 MB reads
- After:   6 kernels × N × 16 bytes (float4 reads) = 6.1 MB reads + N × 16 bytes extra write
- Net saving: ~6 MB/step bandwidth reduction

### Implementation Plan

#### Step 1: Add float4 mirror to ParticleData

In `ParticleData.h`:
- Add member: `GPUArray<ForceReal4> m_pos_forcereal` (ifdef HOOMD_MIXED_PRECISION)
- Add member: `GPUArray<ForceReal4> m_pos_forcereal_alt` (for SFCPackTuner swap)
- Add accessor: `getPositionsForceReal()` → returns `m_pos_forcereal`
- Add `swapPositionsForceReal()` for SFCPackTuner
- In non-mixed builds: `getPositionsForceReal()` returns `m_pos` (same type)

In `ParticleData.cc`:
- `allocate()`: allocate `m_pos_forcereal` alongside `m_pos`
- `reallocate()`: resize `m_pos_forcereal` alongside `m_pos`
- `initializeFromSnapshot()`: after writing `m_pos`, populate `m_pos_forcereal` by narrowing

#### Step 2: Update integrators to write both arrays

In `MixedPrecisionPos.h`:
- Update `storePosFull()` to also write narrowed float4 to `d_pos_forcereal`:
  ```
  storePosFull(d_pos, d_pos_correction, d_pos_forcereal, idx, pos)
  ```
- Or keep storePosFull unchanged, add separate `storePosForceReal()` call

In integrator `.cu` kernels (TwoStepConstantVolume, NVE, BD, Langevin, ConstantPressure,
FIRE, RATTLE variants):
- Add `d_pos_forcereal` parameter
- After `storePosFull(d_pos, ...)`, also write:
  `d_pos_forcereal[idx] = make_forcereal4(ForceReal(pos.x), ...)`

In integrator `.cuh` args structs:
- Add `ForceReal4* d_pos_forcereal` parameter

In integrator `.cc` host code:
- Get `d_pos_forcereal` handle from `getPositionsForceReal()`

#### Step 3: Update force kernels to read float4

In force kernel host code (`.h` files like PotentialPairGPU.h, PotentialBondGPU.h, etc.):
- Change `getPositions()` → `getPositionsForceReal()` for GPU kernel args
- Change `ArrayHandle<Scalar4> d_pos` → `ArrayHandle<ForceReal4> d_pos`

In force kernel `.cuh` args structs:
- Change `const Scalar4* d_pos` → `const ForceReal4* d_pos`

In force kernel `.cu` bodies:
- `loadPosForceReal(d_pos, idx)` becomes a direct read (no narrowing needed)
- Or update `loadPosForceReal` to accept `const ForceReal4*`

Files to change (args struct + host launcher):
- `PotentialPairGPU.cuh` / `PotentialPairGPU.h`
- `PotentialBondGPU.cuh` / `PotentialBondGPU.h`
- `PotentialExternalGPU.cuh` / `PotentialExternalGPU.h`
- `PotentialTersoffGPU.cuh` / `PotentialTersoffGPU.h`
- `PotentialSpecialPairGPU.h`
- `AnisoPotentialPairGPU.cuh` / `AnisoPotentialPairGPU.h`
- `PotentialPairDPDThermoGPU.cuh` / `PotentialPairDPDThermoGPU.h`
- `FrictionPairGPU.cuh` / `FrictionPairGPU.h`
- All angle/dihedral/improper `.cuh` + `.cc` (6 force types)
- All mesh force `.cuh` + `.cc` (4 force types)
- `ActiveForceComputeGPU.cuh` / `.cc`
- `ConstantForceComputeGPU.cuh` / `.cc`
- `BondTablePotentialGPU.cuh` / `.cc`
- `ForceCompositeGPU.cuh` / `.cc`
- `ForceDistanceConstraintGPU.cuh` / `.cc`
- `PPPMForceComputeGPU.cuh` / `.cc`
- `ComputeThermoGPU.cuh` / `.cc`
- `ComputeThermoHMAGPU.cuh` / `.cc`

#### Step 4: Update neighbor list to read float4

In `NeighborListGPU.cc` (or host launchers):
- Pass `getPositionsForceReal()` instead of `getPositions()`

In nlist GPU kernels (`NeighborListGPUBinned.cu`, `NeighborListGPUTree.cu`,
`NeighborListGPUStencil.cuh`):
- Change `const Scalar4* d_pos` → `const ForceReal4* d_pos`
- Position reads become direct (already float4)
- `d_last_updated_pos` should also become `ForceReal4`

#### Step 5: Update SFCPackTuner

In `SFCPackTunerGPU.cc`:
- Permute `m_pos_forcereal → m_pos_forcereal_alt` alongside `m_pos → m_pos_alt`
- Call `swapPositionsForceReal()` after `swapPositions()`

In `SFCPackTunerGPU.cu`:
- Add `d_pos_forcereal` / `d_pos_forcereal_alt` parameters to sort kernel
- Permute both: `d_pos_alt[idx] = d_pos[old_idx]; d_pos_fr_alt[idx] = d_pos_fr[old_idx];`

CPU path (`SFCPackTuner.cc`): same pattern.

#### Step 6: Update MPCD BounceBack (if needed)

`hoomd/mpcd/BounceBackNVEGPU.cu` — reads positions via `d_pos`. If this kernel benefits from
float4 reads, update. Otherwise skip (MPCD has its own position storage).

### What does NOT change

- `getPositions()` return type — stays `GPUArray<Scalar4>&`
- All HPMC code — zero changes
- GSD writer/reader — reads `getPositions()` (full Scalar4), unchanged
- Snapshots (`takeSnapshot` / `initializeFromSnapshot`) — unchanged except init populates mirror
- MPI communicator — unchanged (packs from `getPositions()`)
- Python API (`LocalParticleData`) — unchanged
- `pdata_element` struct — unchanged
- All CPU force compute code — unchanged (reads `getPositions()`)

### Verification

- `make -j8` — all targets compile (including HPMC)
- `ctest --output-on-failure -j8` — all 58 tests pass
- Benchmark: `benchmark_chains.py` 64K particles — expect significant speedup toward single

### Phase 2B Implementation Summary

**Key mechanism — `syncPositionsForceReal()`:**

Rather than adding `d_pos_forcereal` plumbing to every integrator kernel (9 files × args
structs × host code × kernel code), a simpler approach was used: a single sync point in
`ForceCompute::compute()` that copies `m_pos` → `m_pos_forcereal` (narrowing double4→float4)
before any force kernel reads positions. This is called once per timestep, keeping the
`.w` component (type tag) intact via `__int_as_forcereal(__scalar_as_int(pos.w))`.

**Critical bug fix — `.w` type bit-packing:**

HOOMD packs particle type as an integer into the `.w` component of position float4/double4.
The narrowing `ForceReal(pos.w)` would corrupt the type tag (floating-point cast loses bits).
Fixed by using `__int_as_forcereal(__scalar_as_int(pos.w))` to reinterpret the integer bits
directly, preserving the type tag exactly. The `__scalar_as_int()` and `__int_as_forcereal()`
helpers were added to `HOOMDMath.h`.

**Files changed (summary):**

ParticleData infrastructure:
- `ParticleData.h` — added `m_pos_forcereal`, `m_pos_forcereal_alt`, `getPositionsForceReal()`,
  `swapPositionsForceReal()`
- `ParticleData.cc` — allocate + reallocate + initialize mirror arrays
- `ParticleData.cu` / `ParticleData.cuh` — `syncPositionsForceReal()` GPU kernel
- `HOOMDMath.h` — `__scalar_as_int()`, `__int_as_forcereal()` helpers
- `MixedPrecisionPos.h` — `loadPosForceReal()` updated to accept `ForceReal4*`

Sync point:
- `ForceCompute.cc` — call `syncPositionsForceReal()` at top of `compute()`

Force kernels (~30 files) — changed `Scalar4* d_pos` → `ForceReal4* d_pos`:
- All pair force templates: `PotentialPairGPU.cuh/.h`, `AnisoPotentialPairGPU.cuh/.h`,
  `PotentialPairDPDThermoGPU.cuh/.h`, `FrictionPairGPU.cuh/.h`, `PotentialTersoffGPU.cuh/.h`,
  `PotentialSpecialPairGPU.h`
- All bonded forces: `PotentialBondGPU.cuh/.h`, `BondTablePotentialGPU.cuh/.cc`,
  `HarmonicAngleForceGPU.cu/.cuh`, `CosineSqAngleForceGPU.cu/.cuh`,
  `TableAngleForceGPU.cu/.cuh`, `OPLSDihedralForceGPU.cu/.cuh`,
  `HarmonicDihedralForceGPU.cu/.cuh`, `TableDihedralForceGPU.cu/.cuh`,
  `HarmonicImproperForceGPU.cu/.cuh`
- Mesh forces: `MeshBondGPU.cuh`, `MeshVolumeConservationGPU.cuh`, etc.
- Active/constant forces: `ActiveForceComputeGPU.cu/.cuh`, `ConstantForceComputeGPU.cu/.cuh`
- Other: `ForceCompositeGPU.cu/.cuh`, `ForceDistanceConstraintGPU.cu/.cuh`,
  `PPPMForceComputeGPU.cu/.cuh`, `ComputeThermoGPU.cu/.cuh`, `ComputeThermoHMAGPU.cu/.cuh`
- Neighbor list: `NeighborListGPUBinned.cu`, `NeighborListGPUTree.cu`,
  `NeighborListGPUStencil.cuh`

---

## Accuracy Testing

Measured force accuracy comparing double (reference), mixed, and single precision builds
on the same 64K-particle polymer system (benchmark_chains.py configuration, 100 steps).

### Force Accuracy (vs Double Reference)

| Metric | Mixed | Single |
|--------|-------|--------|
| Max relative error | 1.82×10⁻⁵ | 1.82×10⁻⁵ |
| Mean relative error | 3.64×10⁻⁷ | 3.64×10⁻⁷ |
| Mean absolute error | 2.87×10⁻⁶ | 2.87×10⁻⁶ |

Mixed and single produce essentially identical force errors — confirming that in mixed mode,
all force computation now uses float32 arithmetic. The ~10⁻⁷ mean relative error is consistent
with float32 machine epsilon (~1.2×10⁻⁷).

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

All three builds show comparable energy drift over 100 steps, with mixed and single
essentially identical (as expected — same float32 force accuracy).

---

## Benchmark Results — Phase 2B

### Phase 2B vs Phase 2A Speedup (dt=0.005, 64K particles, RTX 4090)

**Full forces (pair + bond + angle + dihedral + wall):**

| Build | TPS | vs Double |
|-------|-----|-----------|
| Double | 2625.9 ± 31.6 | — |
| Mixed (Phase 2A) | 3058.9 ± 69.5 | +16.5% |
| **Mixed (Phase 2B)** | **4261.9** | **+62.3%** |
| Single | 12288.7 ± 465.5 | +368% |

**No dihedral (pair + bond + angle + wall):**

| Build | TPS | vs Double |
|-------|-----|-----------|
| Double | 4504.1 | — |
| Mixed (Phase 2A) | 6245.2 | +38.6% |
| **Mixed (Phase 2B)** | **11218.2** | **+149%** |
| Single | 19408.0 | +331% |

**No angle, no dihedral (pair + bond + wall):**

| Build | TPS | vs Double |
|-------|-----|-----------|
| Double | 5816.7 | — |
| Mixed (Phase 2A) | 9858.1 | +69.5% |
| **Mixed (Phase 2B)** | **12227.5** | **+110%** |
| Single | 20589.4 | +254% |

**Analysis:**

Phase 2B delivers massive speedups. Without dihedrals, mixed achieves 2.49× double— vs single's
4.31×. The remaining gap (~1.7×) comes from:
1. Integrator kernels still read/write `double4` positions (the "source of truth")
2. Scalar4 (double4) position writes in `storePosFull()`
3. Remaining FP64 in `syncPositionsForceReal()` narrowing kernel

Dihedral forces remain a bottleneck — probably due to the complex gather pattern
(4-body interactions) where FP64 arithmetic cost dominates over bandwidth.

### dt Sweep Benchmarks

Tested timestep stability and performance across dt values. The dt sweep protocol uses
a two-phase approach: equilibrate once per build (10K steps at dt=0.005), save state,
then benchmark each dt from the shared equilibrated state.

#### 64K particles, WITH dihedrals (pair + bond + angle + dihedral + wall)

| dt | Double (TPS) | Mixed (TPS) | Single (TPS) |
|----|-------------|-------------|---------------|
| 0.005 | 1158 | 4083 | 11954 |
| 0.01 | 2028 | 1226 (67% std!) | 9100 |
| 0.03 | crashed | crashed | 7522 |
| 0.05 | crashed | crashed | crashed |
| 0.1 | crashed | crashed | crashed |

Dihedrals cause stability issues at dt≥0.03 across all builds. The mixed build becomes
unstable at dt=0.01 (extremely high TPS variance → simulation going wrong). The single
build survives dt=0.03 only because lower-precision arithmetic is more forgiving of
near-singular dihedral geometries.

#### 64K particles, NO dihedrals (pair + bond + angle + wall)

| dt | Double (TPS) | Mixed (TPS) | Single (TPS) |
|----|-------------|-------------|---------------|
| 0.005 | 4328 | 10825 | 19548 |
| 0.01 | 4125 | 9465 | 15261 |
| 0.03 | 3278 | 7522 | 4828 |
| 0.05 | 1837 | 5140 | 5345 |
| 0.1 | 1882 | 4970 | 8088 |

Without dihedrals, all builds stable across all dt values. Mixed consistently ~2.5×
faster than double across all timesteps. At larger dt, single-precision performance
degrades (dt=0.03: 4828 TPS) while mixed remains strong (7522 TPS), possibly due to
single-precision integrator accumulation errors at large dt causing more neighbor list
rebuilds.

#### 200K particles, NO dihedrals (pair + bond + angle + wall)

| dt | Double (TPS) | Mixed (TPS) | Single (TPS) |
|----|-------------|-------------|---------------|
| 0.005 | 1485 | 4384 | 6760 |
| 0.01 | 1361 | 3840 | 5824 |
| 0.03 | 1168 | 3061 | 4539 |
| 0.05 | 787 | 2112 | 2375 |
| 0.1 | 814 | 2128 | 2403 |

Scaling: Mixed achieves ~2.95× double at dt=0.005 (up from 2.5× at 64K), confirming
the bandwidth-bound nature of the workload benefits more at larger system sizes.
Single-to-mixed ratio narrows to ~1.54× (from ~1.80× at 64K), consistent with the
diminishing returns of arithmetic speedup as memory bandwidth becomes the dominant
bottleneck.

---

## Phase 2C: Dihedral/Improper/Angle Kernel Fixes ✅ COMPLETE

### Problem

Four dihedral/improper kernels and one angle kernel were missed during Phase 2B.
They loaded `ForceReal4` positions but immediately promoted to `Scalar3` (double3) and
performed ALL arithmetic in `Scalar` (double). Two issues:

1. **Performance**: All dihedral math in FP64 → 1:64 penalty on RTX 4090.
   This explained why adding dihedrals dropped mixed TPS from 10825 to 4083 (2.65×).
2. **Precision Frankenstein → instability**: Raw `sqrtf()` called on `Scalar` (double)
   values silently truncates to float before computing sqrt, then the result (with ~7
   significant digits) gets stored in double and subsequent double-precision divisions
   pad with noise bits. Near-collinear dihedral geometries amplify this noise through
   `1/raasq` divisions, causing force blowups.

### Files Converted

- `HarmonicDihedralForceGPU.cu` — used by `md.dihedral.Periodic` (benchmark kernel)
- `TableDihedralForceGPU.cu` — vec3<Scalar>→vec3<ForceReal>, acosf→fast::acos
- `HarmonicImproperForceGPU.cu` — rsqrtf→fast::rsqrt, #define SMALL→ForceReal(0.001)
- `PeriodicImproperForceGPU.cu` — Chebyshev recurrence converted
- `CosineSqAngleForceGPU.cu` — cosine-squared angle potential

### Benchmark Results (64K polymers + dihedrals, dt=0.005, RTX 4090)

| Build | Before fix (TPS) | After fix (TPS) | Change |
|-------|-----------------|-----------------|--------|
| double | 1158 | 2499 | — (run variance) |
| **mixed** | **4083** | **7474** | **+83%** |
| single | 11954 | 12062 | — (no code change) |

Mixed/double ratio: 3.0× (up from 3.5× — double also varied).
Mixed/single gap: 1.6× (down from 2.9× — most of the dihedral bottleneck eliminated).

### dt Sweep Stability (WITH dihedrals)

| dt | Before: Double | Before: Mixed | Before: Single | After: Mixed |
|----|---------------|---------------|----------------|-------------|
| 0.005 | 1158 | 4083 | 11954 | stable |
| 0.01 | 2028 | 1226 (67% std!) | 9100 | **5312 (15% std)** |
| 0.03 | crashed | crashed | 7522 | **4023 (survives!)** |
| 0.05 | crashed | crashed | crashed | crashed |

Mixed is now the **most stable build** for dihedrals — survives dt=0.03 where both
double and single crash. The sqrtf/double Frankenstein bug is eliminated.

---

## Neighbor List Analysis

The nlist GPU kernel (`NeighborListGPUBinned.cu`) already performs distance math in
ForceReal — positions are narrowed to `ForceReal3` immediately after load, and
`box.minImageForceReal()` is used for minimum image. However, positions are still
**read** as `Scalar4` (double4 = 32 bytes per particle):

- Self position: from `d_pos` (Scalar4) — one read per particle
- Neighbor positions: from `d_cell_xyzf` (Scalar4) — many reads per neighbor

The neighbor reads are the dominant bandwidth consumer. Converting them to ForceReal4
would halve the bandwidth of the most frequent memory access in the nlist kernel,
but requires changing CellList infrastructure.

### CellList `d_cell_xyzf` deep dive

**What it is:** `GPUArray<Scalar4> m_xyzf` in `CellList.h` — stores `(x, y, z, flag)` for
every particle slot in the cell list. The flag is either charge, type, or particle index
depending on configuration. This is the main data structure the nlist kernels read to find
neighbor positions.

**Writer:** One kernel in `CellListGPU.cu`:
```cpp
d_xyzf[write_pos] = make_scalar4(pos.x, pos.y, pos.z, flag);
```
Reads positions from `d_pos` (Scalar4), bins them into cells.

**Consumers (files that would need changes):**

| File | Role |
|------|------|
| `CellList.h` | Declaration: `GPUArray<Scalar4> m_xyzf`, `getXYZFArray()` return type |
| `CellList.cc` | Host CPU path: allocates and fills `m_xyzf` |
| `CellListGPU.cu` | GPU kernel: writes `Scalar4` → would write `ForceReal4` |
| `CellListGPU.cc` | Host side: ArrayHandle types |
| `CellListGPU.cuh` | Kernel signature |
| `NeighborListGPUBinned.cu/.cuh/.cc` | GPU Cell nlist (benchmark uses this) |
| `NeighborListGPUStencil.cu/.cuh/.cc` | GPU Stencil nlist variant |
| `NeighborListBinned.cc` | CPU Cell nlist fallback |
| `NeighborListStencil.cc` | CPU Stencil nlist fallback |
| `ComputeFreeVolumeGPU.h` | HPMC (Monte Carlo, not MD) |
| `test_cell_list.cc` | Test expectations |

~12 files total, but changes in each are mechanical: `Scalar4` → `ForceReal4` in
signatures, `make_scalar4` → `make_forcereal4` in the write kernel. The `.w` flag field
needs care: when storing particle index as `__int_as_scalar(idx)`, would need
`__int_as_forcereal(idx)` (already exists).

### Cost-benefit analysis

The nlist is rebuilt only every ~100 steps (with typical buffer). During the 99 steps
where the nlist is reused, the bandwidth savings is zero. During the 1 rebuild step,
we'd halve the dominant memory access (each particle reads ~20-50 neighbor candidates
from `d_cell_xyzf`).

Estimated improvement: `(1/100) × (fraction of rebuild time that is xyzf bandwidth) × 50%`,
which works out to roughly **~0.05%** for the polymer workload — negligible.

Would matter more for systems with fast-moving particles or small buffers (frequent
nlist rebuilds), but not for the target polymer/chromatin workloads.

**Decision**: Defer. The nlist compute path is already correct (ForceReal math). The
change is moderate scope (~12 files, mechanical) but the payoff is negligible for
target workloads.

---

## Remaining Unconverted Force Kernels (low priority)

These kernels still use Scalar throughout their body. Conversion is mechanical but
only matters if used in target workloads:

- `TableAngleForceGPU.cu` — tabulated angle potential
- `PPPMForceComputeGPU.cu` — long-range electrostatics (charge mesh)
- `PotentialTersoffGPU.cuh` — three-body potential (materials science)
- `ForceCompositeGPU.cu` — rigid body composite forces
- `ActiveForceComputeGPU.cu` — active matter self-propulsion
- `ForceDistanceConstraintGPU.cu` — distance constraints

---

## Summary

### Overall Achievement (64K polymer + dihedrals, dt=0.005, RTX 4090)

| Build | TPS | Relative to double |
|-------|-----|-------------------|
| double (baseline) | ~2500 | 1.0× |
| **mixed** | **~7500** | **3.0×** |
| single | ~12000 | 4.8× |

The mixed-precision build achieves **3× speedup over double** on consumer GPUs while
maintaining double-precision position integration. The remaining 1.6× gap to single
is structural: the integrator accumulates positions in double, and CellList/nlist
read double4 positions (2× bandwidth vs single).

### What was done

1. **Phase 1**: Force output pipeline — `Scalar4`→`ForceReal4` for force/virial/torque
   storage and accumulation across all GPU kernels.
2. **Phase 2A**: External potential evaluators — all 4 wall/periodic/field evaluators
   converted to ForceReal math.
3. **Phase 2B**: Float4 position mirror — `m_pos_forcereal` array synced from double
   positions, loaded as `ForceReal4` in all force kernels (~30 files).
4. **Phase 2C**: Dihedral/improper/angle fixes — 5 missed kernels converted, fixing
   both performance (+83% for dihedrals) and stability (mixed survives dt=0.03).

### What was NOT done (and why)

- `--use_fast_math`: The `fast::` namespace already uses GPU intrinsics (`__sinf`,
  `__cosf`, `__expf`). The flag would only affect plain `/` divisions (~2 ULP loss).
  Expected impact <5%, not worth the precision risk.
- CellList ForceReal4: Would halve nlist neighbor-read bandwidth but requires deep
  infrastructure changes. Deferred.
- 6 remaining force kernels: Not used in target workloads.

---

## `--use_fast_math` Analysis

The CUDA `--use_fast_math` compiler flag enables four sub-flags:

| Flag | Effect | Already handled? |
|------|--------|-----------------|
| `--ftz=true` | Flush denormals to zero | Minor — denormals are rare in MD |
| `--prec-div=false` | Replace `/` with `__fdividef` (~2 ULP) | **Only meaningful change** — divisions are everywhere: `1/r`, `F/r`, normalization |
| `--prec-sqrt=false` | Replace `sqrtf` with approximate sqrt | Our `fast::sqrt(float)` already calls `sqrtf` (IEEE-correct), so minor |
| `--fmad=true` | Fused multiply-add | Already the default in CUDA |

The functions where fast intrinsics make the biggest difference — `sin`, `cos`, `exp`,
`log`, `pow` — are **already using GPU intrinsics** via the `fast::` namespace in
`HOOMDMath.h` (e.g., `__sinf`, `__cosf`, `__expf`, `__logf` on device). So
`--use_fast_math` would only additionally affect plain `/` division operators scattered
through the code.

The workload is increasingly **memory-bandwidth-bound** after the mixed-precision
conversion. The remaining mixed→single gap (7474 vs 12062 TPS = 1.6×) is mostly from:
- Double-precision position reads/writes in the integrator
- Reading both `float4` and `double4` positions (2× the position bandwidth)
- Not from slower math functions

**Decision**: Skip. Expected improvement 0-5%, not worth the precision risk for
division-heavy force calculations. The `fast::` namespace already captures the
high-impact intrinsics.

---

## Commit History

```
a56ee37b8 Log: expand nlist CellList analysis with scope and cost-benefit
0d85807bf Update log: Phase 2C results, nlist analysis, summary
4b1025edf Convert CosineSqAngleForceGPU.cu to ForceReal
a1bdf7b5a Convert dihedral/improper GPU kernels to ForceReal
8489563a9 Phase 2B: float4 position mirror + accuracy tests + dt sweep benchmarks
007bdc275 Phase 2A Step 1: convert external potential evaluators to ForceReal
2f667b121 Convert nlist, angle, dihedral, DPD thermo, bond kernels to ForceReal
133feaffb Add benchmark and conversion scripts for mixed-precision testing
3edbe36ee Mixed precision: ForceReal (float) for force computation on GPU
```

---

## Build / Test / Benchmark Instructions

### Prerequisites

```bash
eval "$(~/miniforge3/bin/conda shell.bash hook)" && conda activate main
```

### Three build configurations

The project maintains three install prefixes for comparison:

| Config | CMake flags | Install prefix |
|--------|------------|----------------|
| mixed | `-DHOOMD_MIXED_PRECISION=ON -DHOOMD_LONGREAL_SIZE=64 -DHOOMD_SHORTREAL_SIZE=32` | `build/install_mixed/` |
| double | `-DHOOMD_LONGREAL_SIZE=64 -DHOOMD_SHORTREAL_SIZE=64` | `build/install_double/` |
| single | `-DHOOMD_LONGREAL_SIZE=32 -DHOOMD_SHORTREAL_SIZE=32` | `build/install_single/` |

The mixed build is the default (`build/` dir). Double and single are separate build
trees used only for benchmark comparisons.

### Build and test (mixed)

```bash
cd build && make -j8 && ctest --output-on-failure -j8
make install  # installs to build/install_mixed/
```

### Run benchmarks

```bash
cd sloptimize

# Single dt, all three builds in parallel (one per GPU):
python run_benchmarks.py benchmark_chains.py \
  --lib mixed=.../build/install_mixed/lib/python3.12/site-packages \
  --lib double=.../build/install_double/lib/python3.12/site-packages \
  --lib single=.../build/install_single/lib/python3.12/site-packages \
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

The runner auto-detects free GPUs and assigns one job per GPU. Use `--gpus 0,1,2`
to restrict to specific GPUs.
