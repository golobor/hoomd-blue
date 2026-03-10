# Design & Analysis

Technical decisions, benchmark data, accuracy measurements, and analysis of
optimizations attempted, adopted, and rejected.

---

## Background: OpenMM vs HOOMD Precision Strategies

### How OpenMM achieves equal gaming/datacenter GPU performance

OpenMM uses three key techniques:

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
   `x * 2^32`, enabling exact atomic accumulation. HOOMD doesn't need this — it avoids
   atomics by assigning one thread-group per particle with warp-level reduction.

### Where precision actually matters in MD

| Operation | Precision needed | Justification |
|-----------|-----------------|---------------|
| Pair force evaluation (F = -dV/dr) | **Float OK** | Errors comparable to thermal noise |
| Distance calculation (r_ij) | **Float OK** | Periodic wrapping keeps coordinates O(L) |
| Position accumulation (x += v·Δt) | **Double critical** | Float accumulation causes O(ε_float · x) drift per step |
| Velocity update | **Double preferred** | Accumulation argument, though less severe |
| Energy reporting | **Double preferred** | Summing ~N small terms; float loses significance |
| Neighbor list building | **Float OK** | Buffer distance absorbs rounding errors |

### HOOMD's existing infrastructure

HOOMD already had `ShortReal` (`float`) and `ForceReal` (`float`) type aliases but used them
only in HPMC. The MD kernels all used `Scalar` (`double`). The type infrastructure, helper
functions (`loadPosForceReal`, `minImageForceReal`, `fast::` math namespace), and
conditional compilation (`HOOMD_MIXED_PRECISION`) existed — they just weren't wired up.

---

## Key Design Decisions

- **OpenMM's `pos + correction` split** over Kahan summation: cheaper (no extra adds per step)
  and integrates naturally with separate force-kernel vs integrator precision.

- **No fixed-point force accumulation**: HOOMD uses warp-level reduction, not atomics.
  Fixed-point conversion would add overhead without benefit.

- **Float4 position mirror** (not replacing `m_pos`): We added `m_pos_forcereal` alongside
  `m_pos` rather than changing `getPositions()` return type. This avoids breaking HPMC
  (30+ call sites), GSD, snapshots, and MPI. Memory cost: +N × 16 bytes (~1 MB for 64K).

- **Single sync point**: Rather than plumbing `d_pos_forcereal` through every integrator
  kernel, a single `syncPositionsForceReal()` call in `ForceCompute::compute()` copies
  `m_pos → m_pos_forcereal` once per timestep.

- **Gaming GPUs as sole test platform**: RTX 4090's 1:64 FP64:FP32 ratio makes
  success/failure unambiguous. Datacenter GPUs get the same correct results, just smaller speedup.

---

## What Was Done

| Phase | Description | Key change |
|-------|-------------|------------|
| **1** | Force output pipeline | `Scalar4` → `ForceReal4` for force/virial/torque storage |
| **2A** | External evaluators | 4 wall/periodic/field evaluators → ForceReal math |
| **2B** | Float4 position mirror | `m_pos_forcereal` array, loaded as `ForceReal4` in ~30 force kernels |
| **2C** | Dihedral/improper/angle fixes | 5 missed kernels converted, fixing performance (+83%) and stability |

---

## What Was NOT Done (and Why)

### `--use_fast_math` — Skipped (<5% expected gain)

The CUDA `--use_fast_math` flag enables:

| Sub-flag | Effect | Already handled? |
|----------|--------|-----------------|
| `--ftz=true` | Flush denormals to zero | Minor — denormals rare in MD |
| `--prec-div=false` | Approximate division (~2 ULP) | **Only meaningful change** |
| `--prec-sqrt=false` | Approximate sqrt | `fast::sqrt(float)` already uses `sqrtf` |
| `--fmad=true` | Fused multiply-add | Already CUDA default |

The high-impact intrinsics (`sin`, `cos`, `exp`, `log`) are **already GPU intrinsics** via
the `fast::` namespace in `HOOMDMath.h`. The workload is increasingly bandwidth-bound after
conversion — the mixed→single gap (1.6×) comes from double-precision integrator I/O, not
slower math. Expected improvement 0-5%, not worth the precision risk for division-heavy
force calculations.

### CellList ForceReal4 — Deferred (negligible payoff)

The neighbor list reads positions from `d_cell_xyzf` (`Scalar4` = 32 bytes). Converting to
`ForceReal4` would halve this bandwidth, but the nlist is only rebuilt every ~100 steps.

Estimated improvement: `(1/100) × (fraction of rebuild time from xyzf reads) × 50%`
≈ **~0.05%** for polymer workloads.

Scope: ~12 files (CellList.h, CellListGPU.cu, NeighborListGPUBinned.cu, etc.), mechanical
but touches infrastructure shared with HPMC.

### 6 Remaining Force Kernels — Not needed

`TableAngleForceGPU.cu`, `PPPMForceComputeGPU.cu`, `PotentialTersoffGPU.cuh`,
`ForceCompositeGPU.cu`, `ActiveForceComputeGPU.cu`, `ForceDistanceConstraintGPU.cu`
— not used in target polymer/chromatin workloads.

---

## Accuracy & Benchmark Results

See [BENCHMARKING.md](BENCHMARKING.md) for full accuracy measurements, performance
tables across phases, dt sweep data, and scaling analysis.

**Headline**: 3× speedup over double on RTX 4090 (64K polymers + dihedrals), with
force accuracy indistinguishable from single-precision (mean relative error ~3.6×10⁻⁷).

The remaining 1.6× gap to single is structural: double-precision integrator I/O,
position sync kernel, and double4 CellList reads — all fundamental to the
mixed-precision design.
