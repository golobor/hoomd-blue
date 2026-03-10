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

## Benchmark Results

### Headline (64K polymers + dihedrals, dt=0.005, RTX 4090)

| Build | TPS | vs Double |
|-------|-----|-----------|
| double | ~2,500 | 1.0× |
| **mixed** | **~7,500** | **3.0×** |
| single | ~12,000 | 4.8× |

### Progression Through Phases

**With dihedrals (64K particles, dt=0.005):**

| Phase | Mixed TPS | vs Double | vs Single |
|-------|-----------|-----------|-----------|
| Phase 2A (external evaluators only) | 3,059 | +16.5% | 0.25× |
| Phase 2B (float4 positions) | 4,262 | +62.3% | 0.35× |
| Phase 2C (dihedral fix) | 7,474 | +199% | 0.62× |

**Without dihedrals (64K particles, dt=0.005):**

| Build | TPS | vs Double |
|-------|-----|-----------|
| Double | 4,504 | — |
| Mixed (Phase 2B) | 11,218 | +149% |
| Single | 19,408 | +331% |

### dt Sweep — With Dihedrals (64K particles)

| dt | Double | Mixed | Single |
|----|--------|-------|--------|
| 0.005 | 2,499 | 7,474 | 12,062 |
| 0.01 | — | 5,312 (15% std) | 9,100 |
| 0.03 | crashed | **4,023 (survives!)** | 7,522 |
| 0.05 | crashed | crashed | crashed |

Mixed is the **most stable build** for dihedrals at large dt.

### dt Sweep — Without Dihedrals (64K particles)

| dt | Double | Mixed | Single |
|----|--------|-------|--------|
| 0.005 | 4,328 | 10,825 | 19,548 |
| 0.01 | 4,125 | 9,465 | 15,261 |
| 0.03 | 3,278 | 7,522 | 4,828 |
| 0.05 | 1,837 | 5,140 | 5,345 |
| 0.1 | 1,882 | 4,970 | 8,088 |

All builds stable. Mixed consistently ~2.5× double. At large dt single degrades
(likely more nlist rebuilds from float integrator drift).

### 200K Particles — Without Dihedrals

| dt | Double | Mixed | Single |
|----|--------|-------|--------|
| 0.005 | 1,485 | 4,384 | 6,760 |
| 0.01 | 1,361 | 3,840 | 5,824 |
| 0.03 | 1,168 | 3,061 | 4,539 |

Mixed achieves ~2.95× double at 200K — bandwidth-bound workloads benefit more at
larger system sizes. Mixed-to-single gap narrows to 1.54× (from 1.80× at 64K).

---

## Remaining Mixed→Single Performance Gap

The 1.6× gap between mixed (~7,500 TPS) and single (~12,000 TPS) is structural:

1. **Integrator I/O**: Read/write `double4` positions for integration accuracy
2. **Position sync**: `syncPositionsForceReal()` reads double4, writes float4 (
   one extra kernel per timestep)
3. **CellList**: Stores `Scalar4` positions — nlist reads double4 neighbors

These are fundamental to the mixed-precision design (double integration is the point)
and cannot be further optimized without sacrificing the precision guarantees.
