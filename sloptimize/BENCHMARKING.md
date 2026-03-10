# Benchmarking

How to build comparison configurations, run benchmarks, and the full results.

---

## Comparison Builds

All three builds — **mixed**, **double**, **single** — are compiled from **this same
codebase**. Precision is controlled entirely by CMake flags:

| Config | CMake flags | What it does |
|--------|------------|--------------|
| mixed | `-DHOOMD_MIXED_PRECISION=ON -DHOOMD_LONGREAL_SIZE=64 -DHOOMD_SHORTREAL_SIZE=32` | Forces in float, integration in double |
| double | `-DHOOMD_LONGREAL_SIZE=64 -DHOOMD_SHORTREAL_SIZE=64` | Original upstream behavior |
| single | `-DHOOMD_LONGREAL_SIZE=32 -DHOOMD_SHORTREAL_SIZE=32` | Everything float (theoretical max speed) |

The double and single builds exist only for benchmark comparison. Regular users only
need the mixed build.

### Building all three

Each configuration needs its own build tree and install prefix:

```bash
# Mixed (the default build/ directory)
cd build && make -j8 && make install   # → build/install_mixed/

# Double (separate build tree)
mkdir -p build_double && cd build_double
cmake .. -DHOOMD_LONGREAL_SIZE=64 -DHOOMD_SHORTREAL_SIZE=64 \
         -DCMAKE_INSTALL_PREFIX=../build/install_double
make -j8 && make install

# Single (separate build tree)
mkdir -p build_single && cd build_single
cmake .. -DHOOMD_LONGREAL_SIZE=32 -DHOOMD_SHORTREAL_SIZE=32 \
         -DCMAKE_INSTALL_PREFIX=../build/install_single
make -j8 && make install
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
| double | 2,538 ± 46 | 1.0× |
| **mixed** | **7,741 ± 298** | **3.05×** |
| single | 12,093 ± 347 | 4.77× |

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
| 0.005 | 2,538 ± 46 | 7,741 ± 298 | 12,093 ± 347 |
| 0.01 | 2,144 ± 59 | 5,616 ± 103 | 9,391 ± 420 |
| 0.03 | 1,984 ± 60 | 4,775 ± 90 | 7,641 ± 52 |
| 0.05 | crashed | crashed | crashed |
| 0.1 | crashed | crashed | crashed |

All builds stable through dt=0.03. All crash at dt=0.05 (Langevin dynamics
with dihedrals becomes unstable). Mixed delivers 2.4–3.1× over double across
all stable dt values.

### Without Dihedrals (64K particles)

| dt | Double | Mixed | Single |
|----|--------|-------|--------|
| 0.005 | 4,374 ± 77 | 11,141 ± 257 | 18,330 ± 59 |
| 0.01 | 4,178 ± 147 | 9,700 ± 223 | 16,168 ± 373 |
| 0.03 | 3,540 ± 131 | 7,617 ± 110 | 11,643 ± 103 |
| 0.05 | 2,730 ± 69 | 4,894 ± 60 | 8,048 ± 12 |
| 0.1 | 2,677 ± 23 | 4,961 ± 7 | 8,255 ± 5 |

All builds stable at all dt values. Mixed consistently 1.8–2.5× double.
At large dt (0.05–0.1) all builds' TPS plateaus — the overhead of more
frequent neighbor list rebuilds dominates.

### Without Dihedrals (256K particles)

| dt | Double | Mixed | Single |
|----|--------|-------|--------|
| 0.005 | 1,189 ± 13 | 3,730 ± 92 | 5,336 ± 159 |
| 0.01 | 1,097 ± 24 | 3,202 ± 69 | 4,676 ± 86 |
| 0.03 | 976 ± 37 | 2,547 ± 27 | 3,641 ± 11 |
| 0.05 | 775 ± 31 | 1,659 ± 8 | 2,518 ± 15 |
| 0.1 | 772 ± 19 | 1,686 ± 5 | 2,475 ± 11 |

All builds stable. Mixed achieves ~3.1× double at dt=0.005 — bandwidth-bound
workloads benefit more at larger system sizes. Mixed-to-single gap narrows to
1.43× (from 1.64× at 64K), consistent with memory bandwidth becoming the
dominant bottleneck.

### With Attraction, No Dihedrals (64K particles, A=-0.5, r_cut=1.5)

Adds a second DPDConservative pair force (attractive, separate neighbor list).

| dt | Double | Mixed | Single | Mixed/Double |
|----|--------|-------|--------|-------------|
| 0.005 | 2,013 ± 85 | 7,012 ± 126 | 10,200 ± 100 | 3.48× |
| 0.01 | 1,880 ± 126 | 5,689 ± 186 | 8,730 ± 164 | 3.03× |
| 0.03 | 1,527 ± 95 | 3,954 ± 97 | 5,763 ± 129 | 2.59× |
| 0.05 | 1,079 ± 53 | 2,364 ± 44 | 3,732 ± 49 | 2.19× |
| 0.1 | 1,078 ± 24 | 2,439 ± 21 | 3,821 ± 16 | 2.26× |

All stable at every dt. The second pair force increases compute intensity,
giving mixed a higher speedup (3.5× vs 2.5× without attraction at dt=0.005) —
more pair-force compute means more float savings to harvest.

### Patchy Particles, No Dihedrals (64K particles, PatchyGaussian)

Uses `AnisoPotentialPairPatchyGauss` with parameters `eps=1.0, sigma=0.5,
alpha=0.6, omega=20, r_cut=1.5, npatches=2`. This exercises the anisotropic
pair kernel which evaluates orientational (quaternion) math.

| dt | Double | Mixed | Single | Mixed/Double |
|----|--------|-------|--------|-------------|
| 0.005 | 329 ± 19 | 334 ± 18 | 5,211 ± 165 | 1.02× |
| 0.01 | 363 ± 29 | 353 ± 30 | 4,566 ± 157 | 0.97× |
| 0.03 | 367 ± 20 | 376 ± 20 | 3,945 ± 89 | 1.02× |
| 0.05 | 342 ± 10 | 338 ± 11 | 2,738 ± 26 | 0.99× |
| 0.1 | 330 ± 3 | 331 ± 2 | 2,777 ± 7 | 1.00× |

**Mixed ≈ Double** — no speedup. The anisotropic pair evaluator
(`PatchEnvelope`, `PairModulator`) uses `Scalar` (double) quaternion and
orientation math internally. Only the outer force output pipeline and
minimum-image call use `ForceReal` (float), which is a tiny fraction of the
workload. **Single is 8–16× faster** because all `Scalar` operations become
float.

**Takeaway**: Mixed precision only helps when `ForceReal` dominates compute.
For orientation-heavy potentials, the evaluator internals must also be
converted to `ForceReal` to see gains.

### Patchy Particles — After Rotation Optimization (rotmat3 → ForceReal)

Same parameters as above. `rotmat3(quat)` constructor rewritten with
cancellation-free formula (`1 − 2c² − 2d²` diagonals), and PatchEnvelope
rotation unified to use `rotmat3<ForceReal>` on both CPU and GPU (eliminating
all double-precision FLOPs from the quaternion rotation path).

| dt | Double | Mixed | Single | Mixed/Double |
|----|--------|-------|--------|-------------|
| 0.005 | 328 ± 19 | 351 ± 20 | 5,181 ± 160 | 1.07× |
| 0.01 | 361 ± 28 | 372 ± 32 | 4,532 ± 155 | 1.03× |
| 0.03 | 367 ± 20 | 389 ± 24 | 3,941 ± 111 | 1.06× |
| 0.05 | 340 ± 12 | 345 ± 10 | 2,691 ± 21 | 1.01× |
| 0.1 | 332 ± 2 | 351 ± 2 | 2,837 ± 11 | 1.06× |

**Marginal improvement** (~3–6% mixed over double). The rotation was NOT the
bottleneck — the rest of the evaluator pipeline (`PairModulator::evaluate()`,
`PatchEnvelope` distance/angle math, pair loop position arithmetic in the
`AnisoPotentialPairGPU` kernel) still operates in `Scalar` (double).

**Conclusion**: To bring mixed close to single for anisotropic potentials, the
_entire_ aniso pair kernel and evaluator chain would need ForceReal conversion,
analogous to what was done for isotropic `PotentialPairGPU`. The rotation fix
is still valuable as a correctness improvement (float-safe cancellation-free
formula) but does not unlock the expected throughput gain on its own.

---

## Remaining Mixed→Single Performance Gap

The 1.6× gap between mixed (~7,500 TPS) and single (~12,000 TPS) is structural:

1. **Integrator I/O**: Read/write `double4` positions for integration accuracy
2. **Position sync**: `syncPositionsForceReal()` reads double4, writes float4
   (one extra kernel per timestep)
3. **CellList**: Stores `Scalar4` positions — nlist reads double4 neighbors

These are fundamental to the mixed-precision design (double integration is the point)
and cannot be further optimized without sacrificing the precision guarantees.
