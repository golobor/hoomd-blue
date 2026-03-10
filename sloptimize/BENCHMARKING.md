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

The runner auto-detects free GPUs and assigns one job per GPU (round-robin assignment,
`max_workers = min(n_jobs, n_gpus)`). Use `--gpus 0,1,2` to restrict to specific devices.

### Workload

`benchmark_chains.py` — 64K particles (or configurable), 320 chains × 200 monomers,
pair(Gaussian A=5) + harmonic bonds + wall(Gaussian) + angle + dihedral, Langevin
integrator, dt=0.005. Protocol: 10K warmup + 100K benchmark steps, report every 10K.

---

## Results

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
| 0.005 | 2,499 | 7,474 | 12,062 |
| 0.01 | — | 5,312 (15% std) | 9,100 |
| 0.03 | crashed | **4,023 (survives!)** | 7,522 |
| 0.05 | crashed | crashed | crashed |

Mixed is the **most stable build** for dihedrals at large dt.

### Without Dihedrals (64K particles)

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
2. **Position sync**: `syncPositionsForceReal()` reads double4, writes float4
   (one extra kernel per timestep)
3. **CellList**: Stores `Scalar4` positions — nlist reads double4 neighbors

These are fundamental to the mixed-precision design (double integration is the point)
and cannot be further optimized without sacrificing the precision guarantees.
