# Mixed-Precision HOOMD-blue for Consumer GPUs

AI-driven optimization of [HOOMD-blue](https://github.com/glotzerlab/hoomd-blue) v6.1.1
to run efficiently on gaming GPUs (RTX 3090/4090) by eliminating unnecessary FP64
from the GPU hot path.

## The problem

HOOMD-blue uses double precision (FP64) everywhere on the GPU. Consumer GPUs have an
FP64:FP32 throughput ratio of **1:64** (RTX 4090), making every double-precision
operation 64× slower than single-precision.

For coarse-grained polymer/chromatin simulations, float is more than adequate for
force evaluation — only position integration truly needs double.

## Results

**64K-particle polymer system with dihedrals, dt=0.005, RTX 4090:**

| Build | TPS | vs Double |
|-------|-----|-----------|
| double (baseline) | ~2,500 | 1.0× |
| **mixed (this fork)** | **~7,500** | **3.0×** |
| single | ~12,000 | 4.8× |

- **3× speedup** over double with double-precision position integration preserved
- Force accuracy: mean relative error ~3.6×10⁻⁷ vs double (consistent with float32 ε)
- Energy conservation: indistinguishable from double over 100 steps
- Stability: mixed survives dt=0.03 with dihedrals where double and single crash

Scaling: at 200K particles, mixed achieves **~3× double** (bandwidth-bound workloads
benefit more from halved data widths at larger system sizes).

## Further reading

- [**DESIGN.md**](DESIGN.md) — Technical decisions, accuracy data, benchmark tables,
  and analysis of what was/wasn't done and why.
- [**CHANGELOG.md**](CHANGELOG.md) — Full implementation diary: every phase, every bug,
  every file changed. The unabridged developer log.

## Build

### Prerequisites

```bash
eval "$(~/miniforge3/bin/conda shell.bash hook)" && conda activate main
```

### Three build configurations

| Config | CMake flags | Install prefix |
|--------|------------|----------------|
| mixed | `-DHOOMD_MIXED_PRECISION=ON -DHOOMD_LONGREAL_SIZE=64 -DHOOMD_SHORTREAL_SIZE=32` | `build/install_mixed/` |
| double | `-DHOOMD_LONGREAL_SIZE=64 -DHOOMD_SHORTREAL_SIZE=64` | `build/install_double/` |
| single | `-DHOOMD_LONGREAL_SIZE=32 -DHOOMD_SHORTREAL_SIZE=32` | `build/install_single/` |

### Build and test (mixed)

```bash
cd build && make -j8 && ctest --output-on-failure -j8
make install  # installs to build/install_mixed/
```

### Switch between builds

```bash
source sloptimize/use_hoomd.sh mixed    # forces=float, integration=double
source sloptimize/use_hoomd.sh double   # original, everything double
source sloptimize/use_hoomd.sh single   # everything float
```

Auto-detects repo root and Python version from the script location.

## Run benchmarks

```bash
cd sloptimize

# Single dt, all three builds in parallel (one per GPU):
python run_benchmarks.py benchmark_chains.py \
  --lib mixed=../build/install_mixed/lib/python3.12/site-packages \
  --lib double=../build/install_double/lib/python3.12/site-packages \
  --lib single=../build/install_single/lib/python3.12/site-packages \
  --no-dt \
  -- 64000 200

# dt sweep:
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
to restrict to specific devices.

## Commit history

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

## License

BSD 3-Clause — same as upstream HOOMD-blue. See [LICENSE](../LICENSE).
