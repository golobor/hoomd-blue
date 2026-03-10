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

- [**DESIGN.md**](DESIGN.md) — Technical decisions, accuracy data,
  and analysis of what was/wasn't done and why.
- [**BENCHMARKING.md**](BENCHMARKING.md) — How to build comparison configurations,
  run benchmarks, and full results (accuracy, dt sweeps, scaling).
- [**CHANGELOG.md**](CHANGELOG.md) — Full implementation diary: every phase, every bug,
  every file changed.

## Build

### Prerequisites

```bash
eval "$(~/miniforge3/bin/conda shell.bash hook)" && conda activate main
```

### Build and test

```bash
cd build && make -j8 && ctest --output-on-failure -j8
make install  # installs to build/install_mixed/
```

The mixed build uses CMake flags:
`-DHOOMD_MIXED_PRECISION=ON -DHOOMD_LONGREAL_SIZE=64 -DHOOMD_SHORTREAL_SIZE=32`

To build the double/single comparison configurations for benchmarking, see
[BENCHMARKING.md](BENCHMARKING.md).

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
