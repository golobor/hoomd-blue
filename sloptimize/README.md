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
| double (baseline) | 2,371 ± 38 | 1.0× |
| **mixed (this fork)** | **10,197 ± 161** | **4.30×** |
| single | 12,495 ± 732 | 5.27× |

- **4.3× speedup** over double with double-precision position integration preserved
- **+290%** over unmodified upstream double (same codebase, same hardware)
- Force accuracy: mean relative error ~3.6×10⁻⁷ vs double (consistent with float32 ε)
- Energy conservation: indistinguishable from double over 100 steps
- Patchy particles: **8.8× speedup** (anisotropic evaluator is heavily compute-bound)

Scaling: at 256K particles, mixed achieves **~3.3× double** (bandwidth-bound workloads
benefit more from halved data widths at larger system sizes).

## Quick install

```bash
# Clone and install into a conda environment (requires NVIDIA GPU + drivers)
git clone --branch mixed-precision https://github.com/golobor/hoomd-blue.git
bash hoomd-blue/sloptimize/install.sh

# Use it
conda activate hoomd-mixed
python -c "import hoomd; print(hoomd.version.floating_point_precision)"  # (64, 32)
```

See [install.sh](install.sh) for options (`--env`, `--jobs`, `--python`, etc.).

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
cd build
cmake .. -DHOOMD_LONGREAL_SIZE=64 -DHOOMD_SHORTREAL_SIZE=32 -DENABLE_GPU=ON
make -j8 && ctest --output-on-failure -j8
cmake --install . --prefix install_mixed
```

The mixed-precision mode is activated by setting `SHORTREAL_SIZE != LONGREAL_SIZE`.
This auto-defines `HOOMD_MIXED_PRECISION` at compile time (it is **not** a CMake
variable). Always pass the precision flags explicitly — CMake caches them, so a
stale cache can silently produce the wrong build. Verify with:

```bash
cd /tmp && PYTHONPATH=<install>/lib/python3.12/site-packages \
  python3 -c "import hoomd; print(hoomd.version.floating_point_precision)"
# Expected: (64, 32)
```

To build the double/single comparison configurations for benchmarking, see
[BENCHMARKING.md](BENCHMARKING.md).

## Key commits

```
82e2842e2  Update upstream regression check with post-dihedral-clamp numbers
297dc41cb  Dihedral epsilon clamp: +35% TPS, fixes crash at large dt
410e68af0  Use cancellation-free rotmat3 in ForceReal for patchy rotation
919bfd189  Fix aniso pair alignment bug, make minImageForceReal unconditional
8489563a9  Phase 2B: float4 position mirror + accuracy tests + dt sweep benchmarks
007bdc275  Phase 2A Step 1: convert external potential evaluators to ForceReal
3edbe36ee  Mixed precision: ForceReal (float) for force computation on GPU
```

## License

BSD 3-Clause — same as upstream HOOMD-blue. See [LICENSE](../LICENSE).
