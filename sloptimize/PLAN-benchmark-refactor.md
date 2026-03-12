# Plan: Benchmark Infrastructure Refactor

**Status: NOT STARTED**

## Problem

The benchmark infrastructure has 5 scripts that duplicate the same simulation setup
(lattice builder, force factory, equilibration): `benchmark_chains.py` (612 lines),
`benchmark_dt_stability.py` (684 lines), `benchmark_suite.py` (1144 lines),
`test_accuracy.py` (513 lines), and `profile_kernels.py` (105 lines, imports from
benchmark_chains). Plus 4 throwaway diagnostic scripts.

`benchmark_suite.py` was a monolithic rewrite that lost flexibility:
- No `--attract` (DPD attraction pair force)
- No configurable `n_particles` / `chain_length` (hardcoded 64K/200)
- No configurable step counts
- Patchy params hardcoded, not CLI-configurable

## Architecture

**4 Python files** (down from ~10):

### 1. `benchlib.py` — shared library (~250 lines)

Extract from `benchmark_chains.py` (the most complete version):
- `TeeWriter` class
- `make_lattice_chains(n_chains, chain_length, sphere_radius)`
- `make_patch_directors(n_patches)` — Fibonacci sphere placement
- `make_forces(sphere_radius, ...)` — with ALL force options:
  - `include_angle=True`, `include_dihedral=True`
  - `dpd_A=5.0`
  - `attract=None` — tuple `(strength, rcut)` for second DPDConservative pair
  - `patchy=None` — tuple `(eps, sigma, alpha, omega, rcut, n_patches)`
- `equilibrate_and_save(device, n_particles, chain_length, equil_steps, ..., save_path=None)`
  - With patchy-aware sub-phase ramp (dt/gamma ramp for anisotropic forces)
  - Returns `(gsd_path, sphere_radius)`
- `add_common_args(parser)` — adds shared CLI flags to any argparse parser:
  - Positional: `gpu_id`, `n_particles` (default 64000), `chain_length` (default 200)
  - Optional: `--no-angle`, `--no-dihedral`, `--dpd-A`, `--attract STRENGTH,RCUT`,
    `--patchy EPS,SIGMA,ALPHA,OMEGA,RCUT,NPATCHES`, `--dt` (single float, default 0.005),
    `--save-state PATH`, `--load-state PATH`, `--equilibrate-only`, `--log FILE`
- `load_or_equilibrate(args, device)` — convenience: checks `args.load_state`,
  loads GSD+metadata if exists, else calls `equilibrate_and_save()`.
  Returns `(gsd_path, sphere_radius)`.
- `parse_force_kwargs(args)` — extracts `include_angle`, `include_dihedral`,
  `dpd_A`, `attract`, `patchy` from parsed args into a dict for `make_forces(**kwargs)`.

No `main()`, no CLI entry point — pure library.

### 2. `benchmark_tps.py` — Langevin TPS measurement (~150 lines)

Replaces `benchmark_chains.py`. Single-dt Langevin performance measurement.

- Uses `benchlib.add_common_args()` for full CLI flexibility
- Workflow: `load_or_equilibrate()` → Langevin warmup → benchmark with per-chunk TPS streaming
- Reports: TPS mean/std/range, ns/day, temperature stability
- Constants (from benchmark_chains.py): 10K warmup + 100K benchmark, report every 10K
- Supports `--equilibrate-only` (just equilibrate and save state, exit)

CLI: `python benchmark_tps.py [gpu_id] [n_particles] [chain_length] [options]`

### 3. `benchmark_stability.py` — NVE + force accuracy (~350 lines)

Replaces `benchmark_dt_stability.py` + accuracy parts of `benchmark_suite.py`.
Single-dt NVE and force accuracy measurement.

- Uses `benchlib.add_common_args()` for full CLI flexibility
- Extra args: `--tests {all,nve,accuracy}`, `--out-dir PATH` (for .npz/.json)
- `run` subcommand (default): single-dt NVE energy conservation + force accuracy
  - NVE: runs 50K steps, reports drift/RMS/max_deviation/temperature/status
  - Accuracy: computes per-force-type forces at step 0, saves to .npz
- `compare` subcommand: loads .npz from multiple `--out-dir` directories,
  prints cross-build force/energy comparison tables
- Supports `--equilibrate-only`

CLI: `python benchmark_stability.py [gpu_id] [n_particles] [chain_length] [options]`
     `python benchmark_stability.py compare dir1 dir2 [dir3 ...]`

### 4. `profile_kernels.py` — GPU kernel profiling (~105 lines, mostly unchanged)

- Change `from benchmark_chains import equilibrate_and_save, _make_forces`
  → `from benchlib import equilibrate_and_save, make_forces`
- Otherwise keep as-is (unique nsys profiling purpose)

## Orchestration

### `run_benchmarks.py` — multi-GPU parallel runner (keep, update)

Already has the two-phase equilibrate-once-then-sweep design. Changes:
- Update default script name references in help text
- Verify it works with both `benchmark_tps.py` and `benchmark_stability.py`
- The `--no-dt` / dt-sweep logic stays as-is

### `run_full_benchmarks.sh` — recommended test suite launcher (keep, update)

Update to use new script names. The preset workloads (chains, nodih, patchy)
become explicit CLI flag combinations:

```bash
# "chains" workload = default (angle + dihedral)
python benchmark_tps.py 0 64000 200

# "nodih" workload = no dihedral
python benchmark_tps.py 0 64000 200 --no-dihedral

# "patchy" workload = patchy + no dihedral
python benchmark_tps.py 0 64000 200 --no-dihedral --patchy 1.0,0.5,0.6,20.0,1.5,2
```

The three-phase pipeline stays:
1. Equilibrate reference (double) → save state
2. Force accuracy with shared double state (all builds)
3. Per-build NVE + TPS at each dt

## Shared State Workflow

Equilibration lives in `benchlib.equilibrate_and_save()`. Both scripts share
equilibrated states via `--save-state` / `--load-state` CLI flags:

1. Phase 1 — equilibrate once:
   `benchmark_tps.py 0 64000 200 --equilibrate-only --save-state /tmp/state.gsd`

2. Phase 2 — reuse for all runs:
   - `benchmark_tps.py 0 ... --load-state /tmp/state.gsd --dt 0.005`
   - `benchmark_tps.py 0 ... --load-state /tmp/state.gsd --dt 0.01`
   - `benchmark_stability.py 0 ... --load-state /tmp/state.gsd --dt 0.005`

`run_benchmarks.py` already orchestrates this (Phase 1 passes `--equilibrate-only
--save-state`, Phase 2 passes `--load-state`).

For cross-build accuracy comparison, double's equilibrated state is shared:
all builds load the same GSD and compute forces at step 0.

## Files to Delete

After new scripts are working and verified:

| File | Reason |
|------|--------|
| `benchmark_chains.py` | Replaced by `benchlib.py` + `benchmark_tps.py` |
| `benchmark_dt_stability.py` | Replaced by `benchmark_stability.py` |
| `benchmark_suite.py` | Replaced by `benchmark_tps.py` + `benchmark_stability.py` |
| `test_accuracy.py` | Replaced by `benchmark_stability.py compare` |
| `diagnose_dihedral.py` | Throwaway diagnostic (dihedral bug fixed) |
| `direct_dih_test.py` | Throwaway diagnostic |
| `minimal_dih_test.py` | Throwaway diagnostic |
| `quick_accuracy_retest.py` | Throwaway diagnostic |

**Do NOT delete until new scripts are tested and produce correct output.**

## Documentation Update

Update `BENCHMARKING.md`:
- "Running Benchmarks" section: replace `benchmark_chains.py` examples with
  `benchmark_tps.py` and `benchmark_stability.py`
- "Workload" section: document how workload variations map to CLI flags
- Update "Dihedral Numerical Stability Fix": change `SMALL = ForceReal(0.001)`
  to `SMALL = ForceReal(1e-12)` and update the description to reflect that
  all builds now crash at the same dt threshold (physics limit, not float issue)
- Update dt sweep table for "With Dihedrals" — mixed no longer survives dt=0.05
  (the SMALL=0.001 that allowed that is gone, replaced with 1e-12)

## Implementation Order

- [ ] Step 1: Create `benchlib.py` — extract shared code from `benchmark_chains.py`
- [ ] Step 2: Create `benchmark_tps.py` — Langevin TPS using benchlib
- [ ] Step 3: Create `benchmark_stability.py` — NVE + accuracy using benchlib
- [ ] Step 4: Update `profile_kernels.py` — import from benchlib
- [ ] Step 5: Test all new scripts manually (single runs)
- [ ] Step 6: Update `run_full_benchmarks.sh` — use new script names
- [ ] Step 7: Test via `run_full_benchmarks.sh --quick`
- [ ] Step 8: Delete old files (8 files)
- [ ] Step 9: Update `BENCHMARKING.md`
- [ ] Step 10: Commit and push
- [ ] Step 11: Delete this plan file

## Verification Checklist

- [ ] `python benchmark_tps.py 0 64000 200` reproduces headline TPS
- [ ] `python benchmark_tps.py 0 64000 200 --no-dihedral` works
- [ ] `python benchmark_tps.py 0 64000 200 --patchy 1.0,0.5,0.6,20.0,1.5,2` works
- [ ] `python benchmark_tps.py 0 64000 200 --attract -0.5,1.5` works
- [ ] `python benchmark_stability.py 0 64000 200 --tests accuracy --out-dir /tmp/test` saves .npz
- [ ] `python benchmark_stability.py 0 64000 200 --tests nve` prints NVE table
- [ ] `python benchmark_stability.py compare /tmp/d /tmp/m` prints comparison
- [ ] `--save-state` / `--load-state` round-trip works across scripts
- [ ] `--equilibrate-only` works in both scripts
- [ ] `profile_kernels.py` imports from benchlib without errors
- [ ] `bash run_full_benchmarks.sh --quick` completes end-to-end
- [ ] `BENCHMARKING.md` examples are copy-pasteable and correct
