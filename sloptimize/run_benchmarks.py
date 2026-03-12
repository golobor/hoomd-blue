#!/usr/bin/env python3
"""Run HOOMD benchmarks in parallel across free GPUs.

Automatically detects which GPUs are idle (no compute processes),
assigns one benchmark job per free GPU, and streams output with labels.

Usage
-----
    python run_benchmarks.py <script> [--lib label=path ...] [-- script_args...]

The placeholder ``{label}`` in script_args is replaced with each job's label.
This enables per-build state files, e.g.:

    python run_benchmarks.py benchmark_tps.py \
        --lib mixed=... --lib double=... \
        -- --equilibrate-only --save-state /tmp/eq_{label}.gsd

Examples
--------
    # Phase 1: equilibrate (one per build, in parallel)
    python run_benchmarks.py benchmark_tps.py \
        --lib mixed=.../install_mixed/lib/python3.12/site-packages \
        --lib double=.../install_double/lib/python3.12/site-packages \
        -- --equilibrate-only --save-state /tmp/eq_{label}.gsd

    # Phase 2: benchmark (loads saved state, skips equilibration)
    python run_benchmarks.py benchmark_tps.py \
        --lib mixed=... --lib double=... \
        -- --load-state /tmp/eq_{label}.gsd

    # Restrict to specific GPUs:
    python run_benchmarks.py benchmark_stability.py \
        --gpus 1,2,3 \
        --lib mixed=... --lib double=... -- --tests nve

    # Sweep multiple dt values (auto-equilibrates once, then benchmarks each dt):
    python run_benchmarks.py benchmark_tps.py \
        --lib mixed=... --lib double=... \
        --dt 0.005 0.01 0.02

Notes
-----
- GPU assignment is handled via CUDA_VISIBLE_DEVICES. The benchmark script
  always sees device 0. No positional gpu_id is needed.
- Each --lib flag is "label=pythonpath".  The label is used in output prefixes.
- If more jobs than free GPUs, jobs are queued and run as GPUs become free.
- Use {label} in script_args for per-build file paths.
- --dt sweeps reuse a single equilibrated state per lib to save time.
"""

import argparse
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_free_gpus(allowed_gpus=None):
    """Return list of GPU indices with no compute processes running."""
    # Get all GPU indices
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        capture_output=True, text=True
    )
    all_gpus = [int(x.strip()) for x in result.stdout.strip().split("\n") if x.strip()]

    # Get GPUs that have compute processes
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid",
         "--format=csv,noheader"],
        capture_output=True, text=True
    )
    busy_uuids = set()
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line:
            busy_uuids.add(line)

    # Map UUIDs to indices for busy GPUs
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    busy_indices = set()
    for line in result.stdout.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split(",")
        idx = int(parts[0].strip())
        uuid = parts[1].strip()
        if uuid in busy_uuids:
            busy_indices.add(idx)

    free = [g for g in all_gpus if g not in busy_indices]
    if allowed_gpus is not None:
        free = [g for g in free if g in allowed_gpus]

    return sorted(free)


def _stream_pipe(pipe, label, out_lines):
    """Read lines from *pipe*, print with [label] prefix, collect in *out_lines*."""
    for line in pipe:
        tagged = f"[{label:>8s}] {line}"
        sys.stdout.write(tagged)
        sys.stdout.flush()
        out_lines.append(line)


def run_one_benchmark(script_path, gpu_id, label, pythonpath, extra_args):
    """Run a single benchmark on the given GPU.  Returns (label, output, returncode).

    Output is streamed line-by-line to stdout with [label] prefix.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = pythonpath
    # Disable CUDA device auto-selection to avoid conflicts
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Substitute {label} in extra_args
    resolved = [a.replace("{label}", label) for a in extra_args]

    # With CUDA_VISIBLE_DEVICES set, the script sees device 0 (--gpu default)
    cmd = [sys.executable, script_path] + list(resolved)

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    out_lines = []
    for line in proc.stdout:
        tagged = f"[{label:>8s}] {line}"
        sys.stdout.write(tagged)
        sys.stdout.flush()
        out_lines.append(line)
    proc.wait()
    return label, "".join(out_lines), "", proc.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run HOOMD benchmarks in parallel across free GPUs.",
        epilog="Extra arguments after -- are passed to the benchmark script "
               "(minus gpu_id, which is auto-assigned).",
    )
    parser.add_argument("script", help="Benchmark script to run")
    parser.add_argument(
        "--lib", action="append", required=True, metavar="LABEL=PATH",
        help="label=pythonpath pair (repeatable). "
             "Example: --lib mixed=/path/to/site-packages",
    )
    parser.add_argument(
        "--gpus", default=None,
        help="Comma-separated list of allowed GPU indices (default: auto-detect free)",
    )
    parser.add_argument(
        "--log-dir", default="/tmp", metavar="DIR",
        help="Directory for per-job log files (default: /tmp). "
             "Each job writes to DIR/bench_LABEL.log via --log.",
    )
    parser.add_argument(
        "--dt", nargs="+", type=float,
        default=[0.005, 0.01, 0.03, 0.05, 0.1], metavar="DT",
        help="One or more dt values to sweep. Equilibrates once per lib, "
             "then benchmarks each dt from the saved state. "
             "(default: 0.005 0.01 0.03 0.05 0.1). "
             "Use --no-dt to disable the sweep.",
    )
    parser.add_argument(
        "--no-dt", action="store_true",
        help="Disable dt sweep; run a single benchmark per lib.",
    )

    # Split on -- to separate runner args from script args
    if "--" in sys.argv:
        split_idx = sys.argv.index("--")
        runner_argv = sys.argv[1:split_idx]
        script_extra = sys.argv[split_idx + 1:]
    else:
        runner_argv = sys.argv[1:]
        script_extra = []

    args = parser.parse_args(runner_argv)

    # Parse --lib flags
    jobs = []
    for lib_spec in args.lib:
        if "=" not in lib_spec:
            print(f"Error: --lib must be label=path, got: {lib_spec}",
                  file=sys.stderr)
            sys.exit(1)
        label, path = lib_spec.split("=", 1)
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            print(f"Warning: path does not exist: {path}", file=sys.stderr)
        jobs.append((label, path))

    # Parse --gpus
    allowed = None
    if args.gpus:
        allowed = [int(x) for x in args.gpus.split(",")]

    # Find free GPUs
    free_gpus = get_free_gpus(allowed)
    if not free_gpus:
        print("Error: no free GPUs available!", file=sys.stderr)
        sys.exit(1)

    n_jobs = len(jobs)
    n_gpus = len(free_gpus)
    print(f"Free GPUs: {free_gpus}")
    print(f"Jobs: {n_jobs} ({', '.join(l for l, _ in jobs)})")
    if n_jobs > n_gpus:
        print(f"Note: {n_jobs} jobs but only {n_gpus} free GPUs — "
              f"will queue {n_jobs - n_gpus} job(s)")
    print()

    # Copy script to temp location to avoid source-tree shadowing
    script_src = os.path.abspath(args.script)
    tmp_dir = tempfile.mkdtemp(prefix="hoomd_bench_")
    script_dst = os.path.join(tmp_dir, os.path.basename(script_src))
    shutil.copy2(script_src, script_dst)

    # Inject --log for each job so output is always tail -f friendly
    log_dir = os.path.abspath(args.log_dir)
    os.makedirs(log_dir, exist_ok=True)

    if args.dt and not args.no_dt:
        _run_dt_sweep(args, jobs, free_gpus, script_dst, log_dir, script_extra)
    else:
        _run_simple(args, jobs, free_gpus, script_dst, log_dir, script_extra)

    # Cleanup
    try:
        os.unlink(script_dst)
        os.rmdir(tmp_dir)
    except OSError:
        pass


def _run_simple(args, jobs, free_gpus, script_dst, log_dir, script_extra):
    """Original behaviour: one run per lib, all in parallel.

    Uses a GPU pool to guarantee no two jobs share a GPU simultaneously.
    """
    n_jobs = len(jobs)
    n_gpus = len(free_gpus)

    gpu_pool = queue.Queue()
    for g in free_gpus:
        gpu_pool.put(g)

    print(f"  GPU pool: {free_gpus} ({n_gpus} GPUs for {n_jobs} jobs)")
    print(flush=True)

    t0 = time.time()

    def _run_with_pool(label, pythonpath, extra):
        gpu = gpu_pool.get()
        try:
            print(f"  {label:>10s} → GPU {gpu}  (started)", flush=True)
            return run_one_benchmark(script_dst, gpu, label, pythonpath, extra)
        finally:
            gpu_pool.put(gpu)

    results = {}
    with ThreadPoolExecutor(max_workers=min(n_jobs, n_gpus)) as pool:
        futures = {}
        for label, pythonpath in jobs:
            log_path = os.path.join(log_dir, f"bench_{label}.log")
            job_extra = list(script_extra) + ["--log", log_path]
            fut = pool.submit(_run_with_pool, label, pythonpath, job_extra)
            futures[fut] = label

        for fut in as_completed(futures):
            label = futures[fut]
            label_out, stdout, stderr, rc = fut.result()
            results[label] = (stdout, stderr, rc)
            elapsed = time.time() - t0
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            print(f"\n[{elapsed:6.1f}s] {label} finished — {status}",
                  flush=True)

    wall_time = time.time() - t0
    print(f"\nAll jobs done in {wall_time:.1f}s")
    print("=" * 80)

    # Print collected outputs (for easy copy/paste)
    for label, _ in jobs:
        stdout, stderr, rc = results[label]
        print(f"\n{'=' * 80}")
        print(f" {label.upper()}")
        print(f"{'=' * 80}")
        print(stdout)


def _run_dt_sweep(args, jobs, free_gpus, script_dst, log_dir, script_extra):
    """Two-phase dt sweep: equilibrate once per lib, then benchmark each dt."""
    dt_values = args.dt
    n_gpus = len(free_gpus)

    # ── Phase 1: equilibrate once per lib (parallel across GPUs) ──────
    print("=" * 80)
    print(f"DT SWEEP — Phase 1: Equilibrating {len(jobs)} lib(s)")
    print("=" * 80)

    state_dir = os.path.join(log_dir, "dt_sweep_states")
    os.makedirs(state_dir, exist_ok=True)
    state_paths = {}  # label → gsd path

    gpu_pool = queue.Queue()
    for g in free_gpus:
        gpu_pool.put(g)

    print(f"  GPU pool: {free_gpus} ({n_gpus} GPUs for {len(jobs)} libs)")
    print(flush=True)

    def _run_with_pool(label, pythonpath, extra):
        gpu = gpu_pool.get()
        try:
            print(f"  {label:>20s} → GPU {gpu}  (started)", flush=True)
            return run_one_benchmark(script_dst, gpu, label, pythonpath, extra)
        finally:
            gpu_pool.put(gpu)

    t0 = time.time()
    eq_results = {}
    with ThreadPoolExecutor(max_workers=min(len(jobs), n_gpus)) as pool:
        futures = {}
        for i, (label, pythonpath) in enumerate(jobs):
            gsd_path = os.path.join(state_dir, f"eq_{label}.gsd")
            state_paths[label] = gsd_path
            log_path = os.path.join(log_dir, f"bench_{label}_equil.log")
            eq_extra = _strip_args(script_extra,
                                   ["--load-state", "--equilibrate-only",
                                    "--save-state", "--dt"])
            eq_extra += ["--equilibrate-only",
                         "--save-state", gsd_path,
                         "--log", log_path]
            fut = pool.submit(_run_with_pool, label, pythonpath, eq_extra)
            futures[fut] = label

        for fut in as_completed(futures):
            label = futures[fut]
            _, stdout, stderr, rc = fut.result()
            eq_results[label] = rc
            elapsed = time.time() - t0
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            print(f"\n[{elapsed:6.1f}s] {label} equilibration — {status}",
                  flush=True)

    # Check all equilibrations succeeded
    failed = [l for l, rc in eq_results.items() if rc != 0]
    if failed:
        print(f"\nERROR: equilibration failed for: {', '.join(failed)}",
              file=sys.stderr)
        print("Aborting dt sweep.", file=sys.stderr)
        return

    # ── Phase 2: benchmark each (lib × dt) combination ───────────────
    sweep_jobs = []  # (combined_label, lib_label, pythonpath, dt)
    for label, pythonpath in jobs:
        for dt in dt_values:
            dt_str = f"{dt:g}"
            combined = f"{label}_dt{dt_str}"
            sweep_jobs.append((combined, label, pythonpath, dt))

    print(f"\n{'=' * 80}")
    print(f"DT SWEEP — Phase 2: {len(sweep_jobs)} benchmark runs "
          f"({len(jobs)} libs × {len(dt_values)} dt values)")
    print(f"  dt values: {', '.join(f'{d:g}' for d in dt_values)}")
    print(f"  GPU pool: {free_gpus} ({n_gpus} GPUs)")
    print("=" * 80)
    print(flush=True)

    t1 = time.time()
    bench_results = {}
    with ThreadPoolExecutor(max_workers=min(len(sweep_jobs), n_gpus)) as pool:
        futures = {}
        for combined, lib_label, pythonpath, dt in sweep_jobs:
            log_path = os.path.join(log_dir, f"bench_{combined}.log")
            gsd_path = state_paths[lib_label]
            bench_extra = _strip_args(script_extra,
                                      ["--load-state", "--equilibrate-only",
                                       "--save-state", "--dt"])
            bench_extra += ["--load-state", gsd_path,
                            "--dt", str(dt),
                            "--log", log_path]
            fut = pool.submit(
                _run_with_pool, combined, pythonpath, bench_extra,
            )
            futures[fut] = combined

        for fut in as_completed(futures):
            combined = futures[fut]
            _, stdout, stderr, rc = fut.result()
            bench_results[combined] = (stdout, stderr, rc)
            elapsed = time.time() - t1
            status = "OK" if rc == 0 else f"FAILED (rc={rc})"
            print(f"\n[{elapsed:6.1f}s] {combined} — {status}", flush=True)

    wall_time = time.time() - t0
    print(f"\nAll jobs done in {wall_time:.1f}s")
    print("=" * 80)

    # Print collected outputs grouped by dt
    for dt in dt_values:
        dt_str = f"{dt:g}"
        print(f"\n{'=' * 80}")
        print(f" dt = {dt_str}")
        print(f"{'=' * 80}")
        for label, _ in jobs:
            combined = f"{label}_dt{dt_str}"
            if combined in bench_results:
                stdout, stderr, rc = bench_results[combined]
                print(f"\n--- {label} ---")
                print(stdout)


def _strip_args(args_list, flags_to_strip):
    """Remove specified flags and their values from an argument list.

    Handles both '--flag value' and '--flag=value' forms.
    Flags without values (like --equilibrate-only) are also removed.
    """
    result = []
    skip_next = False
    for i, arg in enumerate(args_list):
        if skip_next:
            skip_next = False
            continue
        stripped = False
        for flag in flags_to_strip:
            if arg == flag:
                # Check if it's a value-less flag (boolean) or has a next arg
                if flag == "--equilibrate-only":
                    stripped = True
                elif i + 1 < len(args_list) and not args_list[i + 1].startswith("--"):
                    skip_next = True
                    stripped = True
                else:
                    stripped = True
                break
            elif arg.startswith(flag + "="):
                stripped = True
                break
        if not stripped:
            result.append(arg)
    return result


if __name__ == "__main__":
    main()
