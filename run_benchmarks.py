#!/usr/bin/env python3
"""Run HOOMD benchmarks in parallel across free GPUs.

Automatically detects which GPUs are idle (no compute processes),
assigns one benchmark job per free GPU, and streams output with labels.

Usage
-----
    python run_benchmarks.py <script> [--lib label=path ...] [-- script_args...]

The placeholder ``{label}`` in script_args is replaced with each job's label.
This enables per-build state files, e.g.:

    python run_benchmarks.py benchmark_chains.py \
        --lib mixed=... --lib double=... \
        -- --equilibrate-only --save-state /tmp/eq_{label}.gsd

Examples
--------
    # Phase 1: equilibrate (one per build, in parallel)
    python run_benchmarks.py benchmark_chains.py \
        --lib mixed=.../install_mixed/lib/python3.12/site-packages \
        --lib double=.../install_double/lib/python3.12/site-packages \
        -- 64000 200 --equilibrate-only --save-state /tmp/eq_{label}.gsd

    # Phase 2: benchmark (loads saved state, skips equilibration)
    python run_benchmarks.py benchmark_chains.py \
        --lib mixed=... --lib double=... \
        -- 64000 200 --load-state /tmp/eq_{label}.gsd

    # Restrict to specific GPUs:
    python run_benchmarks.py benchmark_dt_stability.py \
        --gpus 1,2,3 \
        --lib mixed=... --lib double=... -- 64000 200

Notes
-----
- The benchmark script's FIRST positional argument must be gpu_id.
  The runner replaces it with the assigned GPU index automatically.
- Each --lib flag is "label=pythonpath".  The label is used in output prefixes.
- If more jobs than free GPUs, jobs are queued and run as GPUs become free.
- Use {label} in script_args for per-build file paths.
"""

import argparse
import os
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

    # With CUDA_VISIBLE_DEVICES set, the script sees device 0
    cmd = [sys.executable, script_path, "0"] + list(resolved)

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

    # Assign GPUs round-robin and run in parallel
    gpu_assignment = {jobs[i][0]: free_gpus[i % n_gpus] for i in range(n_jobs)}
    for label, gpu in gpu_assignment.items():
        log_path = os.path.join(log_dir, f"bench_{label}.log")
        print(f"  {label:>10s} → GPU {gpu}  (log: {log_path})")
    print(flush=True)

    t0 = time.time()

    results = {}
    with ThreadPoolExecutor(max_workers=min(n_jobs, n_gpus)) as pool:
        futures = {}
        for i, (label, pythonpath) in enumerate(jobs):
            gpu = gpu_assignment[label]
            log_path = os.path.join(log_dir, f"bench_{label}.log")
            job_extra = list(script_extra) + ["--log", log_path]
            fut = pool.submit(
                run_one_benchmark, script_dst, gpu, label, pythonpath,
                job_extra,
            )
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

    # Cleanup
    try:
        os.unlink(script_dst)
        os.rmdir(tmp_dir)
    except OSError:
        pass


if __name__ == "__main__":
    main()
