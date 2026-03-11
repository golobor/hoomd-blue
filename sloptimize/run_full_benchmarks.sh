#!/usr/bin/env bash
# run_full_benchmarks.sh — Run the full benchmark suite across all 5 builds.
#
# Usage:
#   bash sloptimize/run_full_benchmarks.sh              # all builds, all workloads
#   bash sloptimize/run_full_benchmarks.sh --quick       # single dt (0.005 only)
#   bash sloptimize/run_full_benchmarks.sh --workload chains  # one workload only
#
# Results are saved to /tmp/bench_suite/<build_label>/
# Compare with: python sloptimize/benchmark_suite.py compare \
#     /tmp/bench_suite/double /tmp/bench_suite/mixed /tmp/bench_suite/single \
#     /tmp/bench_suite/upstream_double /tmp/bench_suite/upstream_single

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE="$REPO_DIR/build"
SUITE_SCRIPT="$SCRIPT_DIR/benchmark_suite.py"
RUNNER_SCRIPT="$SCRIPT_DIR/run_benchmarks.py"

OUT_BASE="/tmp/bench_suite"
WORKLOAD="all"
EXTRA_ARGS=""

# ── Parse args ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            EXTRA_ARGS="--dt 0.005"
            shift ;;
        --workload)
            WORKLOAD="$2"
            shift 2 ;;
        --out-dir)
            OUT_BASE="$2"
            shift 2 ;;
        --help|-h)
            head -11 "$0" | tail -9
            exit 0 ;;
        *)
            echo "Unknown option: $1"
            exit 1 ;;
    esac
done

# ── Build paths ───────────────────────────────────────────────────────────
declare -a LABELS=(mixed double single upstream_double upstream_single)
declare -A LIB_PATHS
LIB_PATHS[mixed]="$BASE/install_mixed/lib/python3.12/site-packages"
LIB_PATHS[double]="$BASE/install_double/lib/python3.12/site-packages"
LIB_PATHS[single]="$BASE/install_single/lib/python3.12/site-packages"
LIB_PATHS[upstream_double]="$BASE/install_upstream_double/lib/python3.12/site-packages"
LIB_PATHS[upstream_single]="$BASE/install_upstream_single/lib/python3.12/site-packages"

# Verify all builds exist
echo "Checking builds..."
MISSING=0
for label in "${LABELS[@]}"; do
    path="${LIB_PATHS[$label]}"
    if [[ -d "$path/hoomd" ]]; then
        echo "  ✓ $label: $path"
    else
        echo "  ✗ $label: $path (MISSING)"
        MISSING=1
    fi
done
if [[ $MISSING -eq 1 ]]; then
    echo ""
    echo "Some builds are missing. See sloptimize/BENCHMARKING.md for build instructions."
    exit 1
fi
echo ""

# ── Build --lib flags ─────────────────────────────────────────────────────
LIB_FLAGS=""
for label in "${LABELS[@]}"; do
    LIB_FLAGS="$LIB_FLAGS --lib ${label}=${LIB_PATHS[$label]}"
done

# ── Phase 1: Equilibrate reference build (double) ────────────────────────
echo "============================================================"
echo "PHASE 1: Equilibrating reference build (double)"
echo "============================================================"
echo ""

python3 "$RUNNER_SCRIPT" "$SUITE_SCRIPT" \
    --lib "double=${LIB_PATHS[double]}" \
    --no-dt \
    -- --workload "$WORKLOAD" \
       --out-dir "${OUT_BASE}/double" \
       --equilibrate-only

echo ""

# ── Phase 2: Force accuracy with shared reference state ──────────────────
echo "============================================================"
echo "PHASE 2a: Force accuracy (all builds, shared double state)"
echo "============================================================"
echo ""

python3 "$RUNNER_SCRIPT" "$SUITE_SCRIPT" \
    $LIB_FLAGS \
    --no-dt \
    -- --workload "$WORKLOAD" \
       --out-dir "${OUT_BASE}/{label}" \
       --load-dir "${OUT_BASE}/double" \
       --tests accuracy

echo ""

# ── Phase 3: NVE + Langevin (per-build equilibrated states) ──────────────
echo "============================================================"
echo "PHASE 2b: NVE + Langevin (per-build equilibrated states)"
echo "============================================================"
echo ""

# Equilibrate non-double builds
NON_DOUBLE_FLAGS=""
for label in "${LABELS[@]}"; do
    if [[ "$label" != "double" ]]; then
        NON_DOUBLE_FLAGS="$NON_DOUBLE_FLAGS --lib ${label}=${LIB_PATHS[$label]}"
    fi
done

python3 "$RUNNER_SCRIPT" "$SUITE_SCRIPT" \
    $NON_DOUBLE_FLAGS \
    --no-dt \
    -- --workload "$WORKLOAD" \
       --out-dir "${OUT_BASE}/{label}" \
       --equilibrate-only

echo ""

# Run NVE + Langevin for all builds
python3 "$RUNNER_SCRIPT" "$SUITE_SCRIPT" \
    $LIB_FLAGS \
    --no-dt \
    -- --workload "$WORKLOAD" \
       --out-dir "${OUT_BASE}/{label}" \
       --load-dir "${OUT_BASE}/{label}" \
       --tests nve \
       $EXTRA_ARGS

echo ""

python3 "$RUNNER_SCRIPT" "$SUITE_SCRIPT" \
    $LIB_FLAGS \
    --no-dt \
    -- --workload "$WORKLOAD" \
       --out-dir "${OUT_BASE}/{label}" \
       --load-dir "${OUT_BASE}/{label}" \
       --tests langevin \
       $EXTRA_ARGS

echo ""

# ── Phase 3: Compare ─────────────────────────────────────────────────────
echo "============================================================"
echo "PHASE 3: Cross-build comparison"
echo "============================================================"
echo ""

python3 "$SUITE_SCRIPT" compare \
    "${OUT_BASE}/double" \
    "${OUT_BASE}/mixed" \
    "${OUT_BASE}/single" \
    "${OUT_BASE}/upstream_double" \
    "${OUT_BASE}/upstream_single"

echo ""
echo "Done! Results in: $OUT_BASE"
echo "Re-run comparison anytime with:"
echo "  python3 $SUITE_SCRIPT compare ${OUT_BASE}/double ${OUT_BASE}/mixed ${OUT_BASE}/single ${OUT_BASE}/upstream_double ${OUT_BASE}/upstream_single"
