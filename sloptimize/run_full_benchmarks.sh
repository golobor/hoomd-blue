#!/usr/bin/env bash
# run_full_benchmarks.sh — Run the full benchmark suite across all 5 builds.
#
# Usage:
#   bash sloptimize/run_full_benchmarks.sh              # all builds, all workloads
#   bash sloptimize/run_full_benchmarks.sh --quick       # single dt (0.005 only)
#   bash sloptimize/run_full_benchmarks.sh --workload chains  # one workload only
#
# Results are saved to /tmp/bench_suite/<build_label>/
# Compare with: python sloptimize/benchmark_stability.py compare \
#     /tmp/bench_suite/double /tmp/bench_suite/mixed /tmp/bench_suite/single \
#     /tmp/bench_suite/upstream_double /tmp/bench_suite/upstream_single

set -euo pipefail

# ── Config ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

BASE="$REPO_DIR/build"
TPS_SCRIPT="$SCRIPT_DIR/benchmark_tps.py"
STAB_SCRIPT="$SCRIPT_DIR/benchmark_stability.py"
RUNNER_SCRIPT="$SCRIPT_DIR/run_benchmarks.py"

OUT_BASE="/tmp/bench_suite"
WORKLOAD="all"
DT_ARGS=""  # empty = use defaults from run_benchmarks.py (0.005 0.01 0.03 0.05 0.1)

# ── Parse args ────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --quick)
            DT_ARGS="--dt 0.005"
            shift ;;
        --workload)
            WORKLOAD="$2"
            shift 2 ;;
        --out-dir)
            OUT_BASE="$2"
            shift 2 ;;
        --help|-h)
            head -12 "$0" | tail -10
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

# ── Workload definitions ─────────────────────────────────────────────────
# Each workload is a set of CLI flags for benchmark_tps.py / benchmark_stability.py
#   chains: angle + dihedral (default)
#   nodih:  angle, no dihedral
#   patchy: angle, no dihedral, PatchyGaussian
declare -A WORKLOAD_FLAGS
WORKLOAD_FLAGS[chains]=""
WORKLOAD_FLAGS[nodih]="--no-dihedral"
WORKLOAD_FLAGS[patchy]="--no-dihedral --patchy 1.0,0.5,0.6,20.0,1.5,2"

if [[ "$WORKLOAD" == "all" ]]; then
    WORKLOADS=(chains nodih patchy)
else
    WORKLOADS=("$WORKLOAD")
fi

# ── Run each workload ─────────────────────────────────────────────────────
for wl in "${WORKLOADS[@]}"; do
    WL_FLAGS="${WORKLOAD_FLAGS[$wl]}"

    echo "============================================================"
    echo "WORKLOAD: $wl  (flags: ${WL_FLAGS:-<default>})"
    echo "============================================================"
    echo ""

    # ── Phase 1: Equilibrate reference build (double) ─────────────────
    echo "--- Phase 1: Equilibrate reference (double) for '$wl' ---"
    python3 "$RUNNER_SCRIPT" "$STAB_SCRIPT" \
        --lib "double=${LIB_PATHS[double]}" \
        --no-dt \
        -- $WL_FLAGS \
           --equilibrate-only \
           --save-state "${OUT_BASE}/double/state_${wl}.gsd"
    echo ""

    # Equilibrate non-double builds (each gets its own state)
    echo "--- Equilibrate other builds for '$wl' ---"
    NON_DOUBLE_FLAGS=""
    for label in "${LABELS[@]}"; do
        if [[ "$label" != "double" ]]; then
            NON_DOUBLE_FLAGS="$NON_DOUBLE_FLAGS --lib ${label}=${LIB_PATHS[$label]}"
        fi
    done

    python3 "$RUNNER_SCRIPT" "$STAB_SCRIPT" \
        $NON_DOUBLE_FLAGS \
        --no-dt \
        -- $WL_FLAGS \
           --equilibrate-only \
           --save-state "${OUT_BASE}/{label}/state_${wl}.gsd"
    echo ""

    # ── Phase 2: Force accuracy (all builds, shared double state) ─────
    echo "--- Phase 2: Force accuracy (shared double state) for '$wl' ---"
    python3 "$RUNNER_SCRIPT" "$STAB_SCRIPT" \
        $LIB_FLAGS \
        --no-dt \
        -- $WL_FLAGS \
           --tests accuracy \
           --out-dir "${OUT_BASE}/{label}" \
           --load-state "${OUT_BASE}/double/state_${wl}.gsd" \
           --tag "$wl"
    echo ""

    # ── Phase 3: NVE stability (per-build states) ─────────────────────
    echo "--- Phase 3: NVE stability for '$wl' ---"
    python3 "$RUNNER_SCRIPT" "$STAB_SCRIPT" \
        $LIB_FLAGS \
        --no-dt \
        -- $WL_FLAGS \
           --tests nve \
           --out-dir "${OUT_BASE}/{label}" \
           --load-state "${OUT_BASE}/{label}/state_${wl}.gsd" \
           --tag "$wl" \
           $DT_ARGS
    echo ""

    # ── Phase 4: TPS (per-build states, dt sweep) ─────────────────────
    echo "--- Phase 4: TPS for '$wl' ---"
    python3 "$RUNNER_SCRIPT" "$TPS_SCRIPT" \
        $LIB_FLAGS \
        $DT_ARGS \
        -- $WL_FLAGS \
           --load-state "${OUT_BASE}/{label}/state_${wl}.gsd"
    echo ""

done

# ── Phase 5: Compare ─────────────────────────────────────────────────────
echo "============================================================"
echo "CROSS-BUILD COMPARISON"
echo "============================================================"
echo ""

python3 "$STAB_SCRIPT" compare \
    "${OUT_BASE}/double" \
    "${OUT_BASE}/mixed" \
    "${OUT_BASE}/single" \
    "${OUT_BASE}/upstream_double" \
    "${OUT_BASE}/upstream_single"

echo ""
echo "Done! Results in: $OUT_BASE"
echo "Re-run comparison anytime with:"
echo "  python3 $STAB_SCRIPT compare ${OUT_BASE}/double ${OUT_BASE}/mixed ${OUT_BASE}/single ${OUT_BASE}/upstream_double ${OUT_BASE}/upstream_single"
