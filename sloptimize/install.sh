#!/usr/bin/env bash
# install.sh — Build and install the mixed-precision HOOMD-blue fork.
#
# Creates a conda environment with all build dependencies, compiles the
# mixed-precision variant (forces=float, integration=double), and installs
# it so that `import hoomd` works out of the box.
#
# Usage:
#   bash install.sh                     # defaults: env=hoomd-mixed, auto-detect GPU
#   bash install.sh --env hoomd-dev     # custom env name
#   bash install.sh --jobs 4            # limit parallel build jobs
#   bash install.sh --source /path/to   # use existing checkout instead of cloning
#
# Prerequisites:
#   - conda or miniforge (the script will find it automatically)
#   - NVIDIA GPU with CUDA drivers installed (nvidia-smi must work)
#   - ~5 GB disk space for build artifacts
#
# The script is idempotent: re-running it will rebuild and reinstall.

set -euo pipefail

# ─── Defaults ───────────────────────────────────────────────────────────
ENV_NAME="hoomd-mixed"
JOBS="$(nproc 2>/dev/null || echo 4)"
SOURCE_DIR=""
PYTHON_VERSION="3.12"
REPO_URL="https://github.com/golobor/hoomd-blue.git"
BRANCH="mixed-precision"

# ─── Parse arguments ───────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)       ENV_NAME="$2";       shift 2 ;;
        --jobs)      JOBS="$2";           shift 2 ;;
        --source)    SOURCE_DIR="$2";     shift 2 ;;
        --python)    PYTHON_VERSION="$2"; shift 2 ;;
        --help|-h)
            head -18 "$0" | tail -16
            exit 0 ;;
        *)
            echo "Unknown option: $1 (try --help)"
            exit 1 ;;
    esac
done

# ─── Helpers ────────────────────────────────────────────────────────────
info()  { echo -e "\033[1;34m==>\033[0m $*"; }
ok()    { echo -e "\033[1;32m ✓\033[0m  $*"; }
err()   { echo -e "\033[1;31m ✗\033[0m  $*" >&2; }
die()   { err "$@"; exit 1; }

# ─── 1. Find conda ─────────────────────────────────────────────────────
info "Looking for conda..."
if command -v conda &>/dev/null; then
    CONDA_EXE="$(command -v conda)"
elif [[ -x "$HOME/miniforge3/bin/conda" ]]; then
    CONDA_EXE="$HOME/miniforge3/bin/conda"
    eval "$("$CONDA_EXE" shell.bash hook)"
elif [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    CONDA_EXE="$HOME/miniconda3/bin/conda"
    eval "$("$CONDA_EXE" shell.bash hook)"
else
    die "conda not found. Install miniforge: https://github.com/conda-forge/miniforge"
fi
ok "Found conda: $CONDA_EXE"

# ─── 2. Check for NVIDIA GPU ───────────────────────────────────────────
info "Checking for NVIDIA GPU..."
if ! command -v nvidia-smi &>/dev/null; then
    die "nvidia-smi not found. CUDA drivers must be installed."
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
ok "Found GPU: $GPU_NAME"

# Detect CUDA compute capability for cmake
CUDA_CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.')
# e.g. "89" for RTX 4090 (sm_89), "86" for RTX 3090 (sm_86)
ok "CUDA compute capability: ${CUDA_CC:0:1}.${CUDA_CC:1}"

# ─── 3. Create/update conda environment ────────────────────────────────
info "Setting up conda environment: $ENV_NAME"

# Check if env exists
if conda env list | grep -qw "$ENV_NAME"; then
    info "Environment '$ENV_NAME' already exists, updating..."
    conda install -n "$ENV_NAME" -y -q \
        -c conda-forge \
        cmake eigen ninja numpy pybind11 gsd rowan cereal \
        "python=$PYTHON_VERSION" \
        cuda-nvcc cuda-cudart-dev cuda-nvrtc-dev libcufft-dev \
        2>/dev/null || true
else
    info "Creating environment '$ENV_NAME'..."
    conda create -n "$ENV_NAME" -y -q \
        -c conda-forge \
        cmake eigen ninja numpy pybind11 gsd rowan cereal \
        "python=$PYTHON_VERSION" \
        cuda-nvcc cuda-cudart-dev cuda-nvrtc-dev libcufft-dev
fi
conda activate "$ENV_NAME"
ok "Conda environment active: $CONDA_DEFAULT_ENV"

# ─── 4. Get source code ────────────────────────────────────────────────
if [[ -n "$SOURCE_DIR" ]]; then
    info "Using existing source: $SOURCE_DIR"
    cd "$SOURCE_DIR"
else
    # Check if we're already inside the repo
    _git_root=$(git rev-parse --show-toplevel 2>/dev/null || true)
    if [[ -n "$_git_root" ]] && [[ -f "$_git_root/sloptimize/install.sh" ]]; then
        info "Using current repo: $_git_root"
        cd "$_git_root"
    else
        CLONE_DIR="${HOME}/hoomd-mixed-precision"
        if [[ -d "$CLONE_DIR/.git" ]]; then
            info "Updating existing clone: $CLONE_DIR"
            cd "$CLONE_DIR"
            git fetch origin
            git checkout "$BRANCH"
            git pull origin "$BRANCH"
        else
            info "Cloning $REPO_URL ($BRANCH branch)..."
            git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$CLONE_DIR"
            cd "$CLONE_DIR"
        fi
    fi
fi

info "Initializing submodules..."
git submodule update --init --recursive
ok "Source ready: $(pwd)"

# ─── 5. Build ──────────────────────────────────────────────────────────
BUILD_DIR="build_install"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

info "Configuring CMake (mixed precision: LONG=64, SHORT=32)..."
cmake .. \
    -DHOOMD_LONGREAL_SIZE=64 \
    -DHOOMD_SHORTREAL_SIZE=32 \
    -DENABLE_GPU=ON \
    -DCUDA_ARCH_LIST="$CUDA_CC" \
    -DBUILD_TESTING=OFF \
    -DBUILD_MPCD=OFF \
    -DCMAKE_INSTALL_PREFIX="$CONDA_PREFIX" \
    -DCMAKE_BUILD_TYPE=Release \
    -GNinja \
    2>&1 | tail -5
ok "CMake configured"

info "Building with $JOBS parallel jobs (this may take 10-20 minutes)..."
ninja -j"$JOBS"
ok "Build complete"

# ─── 6. Install ────────────────────────────────────────────────────────
info "Installing into conda env ($CONDA_PREFIX)..."
ninja install
ok "Installed"

# ─── 7. Verify ─────────────────────────────────────────────────────────
info "Verifying installation..."
cd /tmp

PRECISION=$(python -c "import hoomd; print(hoomd.version.floating_point_precision)" 2>&1)
if [[ "$PRECISION" == *"(64, 32)"* ]]; then
    ok "Precision check passed: $PRECISION"
else
    err "Unexpected precision: $PRECISION"
    err "Expected (64, 32) for mixed precision"
    exit 1
fi

VERSION=$(python -c "import hoomd; print(hoomd.version.version)" 2>&1)
ok "HOOMD version: $VERSION"

# ─── Done ───────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Mixed-precision HOOMD-blue installed successfully!"
echo ""
echo " Activate:  conda activate $ENV_NAME"
echo " Verify:    python -c \"import hoomd; print(hoomd.version.floating_point_precision)\""
echo "            # should print: (64, 32)"
echo ""
echo " Usage:     conda activate $ENV_NAME"
echo "            python my_simulation.py"
echo ""
echo " GPU:       $GPU_NAME (sm_${CUDA_CC})"
echo " Env:       $ENV_NAME"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
