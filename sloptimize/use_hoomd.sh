#!/bin/bash
# Switch between HOOMD-blue precision variants (mixed / double / single).
#
# Usage:
#   source use_hoomd.sh [mixed|double|single]
#
# The script auto-detects the repository root from its own location and
# expects the three installs under  build/install_{mixed,double,single}/.
# It updates PYTHONPATH so that `import hoomd` loads the chosen variant
# and sets HOOMD_VARIANT for downstream scripts.
#
# Example:
#   source use_hoomd.sh mixed   # forces in float, integration in double
#   python my_simulation.py

# --- Resolve repository root from the script's location -----------------
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOOMD_BASE="$_script_dir/build"

if [[ ! -d "$HOOMD_BASE" ]]; then
    echo "Error: build/ directory not found at $HOOMD_BASE"
    return 1 2>/dev/null || exit 1
fi

# --- Auto-detect the Python site-packages path -------------------------
_site_packages_rel=$(find "$HOOMD_BASE/install_mixed" -type d -name site-packages 2>/dev/null | head -1)
if [[ -z "$_site_packages_rel" ]]; then
    echo "Error: no install_mixed/…/site-packages found under $HOOMD_BASE"
    return 1 2>/dev/null || exit 1
fi
# Extract the relative portion after install_mixed/ (e.g. lib/python3.12/site-packages)
_sp_suffix="${_site_packages_rel#"$HOOMD_BASE/install_mixed/"}"

# --- Parse variant ------------------------------------------------------
_variant="${1:-mixed}"

case "$_variant" in
    mixed)
        _desc="Mixed precision (forces=float, integration=double)"
        ;;
    double)
        _desc="Full double precision (64/64)"
        ;;
    single)
        _desc="Full single precision (32/32)"
        ;;
    *)
        echo "Unknown variant: $_variant"
        echo "Usage: source use_hoomd.sh [mixed|double|single]"
        return 1 2>/dev/null || exit 1
        ;;
esac

_path="$HOOMD_BASE/install_${_variant}/${_sp_suffix}"

if [[ ! -d "$_path" ]]; then
    echo "Error: install directory not found: $_path"
    return 1 2>/dev/null || exit 1
fi

# --- Update PYTHONPATH --------------------------------------------------
# Remove any previous hoomd install paths, then prepend the chosen one.
PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "$HOOMD_BASE/install_" | tr '\n' ':' | sed 's/:$//')
export PYTHONPATH="$_path${PYTHONPATH:+:$PYTHONPATH}"
export HOOMD_VARIANT="$_variant"

echo "HOOMD variant: $_variant — $_desc"
echo "PYTHONPATH prefix: $_path"
