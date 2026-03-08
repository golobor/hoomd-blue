#!/bin/bash
# Source this script to switch between HOOMD-blue precision variants.
# Usage:  source use_hoomd.sh [mixed|double|single]
#
# Sets PYTHONPATH so that `import hoomd` loads the chosen variant.

HOOMD_BASE="/groups/goloborodko/user/anton.goloborodko/src/hoomd-blue/build"

_variant="${1:-mixed}"

case "$_variant" in
    mixed)
        _desc="Mixed precision (forces=float, integration=double)"
        _path="$HOOMD_BASE/install_mixed/lib/python3.12/site-packages"
        ;;
    double)
        _desc="Full double precision (original, 64/64)"
        _path="$HOOMD_BASE/install_double/lib/python3.12/site-packages"
        ;;
    single)
        _desc="Full single precision (32/32)"
        _path="$HOOMD_BASE/install_single/lib/python3.12/site-packages"
        ;;
    *)
        echo "Unknown variant: $_variant"
        echo "Usage: source use_hoomd.sh [mixed|double|single]"
        return 1 2>/dev/null || exit 1
        ;;
esac

# Remove any previous hoomd install paths from PYTHONPATH
PYTHONPATH=$(echo "$PYTHONPATH" | tr ':' '\n' | grep -v "$HOOMD_BASE/install_" | tr '\n' ':' | sed 's/:$//')

# Prepend the chosen variant
export PYTHONPATH="$_path${PYTHONPATH:+:$PYTHONPATH}"
export HOOMD_VARIANT="$_variant"

echo "HOOMD variant: $_variant ($_desc)"
echo "PYTHONPATH prefix: $_path"
