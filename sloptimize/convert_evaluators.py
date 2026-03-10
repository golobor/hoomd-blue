#!/usr/bin/env python3
"""Convert pair evaluator files from Scalar to ForceReal for mixed-precision support.

This script converts the device-side computation types in HOOMD pair evaluator
headers from Scalar (double) to ForceReal (float in mixed precision).

The conversion is targeted:
- param_type data members: Scalar -> ForceReal
- DEVICE constructor: Scalar _rsq, _rcutsq -> ForceReal
- evalForceAndEnergy: Scalar& -> ForceReal&
- setCharge: Scalar -> ForceReal
- Protected members: Scalar -> ForceReal
- Internal DEVICE method vars/casts: Scalar -> ForceReal
- __scalar_as_int -> __forcereal_as_int

NOT changed:
- Scalar4, Scalar3, make_scalar4, make_scalar3 (position types)
- pybind11::array_t<Scalar> (Python interface types)
- Existing ForceReal usages  
"""

import re
import sys
import os

def convert_evaluator(filepath):
    """Convert a single evaluator file from Scalar to ForceReal."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    original = content
    
    # Strategy: Do targeted replacements that avoid touching Scalar4, Scalar3,
    # make_scalar3, make_scalar4, __scalar_as_int patterns, and pybind types.
    
    # 1. Replace standalone 'Scalar' (not part of Scalar3/Scalar4/make_scalar/etc.)
    #    This regex matches 'Scalar' NOT followed by 3,4,_as,2 (for Scalar2) 
    #    and NOT preceded by 'make_' or 'Force' or other prefixes
    
    # First, protect patterns we DON'T want to change by replacing them with placeholders
    protections = [
        ('Scalar4', '__PLACEHOLDER_SCALAR4__'),
        ('Scalar3', '__PLACEHOLDER_SCALAR3__'),
        ('Scalar2', '__PLACEHOLDER_SCALAR2__'),
        ('make_scalar4', '__PLACEHOLDER_MAKE_SCALAR4__'),
        ('make_scalar3', '__PLACEHOLDER_MAKE_SCALAR3__'),
        ('make_scalar2', '__PLACEHOLDER_MAKE_SCALAR2__'),
        ('__scalar_as_int', '__PLACEHOLDER_SCALAR_AS_INT__'),
        ('__int_as_scalar', '__PLACEHOLDER_INT_AS_SCALAR__'),
        ('ForceReal', '__PLACEHOLDER_FORCEREAL__'),
        ('ShortReal', '__PLACEHOLDER_SHORTREAL__'),
        ('LongReal', '__PLACEHOLDER_LONGREAL__'),
        # pybind types
        ('pybind11::array_t<Scalar>', '__PLACEHOLDER_PYBIND_ARRAY_SCALAR__'),
    ]
    
    for orig, placeholder in protections:
        content = content.replace(orig, placeholder)
    
    # Now replace standalone Scalar with ForceReal
    # This handles: Scalar, Scalar&, Scalar*, const Scalar, Scalar(, etc.
    content = content.replace('Scalar', 'ForceReal')
    
    # Restore protected patterns
    for orig, placeholder in protections:
        content = content.replace(placeholder, orig)
    
    # Fix __scalar_as_int -> __forcereal_as_int (was protected, undo and redo)
    content = content.replace('__scalar_as_int', '__forcereal_as_int')
    content = content.replace('__int_as_scalar', '__int_as_forcereal')
    
    # Fix: in asDict() methods, we typically convert back to double for Python
    # These are inside #ifndef __HIPCC__ so they're host-only
    # ForceReal members assigned from Python dicts need explicit cast  
    # The pattern `v["key"].cast<ForceReal>()` should stay as `v["key"].cast<Scalar>()`
    # since Python provides doubles. But actually, cast<ForceReal>() (=cast<float>()) 
    # also works - pybind11 handles the conversion. Keep it.
    
    # Fix: evalPressureLRCIntegral and evalEnergyLRCIntegral - these are DEVICE methods
    # but called on host for LRC corrections. ForceReal is fine since the types match.
    
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    md_dir = os.path.join(base_dir, 'hoomd', 'md')
    
    # List of evaluator files to convert (excluding LJ which is already done)
    evaluators = [
        'EvaluatorPairBuckingham.h',
        'EvaluatorPairLJ1208.h',
        'EvaluatorPairLJ0804.h',
        'EvaluatorPairGauss.h',
        'EvaluatorPairExpandedLJ.h',
        'EvaluatorPairExpandedMie.h',
        'EvaluatorPairYukawa.h',
        'EvaluatorPairEwald.h',
        'EvaluatorPairMorse.h',
        'EvaluatorPairDPDThermoDPD.h',
        'EvaluatorPairMoliere.h',
        'EvaluatorPairZBL.h',
        'EvaluatorPairMie.h',
        'EvaluatorPairReactionField.h',
        'EvaluatorPairDLVO.h',
        'EvaluatorPairFourier.h',
        'EvaluatorPairOPP.h',
        'EvaluatorPairTWF.h',
        'EvaluatorPairLJGauss.h',
        'EvaluatorPairForceShiftedLJ.h',
        'EvaluatorPairTable.h',
        'EvaluatorPairExpandedGaussian.h',
        'EvaluatorPairWangFrenkel.h',
        'EvaluatorPairZetterling.h',
    ]
    
    converted = []
    skipped = []
    
    for evaluator in evaluators:
        filepath = os.path.join(md_dir, evaluator)
        if not os.path.exists(filepath):
            print(f"WARNING: {evaluator} not found, skipping")
            skipped.append(evaluator)
            continue
        
        if convert_evaluator(filepath):
            converted.append(evaluator)
            print(f"Converted: {evaluator}")
        else:
            print(f"No changes needed: {evaluator}")
            skipped.append(evaluator)
    
    print(f"\nConverted {len(converted)} evaluators, skipped {len(skipped)}")
    
    # Verify a converted file to check sanity
    if converted:
        test_file = os.path.join(md_dir, converted[0])
        with open(test_file, 'r') as f:
            content = f.read()
        
        # Check that we didn't break Scalar4/Scalar3 references
        if 'ForceReal4' in content and 'Scalar4' not in content:
            print("WARNING: Scalar4 was incorrectly converted to ForceReal4!")
        if 'ForceReal3' in content and 'make_forcescalar3' in content:
            print("WARNING: make_scalar3 was incorrectly converted!")
        
        # Check that ForceReal appears in key places
        if 'ForceReal& force_divr' in content:
            print(f"OK: {converted[0]} has ForceReal& in evalForceAndEnergy")
        if 'ForceReal _rsq' in content or 'ForceReal _rcutsq' in content:
            print(f"OK: {converted[0]} has ForceReal in constructor")


if __name__ == '__main__':
    main()
