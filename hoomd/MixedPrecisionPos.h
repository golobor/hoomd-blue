// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#pragma once

#include "HOOMDMath.h"

/*! \file MixedPrecisionPos.h
    \brief Device helper functions for mixed-precision position handling.

    In mixed-precision mode (HOOMD_SHORTREAL_SIZE=32, HOOMD_LONGREAL_SIZE=64), positions
    are stored as Scalar4 (double4), but force kernels read them as ForceReal4 (float4) for
    speed. The integrator preserves full double-precision accuracy using a correction term:

        pos_full = pos_main + pos_correction

    where both pos_main and pos_correction are stored as Scalar4. pos_main holds the
    double-precision position, and pos_correction holds the rounding error from the
    last position update:

        correction = pos_new_double - Scalar(ForceReal(pos_new_double))

    This is conceptually similar to OpenMM's posq/posqCorrection split, adapted for
    HOOMD's storage model where positions are already Scalar4 (double4).

    Force kernels use loadPosForceReal() to get a ForceReal4 view (just a narrowing cast).
    Integrators use loadPosFull() / storePosFull() to read/write with correction.
*/

namespace hoomd
    {

#ifdef __HIPCC__

//! Load position as ForceReal4 for use in force evaluation kernels.
/*! In mixed precision with float4 mirror, this reads directly from the float4 array.
    In uniform precision, this is a no-op cast.
    The .w component (particle type as int-in-float) is preserved.
    \param d_pos_forcereal Device pointer to float4 position mirror (ForceReal4)
    \param idx Particle index
    \returns ForceReal4 position (lower precision for force evaluation)
*/
__device__ inline ForceReal4 loadPosForceReal(const ForceReal4* __restrict__ d_pos_forcereal,
                                              unsigned int idx)
    {
    return d_pos_forcereal[idx];
    }

#ifdef HOOMD_MIXED_PRECISION

//! Load full-precision position using main + correction arrays.
/*! Reconstructs double-precision position from the stored position and correction term.
    \param d_pos Device pointer to positions (Scalar4, double4)
    \param d_pos_correction Device pointer to position corrections (Scalar4, double4)
    \param idx Particle index
    \returns Scalar4 with full double-precision position
*/
__device__ inline Scalar4 loadPosFull(const Scalar4* __restrict__ d_pos,
                                      const Scalar4* __restrict__ d_pos_correction,
                                      unsigned int idx)
    {
    Scalar4 pos = d_pos[idx];
    Scalar4 corr = d_pos_correction[idx];
    pos.x += corr.x;
    pos.y += corr.y;
    pos.z += corr.z;
    // .w is the type tag, no correction needed
    return pos;
    }

//! Store full-precision position, splitting into main + correction + float4 mirror.
/*! Stores the position, computes the correction = full_pos - ForceReal(full_pos),
    and also writes the narrowed float4 to the position mirror array.
    \param d_pos Device pointer to positions (Scalar4)
    \param d_pos_correction Device pointer to position corrections (Scalar4)
    \param d_pos_forcereal Device pointer to float4 position mirror (ForceReal4)
    \param idx Particle index
    \param pos Full double-precision position to store (Scalar4, .w = type tag)
*/
__device__ inline void storePosFull(Scalar4* __restrict__ d_pos,
                                    Scalar4* __restrict__ d_pos_correction,
                                    ForceReal4* __restrict__ d_pos_forcereal,
                                    unsigned int idx,
                                    const Scalar4& pos)
    {
    // Store the main position
    d_pos[idx] = pos;

    // Compute and store correction: what gets lost when narrowing to ForceReal
    Scalar4 corr;
    corr.x = pos.x - static_cast<Scalar>(static_cast<ForceReal>(pos.x));
    corr.y = pos.y - static_cast<Scalar>(static_cast<ForceReal>(pos.y));
    corr.z = pos.z - static_cast<Scalar>(static_cast<ForceReal>(pos.z));
    corr.w = Scalar(0.0);
    d_pos_correction[idx] = corr;

    // Write float4 mirror for force kernels
    // .w stores particle type as int bits; re-pack via __int_as_forcereal
    ForceReal4 pos_fr;
    pos_fr.x = static_cast<ForceReal>(pos.x);
    pos_fr.y = static_cast<ForceReal>(pos.y);
    pos_fr.z = static_cast<ForceReal>(pos.z);
    pos_fr.w = __int_as_forcereal(__scalar_as_int(pos.w));
    d_pos_forcereal[idx] = pos_fr;
    }

#else // !HOOMD_MIXED_PRECISION

//! Load full-precision position (no correction needed in uniform precision).
__device__ inline Scalar4 loadPosFull(const Scalar4* __restrict__ d_pos,
                                      const Scalar4* __restrict__ /* d_pos_correction */,
                                      unsigned int idx)
    {
    return d_pos[idx];
    }

//! Store full-precision position (no correction in uniform precision, no separate mirror).
/*! In uniform precision ForceReal4 == Scalar4, so d_pos IS the forcereal array.
*/
__device__ inline void storePosFull(Scalar4* __restrict__ d_pos,
                                    Scalar4* __restrict__ /* d_pos_correction */,
                                    Scalar4* __restrict__ /* d_pos_forcereal */,
                                    unsigned int idx,
                                    const Scalar4& pos)
    {
    d_pos[idx] = pos;
    }

#endif // HOOMD_MIXED_PRECISION

#endif // __HIPCC__

    } // end namespace hoomd
