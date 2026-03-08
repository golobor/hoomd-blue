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
/*! In mixed precision, this narrows double -> float. In uniform precision, this is a no-op cast.
    The .w component (particle type as int-in-float) is preserved.
    \param d_pos Device pointer to positions (Scalar4)
    \param idx Particle index
    \returns ForceReal4 position (lower precision for force evaluation)
*/
__device__ inline ForceReal4 loadPosForceReal(const Scalar4* __restrict__ d_pos,
                                              unsigned int idx)
    {
    Scalar4 pos = d_pos[idx];
    ForceReal4 result;
    result.x = static_cast<ForceReal>(pos.x);
    result.y = static_cast<ForceReal>(pos.y);
    result.z = static_cast<ForceReal>(pos.z);
    result.w = static_cast<ForceReal>(pos.w); // type tag
    return result;
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

//! Store full-precision position, splitting into main + correction.
/*! Stores the position and computes the correction = full_pos - ForceReal(full_pos).
    This ensures that when force kernels read the position as ForceReal, the integrator
    can reconstruct the full double-precision value.
    \param d_pos Device pointer to positions (Scalar4)
    \param d_pos_correction Device pointer to position corrections (Scalar4)
    \param idx Particle index
    \param pos Full double-precision position to store (Scalar4, .w = type tag)
*/
__device__ inline void storePosFull(Scalar4* __restrict__ d_pos,
                                    Scalar4* __restrict__ d_pos_correction,
                                    unsigned int idx,
                                    const Scalar4& pos)
    {
    // Store the main position
    d_pos[idx] = pos;

    // Compute correction: what gets lost when narrowing to ForceReal
    Scalar4 corr;
    corr.x = pos.x - static_cast<Scalar>(static_cast<ForceReal>(pos.x));
    corr.y = pos.y - static_cast<Scalar>(static_cast<ForceReal>(pos.y));
    corr.z = pos.z - static_cast<Scalar>(static_cast<ForceReal>(pos.z));
    corr.w = Scalar(0.0);
    d_pos_correction[idx] = corr;
    }

#else // !HOOMD_MIXED_PRECISION

//! Load full-precision position (no correction needed in uniform precision).
__device__ inline Scalar4 loadPosFull(const Scalar4* __restrict__ d_pos,
                                      const Scalar4* __restrict__ /* d_pos_correction */,
                                      unsigned int idx)
    {
    return d_pos[idx];
    }

//! Store full-precision position (no correction needed in uniform precision).
__device__ inline void storePosFull(Scalar4* __restrict__ d_pos,
                                    Scalar4* __restrict__ /* d_pos_correction */,
                                    unsigned int idx,
                                    const Scalar4& pos)
    {
    d_pos[idx] = pos;
    }

#endif // HOOMD_MIXED_PRECISION

#endif // __HIPCC__

    } // end namespace hoomd
