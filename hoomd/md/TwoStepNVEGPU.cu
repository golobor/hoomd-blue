// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "TwoStepNVEGPU.cuh"
#include "hoomd/MixedPrecisionPos.h"
#include "hoomd/VectorMath.h"

#include <assert.h>

/*! \file TwoStepNVEGPU.cu
    \brief Defines GPU kernel code for NVE integration on the GPU. Used by TwoStepNVEGPU.
*/
namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Takes the first half-step forward in the velocity-verlet NVE integration on a group of particles
/*! \param d_pos array of particle positions
    \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_image array of particle images
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param box Box dimensions for periodic boundary condition handling
    \param deltaT timestep
    \param limit If \a limit is true, then the dynamics will be limited so that particles do not
   move a distance further than \a limit_val in one step. \param limit_val Length to limit particle
   distance movement to \param zero_force Set to true to always assign an acceleration of 0 to all
   particles in the group

    This kernel must be executed with a 1D grid of any block size such that the number of threads is
   greater than or equal to the number of members in the group. The kernel's implementation simply
   reads one particle in each thread and updates that particle.

    <b>Performance notes:</b>
    Particle properties are read via the texture cache to optimize the bandwidth obtained with
   sparse groups. The writes in sparse groups will not be coalesced. However, because ParticleGroup
   sorts the index list the writes will be as contiguous as possible leading to fewer memory
   transactions on compute 1.3 hardware and more cache hits on Fermi.
*/
__global__ void gpu_nve_step_one_kernel(Scalar4* d_pos,
                                        Scalar4* d_pos_correction,
                                        ForceReal4* d_pos_forcereal,
                                        Scalar4* d_vel,
                                        const Scalar3* d_accel,
                                        int3* d_image,
                                        unsigned int* d_group_members,
                                        const unsigned int nwork,
                                        BoxDim box,
                                        Scalar deltaT,
                                        bool limit,
                                        Scalar limit_val,
                                        bool zero_force)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int work_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (work_idx < nwork)
        {
        const unsigned int group_idx = work_idx;
        unsigned int idx = d_group_members[group_idx];

        // do velocity verlet update
        // r(t+deltaT) = r(t) + v(t)*deltaT + (1/2)a(t)*deltaT^2
        // v(t+deltaT/2) = v(t) + (1/2)a*deltaT

        // read the particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 postype = loadPosFull(d_pos, d_pos_correction, idx);
        Scalar3 pos = make_scalar3(postype.x, postype.y, postype.z);

        // read the particle's velocity and acceleration (MEM TRANSFER: 32 bytes)
        Scalar4 velmass = d_vel[idx];
        Scalar3 vel = make_scalar3(velmass.x, velmass.y, velmass.z);

        Scalar3 accel = make_scalar3(Scalar(0.0), Scalar(0.0), Scalar(0.0));
        if (!zero_force)
            accel = d_accel[idx];

        // update the position (FLOPS: 15)
        Scalar3 dx = vel * deltaT + (Scalar(1.0) / Scalar(2.0)) * accel * deltaT * deltaT;

        // limit the movement of the particles
        if (limit)
            {
            Scalar len = sqrtf(dot(dx, dx));
            if (len > limit_val)
                dx = dx / len * limit_val;
            }

        // FLOPS: 3
        pos += dx;

        // update the velocity (FLOPS: 9)
        vel += (Scalar(1.0) / Scalar(2.0)) * accel * deltaT;

        // read in the particle's image (MEM TRANSFER: 16 bytes)
        int3 image = d_image[idx];

        // fix the periodic boundary conditions (FLOPS: 15)
        box.wrap(pos, image);

        // write out the results (MEM_TRANSFER: 48 bytes)
        storePosFull(d_pos, d_pos_correction, d_pos_forcereal, idx,
                     make_scalar4(pos.x, pos.y, pos.z, postype.w));
        d_vel[idx] = make_scalar4(vel.x, vel.y, vel.z, velmass.w);
        d_image[idx] = image;
        }
    }

/*! \param d_pos array of particle positions
    \param d_vel array of particle velocities
    \param d_accel array of particle accelerations
    \param d_image array of particle images
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param box Box dimensions for periodic boundary condition handling
    \param deltaT timestep
    \param limit If \a limit is true, then the dynamics will be limited so that particles do not
   move a distance further than \a limit_val in one step. \param limit_val Length to limit particle
   distance movement to \param zero_force Set to true to always assign an acceleration of 0 to all
   particles in the group

    See gpu_nve_step_one_kernel() for full documentation, this function is just a driver.
*/
hipError_t gpu_nve_step_one(Scalar4* d_pos,
                            Scalar4* d_pos_correction,
                            ForceReal4* d_pos_forcereal,
                            Scalar4* d_vel,
                            const Scalar3* d_accel,
                            int3* d_image,
                            unsigned int* d_group_members,
                            const unsigned int group_size,
                            const BoxDim& box,
                            Scalar deltaT,
                            bool limit,
                            Scalar limit_val,
                            bool zero_force,
                            unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_nve_step_one_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    unsigned int nwork = group_size;

    // setup the grid to run the kernel
    dim3 grid((nwork / run_block_size) + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_nve_step_one_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_pos,
                       d_pos_correction,
                       d_pos_forcereal,
                       d_vel,
                       d_accel,
                       d_image,
                       d_group_members,
                       nwork,
                       box,
                       deltaT,
                       limit,
                       limit_val,
                       zero_force);
    return hipSuccess;
    }

//! NO_SQUISH angular part of the first half step
/*! \param d_orientation array of particle orientations
    \param d_angmom array of particle conjugate quaternions
    \param d_inertia array of moments of inertia
    \param d_net_torque array of net torques
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param deltaT timestep
*/
__global__ void gpu_nve_angular_step_one_kernel(Scalar4* d_orientation,
                                                Scalar4* d_angmom,
                                                const Scalar3* d_inertia,
                                                const ForceReal4* d_net_torque,
                                                const unsigned int* d_group_members,
                                                const unsigned int nwork,
                                                Scalar deltaT,
                                                Scalar scale)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int work_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (work_idx < nwork)
        {
        const unsigned int group_idx = work_idx;
        unsigned int idx = d_group_members[group_idx];

        // The no-squish (Miller) rotation is the compute-heavy part of the rigid-body
        // integrator (~10 cos/sin per particle). Like the force evaluators, it runs in
        // the reduced-precision ForceReal type: single on the mixed-precision build
        // (SHORTREAL=32) where fast:: maps to the __cosf/__sinf intrinsics, and
        // byte-identical double on the double/upstream builds (ForceReal aliases Scalar
        // and fast:: falls through to the same ::cos/::sin/::sqrt as slow::). The
        // orientation/angmom state stays Scalar4; quaternions are bounded and
        // renormalized each step, so single-precision compute is safe.
        ForceReal deltaT_r = ForceReal(deltaT);
        ForceReal scale_r = ForceReal(scale);

        // read the particle's orientation, conjugate quaternion, moment of inertia and net torque
        quat<ForceReal> q(d_orientation[idx]);
        quat<ForceReal> p(d_angmom[idx]);
        ForceReal4 t_raw = d_net_torque[idx];
        vec3<ForceReal> t(t_raw.x, t_raw.y, t_raw.z);
        vec3<ForceReal> I(d_inertia[idx]);

        // rotate torque into principal frame
        t = rotate(conj(q), t);

        // check for zero moment of inertia
        bool x_zero, y_zero, z_zero;
        x_zero = (I.x == 0);
        y_zero = (I.y == 0);
        z_zero = (I.z == 0);

        // ignore torque component along an axis for which the moment of inertia zero
        if (x_zero)
            t.x = ForceReal(0.0);
        if (y_zero)
            t.y = ForceReal(0.0);
        if (z_zero)
            t.z = ForceReal(0.0);

        // advance p(t)->p(t+deltaT/2), q(t)->q(t+deltaT)
        p += deltaT_r * q * t;

        p = p * scale_r;

        quat<ForceReal> p1, p2, p3; // permutated quaternions
        quat<ForceReal> q1, q2, q3;
        ForceReal phi1, cphi1, sphi1;
        ForceReal phi2, cphi2, sphi2;
        ForceReal phi3, cphi3, sphi3;

        if (!z_zero)
            {
            p3 = quat<ForceReal>(-p.v.z, vec3<ForceReal>(p.v.y, -p.v.x, p.s));
            q3 = quat<ForceReal>(-q.v.z, vec3<ForceReal>(q.v.y, -q.v.x, q.s));
            phi3 = ForceReal(1. / 4.) / I.z * dot(p, q3);
            cphi3 = fast::cos(ForceReal(1. / 2.) * deltaT_r * phi3);
            sphi3 = fast::sin(ForceReal(1. / 2.) * deltaT_r * phi3);

            p = cphi3 * p + sphi3 * p3;
            q = cphi3 * q + sphi3 * q3;
            }

        if (!y_zero)
            {
            p2 = quat<ForceReal>(-p.v.y, vec3<ForceReal>(-p.v.z, p.s, p.v.x));
            q2 = quat<ForceReal>(-q.v.y, vec3<ForceReal>(-q.v.z, q.s, q.v.x));
            phi2 = ForceReal(1. / 4.) / I.y * dot(p, q2);
            cphi2 = fast::cos(ForceReal(1. / 2.) * deltaT_r * phi2);
            sphi2 = fast::sin(ForceReal(1. / 2.) * deltaT_r * phi2);

            p = cphi2 * p + sphi2 * p2;
            q = cphi2 * q + sphi2 * q2;
            }

        if (!x_zero)
            {
            p1 = quat<ForceReal>(-p.v.x, vec3<ForceReal>(p.s, p.v.z, -p.v.y));
            q1 = quat<ForceReal>(-q.v.x, vec3<ForceReal>(q.s, q.v.z, -q.v.y));
            phi1 = ForceReal(1. / 4.) / I.x * dot(p, q1);
            cphi1 = fast::cos(deltaT_r * phi1);
            sphi1 = fast::sin(deltaT_r * phi1);

            p = cphi1 * p + sphi1 * p1;
            q = cphi1 * q + sphi1 * q1;
            }

        if (!y_zero)
            {
            p2 = quat<ForceReal>(-p.v.y, vec3<ForceReal>(-p.v.z, p.s, p.v.x));
            q2 = quat<ForceReal>(-q.v.y, vec3<ForceReal>(-q.v.z, q.s, q.v.x));
            phi2 = ForceReal(1. / 4.) / I.y * dot(p, q2);
            cphi2 = fast::cos(ForceReal(1. / 2.) * deltaT_r * phi2);
            sphi2 = fast::sin(ForceReal(1. / 2.) * deltaT_r * phi2);

            p = cphi2 * p + sphi2 * p2;
            q = cphi2 * q + sphi2 * q2;
            }

        if (!z_zero)
            {
            p3 = quat<ForceReal>(-p.v.z, vec3<ForceReal>(p.v.y, -p.v.x, p.s));
            q3 = quat<ForceReal>(-q.v.z, vec3<ForceReal>(q.v.y, -q.v.x, q.s));
            phi3 = ForceReal(1. / 4.) / I.z * dot(p, q3);
            cphi3 = fast::cos(ForceReal(1. / 2.) * deltaT_r * phi3);
            sphi3 = fast::sin(ForceReal(1. / 2.) * deltaT_r * phi3);

            p = cphi3 * p + sphi3 * p3;
            q = cphi3 * q + sphi3 * q3;
            }

        // renormalize (improves stability)
        q = q * (ForceReal(1.0) / fast::sqrt(norm2(q)));

        d_orientation[idx] = quat_to_scalar4(q);
        d_angmom[idx] = quat_to_scalar4(p);
        }
    }

/*! \param d_orientation array of particle orientations
    \param d_angmom array of particle conjugate quaternions
    \param d_inertia array of moments of inertia
    \param d_net_torque array of net torques
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param deltaT timestep
*/
hipError_t gpu_nve_angular_step_one(Scalar4* d_orientation,
                                    Scalar4* d_angmom,
                                    const Scalar3* d_inertia,
                                    const ForceReal4* d_net_torque,
                                    unsigned int* d_group_members,
                                    const unsigned int group_size,
                                    Scalar deltaT,
                                    Scalar scale,
                                    const unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_nve_angular_step_one_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    unsigned int nwork = group_size;

    // setup the grid to run the kernel
    dim3 grid((nwork / run_block_size) + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_nve_angular_step_one_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_orientation,
                       d_angmom,
                       d_inertia,
                       d_net_torque,
                       d_group_members,
                       nwork,
                       deltaT,
                       scale);

    return hipSuccess;
    }

//! NO_SQUISH angular part of the second half step
/*! \param d_orientation array of particle orientations
    \param d_angmom array of particle conjugate quaternions
    \param d_inertia array of moments of inertia
    \param d_net_torque array of net torques
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param deltaT timestep
*/
__global__ void gpu_nve_angular_step_two_kernel(const Scalar4* d_orientation,
                                                Scalar4* d_angmom,
                                                const Scalar3* d_inertia,
                                                const ForceReal4* d_net_torque,
                                                unsigned int* d_group_members,
                                                const unsigned int nwork,
                                                Scalar deltaT,
                                                Scalar scale)
    {
    // determine which particle this thread works on (MEM TRANSFER: 4 bytes)
    int work_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (work_idx < nwork)
        {
        const unsigned int group_idx = work_idx;
        unsigned int idx = d_group_members[group_idx];

        // read the particle's orientation, conjugate quaternion, moment of inertia and net torque
        quat<Scalar> q(d_orientation[idx]);
        quat<Scalar> p(d_angmom[idx]);
        ForceReal4 t_raw = d_net_torque[idx]; vec3<Scalar> t(Scalar(t_raw.x), Scalar(t_raw.y), Scalar(t_raw.z));
        vec3<Scalar> I(d_inertia[idx]);

        // rotate torque into principal frame
        t = rotate(conj(q), t);

        // check for zero moment of inertia
        bool x_zero, y_zero, z_zero;
        x_zero = (I.x == 0);
        y_zero = (I.y == 0);
        z_zero = (I.z == 0);

        // ignore torque component along an axis for which the moment of inertia zero
        if (x_zero)
            t.x = Scalar(0.0);
        if (y_zero)
            t.y = Scalar(0.0);
        if (z_zero)
            t.z = Scalar(0.0);

        // rescale
        p = p * scale;

        // advance p(t)->p(t+deltaT/2), q(t)->q(t+deltaT)
        p += deltaT * q * t;

        d_angmom[idx] = quat_to_scalar4(p);
        }
    }

/*! \param d_orientation array of particle orientations
    \param d_angmom array of particle conjugate quaternions
    \param d_inertia array of moments of inertia
    \param d_net_torque array of net torques
    \param d_group_members Device array listing the indices of the members of the group to integrate
    \param group_size Number of members in the group
    \param deltaT timestep
*/
hipError_t gpu_nve_angular_step_two(const Scalar4* d_orientation,
                                    Scalar4* d_angmom,
                                    const Scalar3* d_inertia,
                                    const ForceReal4* d_net_torque,
                                    unsigned int* d_group_members,
                                    const unsigned int group_size,
                                    Scalar deltaT,
                                    Scalar scale,
                                    const unsigned int block_size)
    {
    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_nve_angular_step_two_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    unsigned int nwork = group_size;

    // setup the grid to run the kernel
    dim3 grid((nwork / run_block_size) + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_nve_angular_step_two_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_orientation,
                       d_angmom,
                       d_inertia,
                       d_net_torque,
                       d_group_members,
                       nwork,
                       deltaT,
                       scale);

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
