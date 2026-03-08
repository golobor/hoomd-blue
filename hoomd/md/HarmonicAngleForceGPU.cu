// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "hip/hip_runtime.h"
// Copyright (c) 2009-2021 The Regents of the University of Michigan
// This file is part of the HOOMD-blue project, released under the BSD 3-Clause License.

#include "HarmonicAngleForceGPU.cuh"
#include "hoomd/TextureTools.h"

#include <assert.h>

// SMALL a relatively small number
#define SMALL ForceReal(0.001)

/*! \file HarmonicAngleForceGPU.cu
    \brief Defines GPU kernel code for calculating the harmonic angle forces. Used by
   HarmonicAngleForceComputeGPU.
*/

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel for calculating harmonic angle forces on the GPU
/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch Pitch of 2D virial array
    \param N number of particles
    \param d_pos device array of particle positions
    \param d_params Parameters for the angle force
    \param box Box dimensions for periodic boundary condition handling
    \param alist Angle data to use in calculating the forces
    \param pitch Pitch of 2D angles list
    \param n_angles_list List of numbers of angles stored on the GPU
*/
__global__ void gpu_compute_harmonic_angle_forces_kernel(Scalar4* d_force,
                                                         Scalar* d_virial,
                                                         const size_t virial_pitch,
                                                         const unsigned int N,
                                                         const Scalar4* d_pos,
                                                         const Scalar2* d_params,
                                                         BoxDim box,
                                                         const group_storage<3>* alist,
                                                         const unsigned int* apos_list,
                                                         const unsigned int pitch,
                                                         const unsigned int* n_angles_list)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_angles = n_angles_list[idx];

    // read in the position of our b-particle from the a-b-c triplet. (MEM TRANSFER: 16 bytes)
    Scalar4 idx_postype = d_pos[idx]; // we can be either a, b, or c in the a-b-c triplet
    ForceReal3 idx_pos = make_forcereal3(ForceReal(idx_postype.x), ForceReal(idx_postype.y), ForceReal(idx_postype.z));
    ForceReal3 a_pos, b_pos, c_pos; // allocate space for the a,b, and c atom in the a-b-c triplet

    // initialize the force to 0
    ForceReal4 force_idx = make_forcereal4(ForceReal(0.0), ForceReal(0.0), ForceReal(0.0), ForceReal(0.0));

    ForceReal fab[3], fcb[3];

    // initialize the virial to 0
    ForceReal virial[6];
    for (int i = 0; i < 6; i++)
        virial[i] = ForceReal(0.0);

    // loop over all angles
    for (int angle_idx = 0; angle_idx < n_angles; angle_idx++)
        {
        group_storage<3> cur_angle = alist[pitch * angle_idx + idx];

        int cur_angle_x_idx = cur_angle.idx[0];
        int cur_angle_y_idx = cur_angle.idx[1];
        int cur_angle_type = cur_angle.idx[2];

        int cur_angle_abc = apos_list[pitch * angle_idx + idx];

        // get the a-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 x_postype = d_pos[cur_angle_x_idx];
        ForceReal3 x_pos = make_forcereal3(ForceReal(x_postype.x), ForceReal(x_postype.y), ForceReal(x_postype.z));
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        Scalar4 y_postype = d_pos[cur_angle_y_idx];
        ForceReal3 y_pos = make_forcereal3(ForceReal(y_postype.x), ForceReal(y_postype.y), ForceReal(y_postype.z));

        if (cur_angle_abc == 0)
            {
            a_pos = idx_pos;
            b_pos = x_pos;
            c_pos = y_pos;
            }
        if (cur_angle_abc == 1)
            {
            b_pos = idx_pos;
            a_pos = x_pos;
            c_pos = y_pos;
            }
        if (cur_angle_abc == 2)
            {
            c_pos = idx_pos;
            a_pos = x_pos;
            b_pos = y_pos;
            }

        // calculate dr for a-b,c-b,and a-c
        ForceReal3 dab = a_pos - b_pos;
        ForceReal3 dcb = c_pos - b_pos;
        ForceReal3 dac = a_pos - c_pos;

        // apply periodic boundary conditions
        dab = box.minImageForceReal(dab);
        dcb = box.minImageForceReal(dcb);
        dac = box.minImageForceReal(dac);

        // get the angle parameters (MEM TRANSFER: 8 bytes)
        Scalar2 params = __ldg(d_params + cur_angle_type);
        ForceReal K = ForceReal(params.x);
        ForceReal t_0 = ForceReal(params.y);

        ForceReal rsqab = dot(dab, dab);
        ForceReal rab = sqrtf(rsqab);
        ForceReal rsqcb = dot(dcb, dcb);
        ForceReal rcb = sqrtf(rsqcb);

        ForceReal c_abbc = dot(dab, dcb);
        c_abbc /= rab * rcb;

        if (c_abbc > ForceReal(1.0))
            c_abbc = ForceReal(1.0);
        if (c_abbc < -ForceReal(1.0))
            c_abbc = -ForceReal(1.0);

        ForceReal s_abbc = sqrtf(ForceReal(1.0) - c_abbc * c_abbc);
        if (s_abbc < SMALL)
            s_abbc = SMALL;
        s_abbc = ForceReal(1.0) / s_abbc;

        // actually calculate the force
        ForceReal dth = fast::acos(c_abbc) - t_0;
        ForceReal tk = K * dth;

        ForceReal a = -ForceReal(1.0) * tk * s_abbc;
        ForceReal a11 = a * c_abbc / rsqab;
        ForceReal a12 = -a / (rab * rcb);
        ForceReal a22 = a * c_abbc / rsqcb;

        fab[0] = a11 * dab.x + a12 * dcb.x;
        fab[1] = a11 * dab.y + a12 * dcb.y;
        fab[2] = a11 * dab.z + a12 * dcb.z;

        fcb[0] = a22 * dcb.x + a12 * dab.x;
        fcb[1] = a22 * dcb.y + a12 * dab.y;
        fcb[2] = a22 * dcb.z + a12 * dab.z;

        // compute 1/3 of the energy, 1/3 for each atom in the angle
        ForceReal angle_eng = tk * dth * ForceReal(ForceReal(1.0) / ForceReal(6.0));

        // upper triangular version of virial tensor
        ForceReal angle_virial[6];
        angle_virial[0] = ForceReal(1. / 3.) * (dab.x * fab[0] + dcb.x * fcb[0]);
        angle_virial[1] = ForceReal(1. / 3.) * (dab.y * fab[0] + dcb.y * fcb[0]);
        angle_virial[2] = ForceReal(1. / 3.) * (dab.z * fab[0] + dcb.z * fcb[0]);
        angle_virial[3] = ForceReal(1. / 3.) * (dab.y * fab[1] + dcb.y * fcb[1]);
        angle_virial[4] = ForceReal(1. / 3.) * (dab.z * fab[1] + dcb.z * fcb[1]);
        angle_virial[5] = ForceReal(1. / 3.) * (dab.z * fab[2] + dcb.z * fcb[2]);

        if (cur_angle_abc == 0)
            {
            force_idx.x += fab[0];
            force_idx.y += fab[1];
            force_idx.z += fab[2];
            }
        if (cur_angle_abc == 1)
            {
            force_idx.x -= fab[0] + fcb[0];
            force_idx.y -= fab[1] + fcb[1];
            force_idx.z -= fab[2] + fcb[2];
            }
        if (cur_angle_abc == 2)
            {
            force_idx.x += fcb[0];
            force_idx.y += fcb[1];
            force_idx.z += fcb[2];
            }

        force_idx.w += angle_eng;

        for (int i = 0; i < 6; i++)
            virial[i] += angle_virial[i];
        }

    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    d_force[idx] = make_scalar4(Scalar(force_idx.x), Scalar(force_idx.y), Scalar(force_idx.z), Scalar(force_idx.w));
    for (int i = 0; i < 6; i++)
        d_virial[i * virial_pitch + idx] = Scalar(virial[i]);
    }

/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos device array of particle positions
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param atable List of angles stored on the GPU
    \param pitch Pitch of 2D angles list
    \param n_angles_list List of numbers of angles stored on the GPU
    \param d_params K and t_0 params packed as Scalar2 variables
    \param n_angle_types Number of angle types in d_params
    \param block_size Block size to use when performing calculations
    \param compute_capability Device compute capability (200, 300, 350, ...)

    \returns Any error code resulting from the kernel launch
    \note Always returns hipSuccess in release builds to avoid the hipDeviceSynchronize()

    \a d_params should include one Scalar2 element per angle type. The x component contains K the
   spring constant and the y component contains t_0 the equilibrium angle.
*/
hipError_t gpu_compute_harmonic_angle_forces(Scalar4* d_force,
                                             Scalar* d_virial,
                                             const size_t virial_pitch,
                                             const unsigned int N,
                                             const Scalar4* d_pos,
                                             const BoxDim& box,
                                             const group_storage<3>* atable,
                                             const unsigned int* apos_list,
                                             const unsigned int pitch,
                                             const unsigned int* n_angles_list,
                                             Scalar2* d_params,
                                             unsigned int n_angle_types,
                                             int block_size)
    {
    assert(d_params);

    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_compute_harmonic_angle_forces_kernel);
    max_block_size = attr.maxThreadsPerBlock;

    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid(N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_compute_harmonic_angle_forces_kernel),
                       dim3(grid),
                       dim3(threads),
                       0,
                       0,
                       d_force,
                       d_virial,
                       virial_pitch,
                       N,
                       d_pos,
                       d_params,
                       box,
                       atable,
                       apos_list,
                       pitch,
                       n_angles_list);

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
