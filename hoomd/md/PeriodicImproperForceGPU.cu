// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#include "PeriodicImproperForceGPU.cuh"
#include "hoomd/TextureTools.h"

#include <assert.h>

// Tiny epsilon to prevent 1/raasq overflow in float for near-collinear
// geometries. At 1e-12 the inverse is 1e12 — well within float range.
#define SMALL ForceReal(1e-12)

namespace hoomd
    {
namespace md
    {
namespace kernel
    {
//! Kernel for calculating periodic improper forces on the GPU
/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos particle positions on the device
    \param d_params Parameters for the angle force
    \param box Box dimensions for periodic boundary condition handling
    \param tlist improper data to use in calculating the forces
    \param improper_ABCD List of relative atom positions in the impropers
    \param pitch Pitch of 2D improper list
    \param n_impropers_list List of numbers of impropers per atom
*/
__global__ void
gpu_compute_periodic_improper_forces_kernel(ForceReal4* d_force,
                                            ForceReal* d_virial,
                                            const size_t virial_pitch,
                                            const unsigned int N,
                                            const ForceReal4* d_pos,
                                            const periodic_improper_params* d_params,
                                            BoxDim box,
                                            const group_storage<4>* tlist,
                                            const unsigned int* improper_ABCD,
                                            const unsigned int pitch,
                                            const unsigned int* n_impropers_list)
    {
    // start by identifying which particle we are to handle
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= N)
        return;

    // load in the length of the list for this thread (MEM TRANSFER: 4 bytes)
    int n_impropers = n_impropers_list[idx];

    // read in the position of our b-particle from the a-b-c-d set. (MEM TRANSFER: 16 bytes)
    ForceReal4 idx_postype = d_pos[idx]; // we can be either a, b, or c in the a-b-c-d quartet
    ForceReal3 idx_pos = make_forcereal3(ForceReal(idx_postype.x), ForceReal(idx_postype.y), ForceReal(idx_postype.z));
    ForceReal3 pos_a, pos_b, pos_c,
        pos_d; // allocate space for the a,b, and c atoms in the a-b-c-d quartet

    // initialize the force to 0
    ForceReal4 force_idx = make_forcereal4(ForceReal(0.0), ForceReal(0.0), ForceReal(0.0), ForceReal(0.0));

    // initialize the virial to 0
    ForceReal virial_idx[6];
    for (unsigned int i = 0; i < 6; i++)
        virial_idx[i] = ForceReal(0.0);

    // loop over all impropers
    for (int improper_idx = 0; improper_idx < n_impropers; improper_idx++)
        {
        group_storage<4> cur_improper = tlist[pitch * improper_idx + idx];
        unsigned int cur_ABCD = improper_ABCD[pitch * improper_idx + idx];

        int cur_improper_x_idx = cur_improper.idx[0];
        int cur_improper_y_idx = cur_improper.idx[1];
        int cur_improper_z_idx = cur_improper.idx[2];
        int cur_improper_type = cur_improper.idx[3];
        int cur_improper_abcd = cur_ABCD;

        // get the a-particle's position (MEM TRANSFER: 16 bytes)
        ForceReal4 x_postype = d_pos[cur_improper_x_idx];
        ForceReal3 x_pos = make_forcereal3(ForceReal(x_postype.x), ForceReal(x_postype.y), ForceReal(x_postype.z));
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        ForceReal4 y_postype = d_pos[cur_improper_y_idx];
        ForceReal3 y_pos = make_forcereal3(ForceReal(y_postype.x), ForceReal(y_postype.y), ForceReal(y_postype.z));
        // get the c-particle's position (MEM TRANSFER: 16 bytes)
        ForceReal4 z_postype = d_pos[cur_improper_z_idx];
        ForceReal3 z_pos = make_forcereal3(ForceReal(z_postype.x), ForceReal(z_postype.y), ForceReal(z_postype.z));

        if (cur_improper_abcd == 0)
            {
            pos_a = idx_pos;
            pos_b = x_pos;
            pos_c = y_pos;
            pos_d = z_pos;
            }
        if (cur_improper_abcd == 1)
            {
            pos_b = idx_pos;
            pos_a = x_pos;
            pos_c = y_pos;
            pos_d = z_pos;
            }
        if (cur_improper_abcd == 2)
            {
            pos_c = idx_pos;
            pos_a = x_pos;
            pos_b = y_pos;
            pos_d = z_pos;
            }
        if (cur_improper_abcd == 3)
            {
            pos_d = idx_pos;
            pos_a = x_pos;
            pos_b = y_pos;
            pos_c = z_pos;
            }

        // calculate dr for a-b,c-b,and a-c
        ForceReal3 dab = pos_a - pos_b;
        ForceReal3 dcb = pos_c - pos_b;
        ForceReal3 ddc = pos_d - pos_c;

        dab = box.minImageForceReal(dab);
        dcb = box.minImageForceReal(dcb);
        ddc = box.minImageForceReal(ddc);

        ForceReal3 dcbm = -dcb;
        dcbm = box.minImageForceReal(dcbm);

        // get the improper parameters
        periodic_improper_params params = d_params[cur_improper_type];
        ForceReal K = ForceReal(params.k);
        ForceReal sign = ForceReal(params.d);
        int m = params.n;
        ForceReal chi_0 = ForceReal(params.chi_0);

        ForceReal aax = dab.y * dcbm.z - dab.z * dcbm.y;
        ForceReal aay = dab.z * dcbm.x - dab.x * dcbm.z;
        ForceReal aaz = dab.x * dcbm.y - dab.y * dcbm.x;

        ForceReal bbx = ddc.y * dcbm.z - ddc.z * dcbm.y;
        ForceReal bby = ddc.z * dcbm.x - ddc.x * dcbm.z;
        ForceReal bbz = ddc.x * dcbm.y - ddc.y * dcbm.x;

        ForceReal raasq = aax * aax + aay * aay + aaz * aaz;
        ForceReal rbbsq = bbx * bbx + bby * bby + bbz * bbz;
        ForceReal rgsq = dcbm.x * dcbm.x + dcbm.y * dcbm.y + dcbm.z * dcbm.z;
        ForceReal rg = fast::sqrt(rgsq);

        // Clamp squared cross-product magnitudes to prevent overflow
        if (raasq < SMALL)
            raasq = SMALL;
        if (rbbsq < SMALL)
            rbbsq = SMALL;

        ForceReal rginv, raa2inv, rbb2inv;
        rginv = raa2inv = rbb2inv = ForceReal(0.0);
        if (rg > ForceReal(0.0))
            rginv = ForceReal(1.0) / rg;
        raa2inv = ForceReal(1.0) / raasq;
        rbb2inv = ForceReal(1.0) / rbbsq;
        ForceReal rabinv = fast::sqrt(raa2inv * rbb2inv);

        ForceReal c_abcd = (aax * bbx + aay * bby + aaz * bbz) * rabinv;
        ForceReal s_abcd = rg * rabinv * (aax * ddc.x + aay * ddc.y + aaz * ddc.z);

        if (c_abcd > ForceReal(1.0))
            c_abcd = ForceReal(1.0);
        if (c_abcd < -ForceReal(1.0))
            c_abcd = -ForceReal(1.0);
        if (s_abcd > ForceReal(1.0))
            s_abcd = ForceReal(1.0);
        if (s_abcd < -ForceReal(1.0))
            s_abcd = -ForceReal(1.0);

        ForceReal p = ForceReal(1.0);
        ForceReal ddfab;
        ForceReal dfab = ForceReal(0.0);

        for (int jj = 0; jj < m; jj++)
            {
            ddfab = p * c_abcd - dfab * s_abcd;
            dfab = p * s_abcd + dfab * c_abcd;
            p = ddfab;
            }

        /////////////////////////
        // FROM LAMMPS: sin_shift is always 0... so dropping all sin_shift terms!!!!
        // Adding charmm improper functionality, sin_shift not always 0,
        // cos_shift not always 1
        /////////////////////////
        ForceReal sin_chi_0 = fast::sin(chi_0);
        ForceReal cos_chi_0 = fast::cos(chi_0);
        p = p * cos_chi_0 + dfab * sin_chi_0;
        p *= sign;
        dfab = dfab * cos_chi_0 - ddfab * sin_chi_0;
        dfab *= sign;
        dfab *= -ForceReal(m);
        p += ForceReal(1.0);

        if (m < 1)
            {
            p = ForceReal(1.0) + sign;
            dfab = ForceReal(0.0);
            }

        ForceReal fg = dab.x * dcbm.x + dab.y * dcbm.y + dab.z * dcbm.z;
        ForceReal hg = ddc.x * dcbm.x + ddc.y * dcbm.y + ddc.z * dcbm.z;

        ForceReal fga = fg * raa2inv * rginv;
        ForceReal hgb = hg * rbb2inv * rginv;
        ForceReal gaa = -raa2inv * rg;
        ForceReal gbb = rbb2inv * rg;

        ForceReal dtfx = gaa * aax;
        ForceReal dtfy = gaa * aay;
        ForceReal dtfz = gaa * aaz;
        ForceReal dtgx = fga * aax - hgb * bbx;
        ForceReal dtgy = fga * aay - hgb * bby;
        ForceReal dtgz = fga * aaz - hgb * bbz;
        ForceReal dthx = gbb * bbx;
        ForceReal dthy = gbb * bby;
        ForceReal dthz = gbb * bbz;

        ForceReal df = -K * dfab * ForceReal(0.500); // the 0.5 term is for 1/2K in the forces

        ForceReal sx2 = df * dtgx;
        ForceReal sy2 = df * dtgy;
        ForceReal sz2 = df * dtgz;

        ForceReal ffax = df * dtfx;
        ForceReal ffay = df * dtfy;
        ForceReal ffaz = df * dtfz;

        ForceReal ffbx = sx2 - ffax;
        ForceReal ffby = sy2 - ffay;
        ForceReal ffbz = sz2 - ffaz;

        ForceReal ffdx = df * dthx;
        ForceReal ffdy = df * dthy;
        ForceReal ffdz = df * dthz;

        ForceReal ffcx = -sx2 - ffdx;
        ForceReal ffcy = -sy2 - ffdy;
        ForceReal ffcz = -sz2 - ffdz;

        // Now, apply the force to each individual atom a,b,c,d
        // and accumulate the energy/virial
        // compute 1/4 of the energy, 1/4 for each atom in the improper
        // ForceReal improper_eng = p*K*ForceReal(1.0/4.0);
        ForceReal improper_eng = p * K * ForceReal(1.0 / 8.0); // the 1/8th term is (1/2)K * 1/4
        // compute 1/4 of the virial, 1/4 for each atom in the improper
        // upper triangular version of virial tensor
        ForceReal improper_virial[6];
        improper_virial[0]
            = ForceReal(1. / 4.) * (dab.x * ffax + dcb.x * ffcx + (ddc.x + dcb.x) * ffdx);
        improper_virial[1]
            = ForceReal(1. / 4.) * (dab.y * ffax + dcb.y * ffcx + (ddc.y + dcb.y) * ffdx);
        improper_virial[2]
            = ForceReal(1. / 4.) * (dab.z * ffax + dcb.z * ffcx + (ddc.z + dcb.z) * ffdx);
        improper_virial[3]
            = ForceReal(1. / 4.) * (dab.y * ffay + dcb.y * ffcy + (ddc.y + dcb.y) * ffdy);
        improper_virial[4]
            = ForceReal(1. / 4.) * (dab.z * ffay + dcb.z * ffcy + (ddc.z + dcb.z) * ffdy);
        improper_virial[5]
            = ForceReal(1. / 4.) * (dab.z * ffaz + dcb.z * ffcz + (ddc.z + dcb.z) * ffdz);

        if (cur_improper_abcd == 0)
            {
            force_idx.x += ffax;
            force_idx.y += ffay;
            force_idx.z += ffaz;
            }
        if (cur_improper_abcd == 1)
            {
            force_idx.x += ffbx;
            force_idx.y += ffby;
            force_idx.z += ffbz;
            }
        if (cur_improper_abcd == 2)
            {
            force_idx.x += ffcx;
            force_idx.y += ffcy;
            force_idx.z += ffcz;
            }
        if (cur_improper_abcd == 3)
            {
            force_idx.x += ffdx;
            force_idx.y += ffdy;
            force_idx.z += ffdz;
            }

        force_idx.w += improper_eng;
        for (int k = 0; k < 6; k++)
            virial_idx[k] += improper_virial[k];
        }

    // now that the force calculation is complete, write out the result (MEM TRANSFER: 20 bytes)
    d_force[idx] = force_idx;
    for (int k = 0; k < 6; k++)
        d_virial[k * virial_pitch + idx] = virial_idx[k];
    }

/*! \param d_force Device memory to write computed forces
    \param d_virial Device memory to write computed virials
    \param virial_pitch pitch of 2D virial array
    \param N number of particles
    \param d_pos particle positions on the GPU
    \param box Box dimensions (in GPU format) to use for periodic boundary conditions
    \param tlist improper data to use in calculating the forces
    \param improper_ABCD List of relative atom positions in the impropers
    \param pitch Pitch of 2D improper list
    \param n_impropers_list List of numbers of impropers per atom
    \param d_params K, sign,multiplicity params packed as padded Scalar4 variables
    \param n_improper_types Number of improper types in d_params
    \param block_size Block size to use when performing calculations
    \param compute_capability Compute capability of the device (200, 300, 350, ...)

    \returns Any error code resulting from the kernel launch
    \note Always returns hipSuccess in release builds to avoid the hipDeviceSynchronize()

    \a d_params should include one Scalar4 element per improper type. The x component contains K the
   spring constant and the y component contains sign, and the z component the multiplicity.
*/
hipError_t gpu_compute_periodic_improper_forces(ForceReal4* d_force,
                                                ForceReal* d_virial,
                                                const size_t virial_pitch,
                                                const unsigned int N,
                                                const ForceReal4* d_pos,
                                                const BoxDim& box,
                                                const group_storage<4>* tlist,
                                                const unsigned int* improper_ABCD,
                                                const unsigned int pitch,
                                                const unsigned int* n_impropers_list,
                                                periodic_improper_params* d_params,
                                                unsigned int n_improper_types,
                                                int block_size,
                                                int warp_size)
    {
    assert(d_params);

    unsigned int max_block_size;
    hipFuncAttributes attr;
    hipFuncGetAttributes(&attr, (const void*)gpu_compute_periodic_improper_forces_kernel);
    max_block_size = attr.maxThreadsPerBlock;
    if (max_block_size % warp_size)
        // handle non-sensical return values from hipFuncGetAttributes
        max_block_size = (max_block_size / warp_size - 1) * warp_size;

    unsigned int run_block_size = min(block_size, max_block_size);

    // setup the grid to run the kernel
    dim3 grid(N / run_block_size + 1, 1, 1);
    dim3 threads(run_block_size, 1, 1);

    // run the kernel
    hipLaunchKernelGGL((gpu_compute_periodic_improper_forces_kernel),
                       grid,
                       threads,
                       0,
                       0,
                       d_force,
                       d_virial,
                       virial_pitch,
                       N,
                       d_pos,
                       d_params,
                       box,
                       tlist,
                       improper_ABCD,
                       pitch,
                       n_impropers_list);

    return hipSuccess;
    }

    } // end namespace kernel
    } // end namespace md
    } // end namespace hoomd
