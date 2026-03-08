// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __GENERAL_ENVELOPE_H__
#define __GENERAL_ENVELOPE_H__

#ifndef __HIPCC__
#include <string>
#endif
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include <string.h>

/** need to declare these class methods with __device__ qualifiers when building in nvcc
    DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
    compiler
*/
#ifdef __HIPCC__
#define DEVICE __device__
#define HOSTDEVICE __host__ __device__
#else
#define DEVICE
#define HOSTDEVICE
#endif

namespace hoomd
    {
namespace md
    {

/** PatchEnvelope is an angle-dependent multiplier on an isotropic pair force to make it
   directional.

    Defines the envelopes \f( f_i, f_j \f):

    \f{align*}
    f_i(\vec{dr}, \vec{n}_i, \alpha) = \Big(1 + e^{-\omega (\frac{-\vec{dr} \cdot
   \vec{n_i}}{|\vec{dr}|} - \cos{\alpha})}\Big)^{-1}\\ f_j(\vec{dr}, \vec{n}_j, \alpha) = \Big(1 +
   e^{-\omega (\frac{\vec{dr} \cdot \vec{n_j}}{|\vec{dr}|} - \cos{\alpha})}\Big)^{-1} \f}

    where \f$ \vec{n}_i, \vec{n}_j \f$ are the patch directions in the world frame,
    \f$ \alpha \f$ is the patch half-angle, and \f$ \omega \f$ is the patch steepness.
*/
class PatchEnvelope
    {
    public:
    struct param_type
        {
        param_type() : cosalpha(0), omega(0) { }
#ifndef __HIPCC__
        param_type(pybind11::dict params) //<! param dict can take any python type
            {
            cosalpha = fast::cos(params["alpha"].cast<Scalar>());
            omega = params["omega"].cast<Scalar>();
            }

        pybind11::dict toPython()
            {
            pybind11::dict v;
            v["alpha"] = fast::acos(cosalpha);
            v["omega"] = omega;
            return v;
            }
#endif
        Scalar cosalpha;
        Scalar omega;
        } __attribute__((aligned(16)));

    struct shape_type
        {
        HOSTDEVICE shape_type() { }

#ifndef __HIPCC__

        shape_type(pybind11::object patch_location)
            {
            pybind11::tuple p_py = patch_location;
            if (len(p_py) != 3)
                throw std::runtime_error("Each patch position must have 3 elements");
            vec3<Scalar> p = vec3<Scalar>(pybind11::cast<Scalar>(p_py[0]),
                                          pybind11::cast<Scalar>(p_py[1]),
                                          pybind11::cast<Scalar>(p_py[2]));

            // normalize
            p = p * fast::rsqrt(dot(p, p));
            m_norm_patch_local_dir = vec_to_scalar3(p);
            }

        pybind11::object toPython()
            {
            return pybind11::make_tuple(m_norm_patch_local_dir.x,
                                        m_norm_patch_local_dir.y,
                                        m_norm_patch_local_dir.z);
            }
#endif

        Scalar3 m_norm_patch_local_dir;
        };

    /**  Constructor

         \param _dr Displacement vector from particle j to particle i
         \param q_i Quaternion of i^{th} particle
         \param q_j Quaternion of j^{th} particle
         \param _rcutsq Squared distance at which the potential goes to 0
         \param _params Per type pair parameters of this potential
         \param shape_i The patch location on the i^{th} particle
         \param shape_j The patch location on the j^{th} particle
    */
    DEVICE PatchEnvelope(const ForceReal3& _dr,
                         const Scalar4& _q_i,
                         const Scalar4& _q_j,
                         const ForceReal _rcutsq,
                         const param_type& _params,
                         const shape_type& shape_i,
                         const shape_type& shape_j)
        : dr(_dr.x, _dr.y, _dr.z), params(_params),
          p_i(shape_i.m_norm_patch_local_dir),
          p_j(shape_j.m_norm_patch_local_dir)
        {
        // compute current particle direction vectors

        // rotate from particle to world frame — use LongReal for rotation accuracy
        vec3<LongReal> ex(1, 0, 0);
        vec3<LongReal> ey(0, 1, 0);
        vec3<LongReal> ez(0, 0, 1);

        // a1, a2, a3 are orientation vectors of particle a in world frame
        // b1, b2, b3 are orientation vectors of particle b in world frame
        // ni_world is patch direction of particle i in world frame

        auto q_i = quat<LongReal>(_q_i);
        auto q_j = quat<LongReal>(_q_j);

#ifndef __HIPCC__
        auto R_i = rotmat3<LongReal>(q_i);
        auto R_j = rotmat3<LongReal>(q_j);
        auto a1_lr = R_i * ex;
        auto a2_lr = R_i * ey;
        auto a3_lr = R_i * ez;
        auto ni_world_lr = R_i * (vec3<LongReal>)p_i;
        auto b1_lr = R_j * ex;
        auto b2_lr = R_j * ey;
        auto b3_lr = R_j * ez;
        auto nj_world_lr = R_j * (vec3<LongReal>)p_j;
#else
        auto a1_lr = rotate(q_i, ex);
        auto a2_lr = rotate(q_i, ey);
        auto a3_lr = rotate(q_i, ez);
        auto ni_world_lr = rotate(q_i, vec3<LongReal>(p_i.x, p_i.y, p_i.z));
        auto b1_lr = rotate(q_j, ex);
        auto b2_lr = rotate(q_j, ey);
        auto b3_lr = rotate(q_j, ez);
        auto nj_world_lr = rotate(q_j, vec3<LongReal>(p_j.x, p_j.y, p_j.z));
#endif
        // narrow rotation results to ForceReal for force computation
        a1 = vec3<ForceReal>(ForceReal(a1_lr.x), ForceReal(a1_lr.y), ForceReal(a1_lr.z));
        a2 = vec3<ForceReal>(ForceReal(a2_lr.x), ForceReal(a2_lr.y), ForceReal(a2_lr.z));
        a3 = vec3<ForceReal>(ForceReal(a3_lr.x), ForceReal(a3_lr.y), ForceReal(a3_lr.z));
        ni_world = vec3<ForceReal>(ForceReal(ni_world_lr.x), ForceReal(ni_world_lr.y), ForceReal(ni_world_lr.z));
        b1 = vec3<ForceReal>(ForceReal(b1_lr.x), ForceReal(b1_lr.y), ForceReal(b1_lr.z));
        b2 = vec3<ForceReal>(ForceReal(b2_lr.x), ForceReal(b2_lr.y), ForceReal(b2_lr.z));
        b3 = vec3<ForceReal>(ForceReal(b3_lr.x), ForceReal(b3_lr.y), ForceReal(b3_lr.z));
        nj_world = vec3<ForceReal>(ForceReal(nj_world_lr.x), ForceReal(nj_world_lr.y), ForceReal(nj_world_lr.z));

        // compute distance
        drsq = dot(dr, dr);
        magdr = fast::sqrt(drsq);

        rhat = dr / magdr;

        // cos(angle between dr and pointing vector) — use ForceReal
        ForceReal costhetai = -dot(rhat, ni_world); // negative because dr = dx = pi - pj
        ForceReal costhetaj = dot(rhat, nj_world);

        ForceReal omega_fr = ForceReal(params.omega);
        ForceReal cosalpha_fr = ForceReal(params.cosalpha);

        exp_neg_omega_times_cos_theta_i_minus_cos_alpha
            = fast::exp(-omega_fr * (costhetai - cosalpha_fr));
        exp_neg_omega_times_cos_theta_j_minus_cos_alpha
            = fast::exp(-omega_fr * (costhetaj - cosalpha_fr));
        }

    DEVICE static bool needsCharge()
        {
        return false;
        }

    DEVICE void setCharge(ForceReal qi, ForceReal qj)
        {
        m_charge_i = qi;
        m_charge_j = qj;
        }

    //! Evaluate the force and energy
    /*
      \Param force Output parameter to write the computed force.
      \param envelope Output parameter to write the amount of modulation of the isotropic part
      \param torque_div_energy_i The torque exterted on the i^th particle, divided by energy of
      interaction. \param torque_div_energy_j The torque exterted on the j^th particle, divided by
      energy of interaction. \note There is no need to check if rsq < rcutsq in this method. Cutoff
      tests are performed in PotentialPair from the PairModulator. \return Always true
    */
    DEVICE bool evaluate(ForceReal3& force,
                         ForceReal& envelope,
                         ForceReal3& torque_div_energy_i,
                         ForceReal3& torque_div_energy_j)
        {
        // common calculations — all in ForceReal
        ForceReal omega_fr = ForceReal(params.omega);
        ForceReal cosalpha_fr = ForceReal(params.cosalpha);

        ForceReal f_min, f_max, f_max_min_inv;

        f_min = ForceReal(1.0) / (ForceReal(1.0) + fast::exp(-omega_fr * (ForceReal(-1) - cosalpha_fr)));
        f_max = ForceReal(1.0) / (ForceReal(1.0) + fast::exp(-omega_fr * (ForceReal(1) - cosalpha_fr)));

        f_max_min_inv = ForceReal(1) / (f_max - f_min);

        ForceReal fi = ForceReal(1.0) / (ForceReal(1.0) + exp_neg_omega_times_cos_theta_i_minus_cos_alpha);
        ForceReal dfi_du = omega_fr * exp_neg_omega_times_cos_theta_i_minus_cos_alpha
                        * f_max_min_inv * fi * fi;
        // normalize the modulator function
        fi = (fi - f_min) * f_max_min_inv;

        ForceReal fj = ForceReal(1.0) / (ForceReal(1.0) + exp_neg_omega_times_cos_theta_j_minus_cos_alpha);
        ForceReal dfj_du = omega_fr * exp_neg_omega_times_cos_theta_j_minus_cos_alpha
                        * f_max_min_inv * fj * fj;
        fj = (fj - f_min) * f_max_min_inv;

        // the overall modulation
        envelope = fi * fj;

        vec3<ForceReal> dfi_dni = dfi_du * -rhat;

        // narrowing p_i components to ForceReal for torque calculation
        ForceReal p_ix = ForceReal(p_i.x), p_iy = ForceReal(p_i.y), p_iz = ForceReal(p_i.z);
        ForceReal p_jx = ForceReal(p_j.x), p_jy = ForceReal(p_j.y), p_jz = ForceReal(p_j.z);

        torque_div_energy_i = vec_to_forcereal3(p_ix * cross(a1, dfi_dni))
                              + vec_to_forcereal3(p_iy * cross(a2, dfi_dni))
                              + vec_to_forcereal3(p_iz * cross(a3, dfi_dni));

        torque_div_energy_i *= ForceReal(-1) * fj;

        vec3<ForceReal> dfj_dnj = dfj_du * rhat; // still positive

        torque_div_energy_j = vec_to_forcereal3(p_jx * cross(b1, dfj_dnj))
                              + vec_to_forcereal3(p_jy * cross(b2, dfj_dnj))
                              + vec_to_forcereal3(p_jz * cross(b3, dfj_dnj));

        torque_div_energy_j *= ForceReal(-1) * fi;

        // find df/dr = df/du * du/dr (using chain rule)
        // find du/dr using quotient rule, where u = "hi" / "lo" = dot(dr,n) / magdr
        ForceReal lo = magdr;
        vec3<ForceReal> dlo = rhat;

        ForceReal dfi_dui = dfi_du;

        ForceReal hi = -dot(dr, ni_world);
        vec3<ForceReal> dhi = -ni_world;
        /// quotient rule
        vec3<ForceReal> dui_dr = (lo * dhi - hi * dlo) / (lo * lo);

        ForceReal dfj_duj = dfj_du;
        hi = dot(dr, nj_world);
        dhi = nj_world;
        // lo and dlo are the same as above
        vec3<ForceReal> duj_dr = (lo * dhi - hi * dlo) / (lo * lo);

        force = -vec_to_forcereal3(dfj_duj * duj_dr * fi + dfi_dui * dui_dr * fj);

        return true;
        }

#ifndef _HIPCC_
    static std::string getName()
        {
        return std::string("patchenvelope");
        }
#endif

    private:
    vec3<ForceReal> dr;

    const param_type& params;
    vec3<ForceReal> ni_world, nj_world;
    vec3<Scalar> p_i, p_j;  // patch directions in body frame — keep Scalar for rotation input
    vec3<ForceReal> a1, a2, a3;
    vec3<ForceReal> b1, b2, b3;

    ForceReal m_charge_i, m_charge_j;

    ForceReal drsq;
    ForceReal magdr;
    vec3<ForceReal> rhat;

    ForceReal exp_neg_omega_times_cos_theta_i_minus_cos_alpha;
    ForceReal exp_neg_omega_times_cos_theta_j_minus_cos_alpha;
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __GENERAL_ENVELOPE_H__
