// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __BOND_EVALUATOR_FENE_H__
#define __BOND_EVALUATOR_FENE_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorBondFENE.h
    \brief Defines the bond evaluator class for FENE potentials
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
#ifdef __HIPCC__
#define DEVICE __device__
#else
#define DEVICE
#endif

namespace hoomd
    {
namespace md
    {
struct fene_params
    {
    ForceReal k;
    ForceReal r_0;
    ForceReal epsilon_x_4;
    ForceReal sigma_6;
    ForceReal delta;

#ifndef __HIPCC__
    fene_params()
        {
        k = 0;
        r_0 = 0;
        epsilon_x_4 = 0;
        sigma_6 = 0;
        }

    fene_params(pybind11::dict v)
        {
        k = v["k"].cast<ForceReal>();
        r_0 = v["r0"].cast<ForceReal>();
        delta = v["delta"].cast<ForceReal>();
        ForceReal epsilon = v["epsilon"].cast<ForceReal>();
        ForceReal sigma = v["sigma"].cast<ForceReal>();
        sigma_6 = sigma * sigma * sigma * sigma * sigma * sigma;
        epsilon_x_4 = ForceReal(4.0) * epsilon;
        }

    pybind11::dict asDict()
        {
        pybind11::dict v;
        v["k"] = k;
        v["r0"] = r_0;
        v["sigma"] = pow(sigma_6, 1. / 6.);
        v["epsilon"] = epsilon_x_4 / 4.0;
        v["delta"] = delta;
        return v;
        }
#endif
    } __attribute__((aligned(16)));

//! Class for evaluating the FENE bond potential
/*! The parameters are:
    - \a K (params.x) Stiffness parameter for the force computation
    - \a r_0 (params.y) maximum bond length for the force computation
    - \a lj1 (params.z) Value of lj1 = 4.0*epsilon*pow(sigma,12.0)
       of the WCA potential in the force calculation
    - \a lj2 (params.w) Value of lj2 = 4.0*epsilon*pow(sigma,6.0)
       of the WCA potential in the force calculation
*/
class EvaluatorBondFENE
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    typedef fene_params param_type;

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _params Per type pair parameters of this potential
    */
    DEVICE EvaluatorBondFENE(ForceReal _rsq, const param_type& _params)
        : rsq(_rsq), K(_params.k), r_0(_params.r_0),
          lj1(_params.epsilon_x_4 * _params.sigma_6 * _params.sigma_6),
          lj2(_params.epsilon_x_4 * _params.sigma_6), delta(_params.delta)
        {
        }

    //! FENE  doesn't use charge
    DEVICE static bool needsCharge()
        {
        return false;
        }

    //! Accept the optional charge values
    /*! \param qa Charge of particle a
        \param qb Charge of particle b
    */
    DEVICE void setCharge(ForceReal qa, ForceReal qb) { }

    //! Evaluate the force and energy
    /*! \param force_divr Output parameter to write the computed force divided by r.
        \param bond_eng Output parameter to write the computed bond energy

        \return True if they are evaluated or false if the bond
                energy is not defined
    */
    DEVICE bool evalForceAndEnergy(ForceReal& force_divr, ForceReal& bond_eng)
        {
        ForceReal rmdoverr = ForceReal(1.0);

        // Correct the rsq for particles that are not unit in size.
        ForceReal rtemp = sqrt(rsq) - delta;
        rmdoverr = rtemp / sqrt(rsq);
        rsq = rtemp * rtemp;

        // compute the force magnitude/r in forcemag_divr (FLOPS: 9)
        ForceReal r2inv = ForceReal(1.0) / rsq;
        ForceReal r6inv = r2inv * r2inv * r2inv;

        ForceReal WCAforcemag_divr = ForceReal(0.0);
        ForceReal pair_eng = ForceReal(0.0);

        ForceReal sigma6inv = lj2 / lj1;
        ForceReal epsilon = lj2 * lj2 / ForceReal(4.0) / lj1;

        if (lj1 != 0 && r6inv > sigma6inv / ForceReal(2.0)) // wcalimit 2^(1/6))^6 sigma^6
            {
            WCAforcemag_divr = r2inv * r6inv * (ForceReal(12.0) * lj1 * r6inv - ForceReal(6.0) * lj2);
            pair_eng = (r6inv * (lj1 * r6inv - lj2) + epsilon);
            }

        force_divr = WCAforcemag_divr * rmdoverr;
        bond_eng = pair_eng;

        // need to check that K is nonzero, to avoid division by zero
        if (K != ForceReal(0.0))
            {
            // Check if bond length restriction is violated
            if (rsq >= r_0 * r_0)
                return false;

            force_divr += -K / (ForceReal(1.0) - rsq / (r_0 * r_0)) * rmdoverr;
            bond_eng += -ForceReal(0.5) * K * (r_0 * r_0) * log(ForceReal(1.0) - rsq / (r_0 * r_0));
            }

        return true;
        }

#ifndef __HIPCC__
    //! Get the name of this potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return std::string("fene");
        }
#endif

    protected:
    ForceReal rsq;   //!< Stored rsq from the constructor
    ForceReal K;     //!< K parameter
    ForceReal r_0;   //!< r_0 parameter
    ForceReal lj1;   //!< lj1 parameter
    ForceReal lj2;   //!< lj2 parameter
    ForceReal delta; //!< Radial shift
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __BOND_EVALUATOR_FENE_H__
