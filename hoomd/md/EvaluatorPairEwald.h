// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_EWALD_H__
#define __PAIR_EVALUATOR_EWALD_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairEwald.h
    \brief Defines the pair evaluator class for Ewald potentials
*/

// need to declare these class methods with __device__ qualifiers when building in nvcc
// DEVICE is __host__ __device__ when included in nvcc and blank when included into the host
// compiler
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
//! Class for evaluating the Ewald pair potential
/*! <b>General Overview</b>

    See EvaluatorPairLJ

    <b>Ewald specifics</b>

    EvaluatorPairEwald evaluates the function:

    \f[
    V_{\mathrm{ewald}}(r)  = q_i q_j \left[\mathrm{erfc}\left(\kappa r +
   \frac{\alpha}{2\kappa}\right) \exp(\alpha r)+ \mathrm{erfc}\left(\kappa r - \frac{\alpha}{2
   \kappa}\right) \exp(-\alpha r)\right] \f]
*/
class EvaluatorPairEwald
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
        ForceReal kappa;
        ForceReal alpha;

        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifdef ENABLE_HIP
        //! Set CUDA memory hints
        void set_memory_hints() const { }
#endif

#ifndef __HIPCC__
        param_type() : kappa(0), alpha(0) { }

        param_type(pybind11::dict v, bool managed = false)
            {
            kappa = v["kappa"].cast<ForceReal>();
            alpha = v["alpha"].cast<ForceReal>();
            }

        pybind11::dict asDict()
            {
            pybind11::dict v;
            v["kappa"] = kappa;
            v["alpha"] = alpha;
            return v;
            }
#endif
        }
#if HOOMD_LONGREAL_SIZE == 32
        __attribute__((aligned(8)));
#else
        __attribute__((aligned(16)));
#endif

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _params Per type pair parameters of this potential
    */
    DEVICE EvaluatorPairEwald(ForceReal _rsq, ForceReal _rcutsq, const param_type& _params)
        : rsq(_rsq), rcutsq(_rcutsq), kappa(_params.kappa), alpha(_params.alpha)
        {
        }

    //! Ewald uses charge !!!
    DEVICE static bool needsCharge()
        {
        return true;
        }
    //! Accept the optional charge values.
    /*! \param qi Charge of particle i
        \param qj Charge of particle j
    */
    DEVICE void setCharge(ForceReal qi, ForceReal qj)
        {
        qiqj = qi * qj;
        }

    //! Evaluate the force and energy
    /*! \param force_divr Output parameter to write the computed force divided by r.
        \param pair_eng Output parameter to write the computed pair energy
        \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the
       cutoff \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are
       performed in PotentialPair.

        \return True if they are evaluated or false if they are not because we are beyond the cutoff
    */
    DEVICE bool evalForceAndEnergy(ForceReal& force_divr, ForceReal& pair_eng, bool energy_shift)
        {
        if (rsq < rcutsq && qiqj != 0)
            {
            ForceReal rinv = fast::rsqrt(rsq);
            ForceReal r = ForceReal(1.0) / rinv;
            ForceReal r2inv = ForceReal(1.0) / rsq;

            ForceReal arg1 = kappa * r + alpha / (ForceReal(2.0) * kappa);
            ForceReal arg2 = kappa * r - alpha / (ForceReal(2.0) * kappa);
            ForceReal expfac1 = fast::exp(alpha * r);
            ForceReal expfac2 = fast::exp(-alpha * r);
            ForceReal val
                = ForceReal(0.5) * (fast::erfc(arg1) * expfac1 + fast::erfc(arg2) * expfac2) * rinv;

            force_divr = qiqj * r2inv
                         * (val
                            + expfac2 * ForceReal(2.0) * kappa * fast::exp(-arg2 * arg2)
                                  / fast::sqrt(ForceReal(M_PI))
                            + alpha * ForceReal(0.5) * expfac2 * fast::erfc(arg2)
                            - alpha * ForceReal(0.5) * expfac1 * fast::erfc(arg1));
            pair_eng = qiqj * val;

            return true;
            }
        else
            return false;
        }

    DEVICE ForceReal evalPressureLRCIntegral()
        {
        return 0;
        }

    DEVICE ForceReal evalEnergyLRCIntegral()
        {
        return 0;
        }

#ifndef __HIPCC__
    //! Get the name of this potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return std::string("ewald");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    ForceReal rsq;    //!< Stored rsq from the constructor
    ForceReal rcutsq; //!< Stored rcutsq from the constructor
    ForceReal kappa;  //!< Splitting parameter
    ForceReal alpha;  //!< Debye screening parameter
    ForceReal qiqj;   //!< product of qi and qj
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_EWALD_H__
