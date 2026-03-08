// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __PAIR_EVALUATOR_ExpandedMie_H__
#define __PAIR_EVALUATOR_ExpandedMie_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "EvaluatorPairLJ.h"
#include "hoomd/HOOMDMath.h"

/*! \file EvaluatorPairExpandedMie.h
    \brief Defines the pair evaluator class for Expanded Mie potentials
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
//! Class for evaluating the Expanded Mie pair potential
class EvaluatorPairExpandedMie
    {
    public:
    //! Define the parameter type used by this pair potential evaluator
    struct param_type
        {
        ForceReal repulsive;  //!< Lumped repulsive term to simplify/speed up computation
        ForceReal attractive; //!< Lumped attractive term to simplify/speed up computation
        ForceReal n_pow;      //!< Higher exponent for potential
        ForceReal m_pow;      //!< Lower exponent for potential
        ForceReal delta;      //!< shift in radial distance for use in Mie potential
        ForceReal sigma;
        ForceReal epsilon;

        DEVICE void load_shared(char*& ptr, unsigned int& available_bytes) { }

        HOSTDEVICE void allocate_shared(char*& ptr, unsigned int& available_bytes) const { }

#ifdef ENABLE_HIP
        //! set CUDA memory hints
        void set_memory_hint() const { }
#endif

#ifndef __HIPCC__
        param_type()
            : repulsive(0), attractive(0), n_pow(0), m_pow(0), delta(0), sigma(0), epsilon(0)
            {
            }

        param_type(const pybind11::dict v, bool managed = false)
            {
            n_pow = v["n"].cast<ForceReal>();
            m_pow = v["m"].cast<ForceReal>();

            sigma = v["sigma"].cast<ForceReal>();
            epsilon = v["epsilon"].cast<ForceReal>();

            ForceReal prefactor
                = (n_pow / (n_pow - m_pow)) * fast::pow(n_pow / m_pow, m_pow / (n_pow - m_pow));
            repulsive = prefactor * epsilon * fast::pow(sigma, n_pow);
            attractive = prefactor * epsilon * fast::pow(sigma, m_pow);

            delta = v["delta"].cast<ForceReal>();
            }

        pybind11::dict asDict() const
            {
            pybind11::dict v;
            v["n"] = n_pow;
            v["m"] = m_pow;

            v["epsilon"] = epsilon;
            v["sigma"] = sigma;

            v["delta"] = delta;
            return v;
            }
#endif
        } __attribute__((aligned(16)));

    //! Constructs the pair potential evaluator
    /*! \param _rsq Squared distance between the particles
        \param _rcutsq Squared distance at which the potential goes to 0
        \param _n First, larger exponent that captures hard-core repulsion
        \param _m Second, smaller exponent that captures attraction
        \param _params Per type pair parameters of this potential
        \param _delta Horizontal shift in r
    */
    DEVICE
    EvaluatorPairExpandedMie(const ForceReal _rsq, const ForceReal _rcutsq, const param_type& _params)
        : rsq(_rsq), rcutsq(_rcutsq), repulsive(_params.repulsive), attractive(_params.attractive),
          n_pow(_params.n_pow), m_pow(_params.m_pow), delta(_params.delta)
        {
        }

    //! ExpandedMie doesn't use charge
    DEVICE static bool needsCharge()
        {
        return false;
        }
    //! Accept the optional charge values.
    /*! \param qi Charge of particle i
        \param qj Charge of particle j
    */
    DEVICE void setCharge(ForceReal qi, ForceReal qj) const { }

    //! Evaluate the force and energy
    /*! \param force_divr Output parameter to write the computed force divided by r.
        \param pair_eng Output parameter to write the computed pair energy
        \param energy_shift If true, the potential must be shifted so that V(r) is continuous at the
       cutoff \note There is no need to check if rsq < rcutsq in this method. Cutoff tests are
       performed in PotentialPair.

        \return True if they are evaluated or false if they are not because we are beyond the cutoff
    */
    DEVICE bool evalForceAndEnergy(ForceReal& force_divr, ForceReal& pair_eng, bool energy_shift) const
        {
        // precompute some quantities
        ForceReal rinv = fast::rsqrt(rsq);
        ForceReal r = ForceReal(1.0) / rinv;

        // compute the force divided by r in force_divr
        if (rsq < rcutsq && repulsive != 0)
            {
            ForceReal rmd = r - delta;
            ForceReal rmdinv = ForceReal(1.0) / rmd;
            ForceReal rmd2inv = rmdinv * rmdinv;
            ForceReal rmdninv = fast::pow(rmd2inv, n_pow / ForceReal(2.0));
            ForceReal rmdminv = fast::pow(rmd2inv, m_pow / ForceReal(2.0));
            force_divr
                = rinv * rmdinv * (n_pow * repulsive * rmdninv - m_pow * attractive * rmdminv);

            pair_eng = repulsive * rmdninv - attractive * rmdminv;

            if (energy_shift)
                {
                ForceReal r_cut = fast::sqrt(rcutsq);
                ForceReal r_cut_shifted = r_cut - delta;

                ForceReal r_cut_n_inv = fast::pow(r_cut_shifted, -n_pow);
                ForceReal r_cut_m_inv = fast::pow(r_cut_shifted, -m_pow);
                pair_eng -= repulsive * r_cut_n_inv - attractive * r_cut_m_inv;
                }
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
    /*! \returns The potential name. Must be short and all lowercase, as this is the name energies
       will be logged as via analyze.log.
    */
    static std::string getName()
        {
        return std::string("expanded_mie");
        }

    std::string getShapeSpec() const
        {
        throw std::runtime_error("Shape definition not supported for this pair potential.");
        }
#endif

    protected:
    ForceReal rsq;        //!< distance between particles squared
    ForceReal rcutsq;     //!< the cutoff radius of the potential squared
    ForceReal repulsive;  //!< Lumped repulsive term to simplify/speed up computation
    ForceReal attractive; //!< Lumped attractive term to simplify/speed up computation
    ForceReal n_pow;      //!< Higher exponent for potential
    ForceReal m_pow;      //!< Lower exponent for potential
    ForceReal delta;      //!< shift in radial distance for use in Mie potential
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __PAIR_EVALUATOR_EXPANDEDMIE_H__
