// Copyright (c) 2009-2026 The Regents of the University of Michigan.
// Part of HOOMD-blue, released under the BSD 3-Clause License.

#ifndef __EVALUATOR_EXTERNAL_PERIODIC_H__
#define __EVALUATOR_EXTERNAL_PERIODIC_H__

#ifndef __HIPCC__
#include <string>
#endif

#include "hoomd/BoxDim.h"
#include "hoomd/HOOMDMath.h"
#include "hoomd/VectorMath.h"
#include <math.h>

/*! \file EvaluatorExternalPeriodic.h
    \brief Defines the external potential evaluator to induce a periodic ordered phase
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
//! Class for evaluating sphere constraints
/*! <b>General Overview</b>
    EvaluatorExternalPeriodic is an evaluator to induce a periodic modulation on the concentration
   profile in the system, e.g. to generate a periodic phase in a system of diblock copolymers.

    The external potential \f$V(\vec{r}) \f$ is implemented using the following formula:

    \f[
    V(\vec{r}) = A * \tanh\left[\frac{1}{2 \pi p w} \cos\left(p \vec{b}_i\cdot\vec{r}\right)\right]
    \f]

    where \f$A\f$ is the ordering parameter, \f$\vec{b}_i\f$ is the reciprocal lattice vector
   direction \f$i=0..2\f$, \f$p\f$ the periodicity and \f$w\f$ the interface width (relative to the
   distance \f$2\pi/|\mathbf{b_i}|\f$ between planes in the \f$i\f$-direction). The modulation is
   one-dimensional. It extends along the lattice vector \f$\mathbf{a}_i\f$ of the simulation cell.
*/
class EvaluatorExternalPeriodic
    {
    public:
    //! type of parameters this external potential accepts
    struct param_type
        {
        Scalar A;
        Scalar w;
        int i;
        int p;

#ifndef __HIPCC__
        param_type() : A(1.0), w(1.0), i(0), p(1) { }

        param_type(pybind11::dict params)
            {
            i = params["i"].cast<int>();
            A = params["A"].cast<Scalar>();
            w = params["w"].cast<Scalar>();
            p = params["p"].cast<int>();
            }

        param_type(int i_, Scalar A_, Scalar w_, int p_) : A(A_), w(w_), i(i_), p(p_) { }

        pybind11::dict toPython()
            {
            pybind11::dict d;
            d["i"] = i;
            d["A"] = A;
            d["w"] = w;
            d["p"] = p;
            return d;
            }
#endif
        } __attribute__((aligned(16)));

    typedef void* field_type; // dummy type

    //! Constructs the constraint evaluator
    /*! \param X position of particle
        \param box box dimensions
        \param params per-type parameters of external potential
    */
    DEVICE EvaluatorExternalPeriodic(ForceReal3 X,
                                     quat<Scalar> q,
                                     const BoxDim& box,
                                     const param_type& params,
                                     const field_type& field)
        : m_pos(X), m_box(box), m_index(params.i), m_orderParameter(params.A),
          m_interfaceWidth(params.w), m_periodicity(params.p)
        {
        }

    //! External Periodic doesn't need charges
    DEVICE static bool needsCharge()
        {
        return false;
        }
    //! Accept the optional charge value.
    /*! \param qi Charge of particle i
     */
    DEVICE void setCharge(ForceReal qi) { }

    //! Declares additional virial contributions are needed for the external field
    /*! No contributions
     */
    DEVICE static bool requestFieldVirialTerm()
        {
        return true;
        }

    //! Evaluate the force, energy and virial
    /*! \param F force vector
        \param T torque vector
        \param energy value of the energy
        \param virial array of six scalars for the upper triangular virial tensor
    */
    DEVICE void
    evalForceTorqueEnergyAndVirial(ForceReal3& F, ForceReal3& T, ForceReal& energy, ForceReal* virial)
        {
        F.x = ForceReal(0.0);
        F.y = ForceReal(0.0);
        F.z = ForceReal(0.0);

        T.x = ForceReal(0.0);
        T.y = ForceReal(0.0);
        T.z = ForceReal(0.0);

        energy = ForceReal(0.0);

        // For this potential, since it uses scaled positions, the virial is always zero.
        for (unsigned int i = 0; i < 6; i++)
            virial[i] = ForceReal(0.0);

        ForceReal V_box = ForceReal(m_box.getVolume());
        // compute the vector pointing from P to V
        Scalar3 a2_s = make_scalar3(0, 0, 0);
        Scalar3 a3_s = make_scalar3(0, 0, 0);
        if (m_index == 0)
            {
            a2_s = m_box.getLatticeVector(1);
            a3_s = m_box.getLatticeVector(2);
            }
        else if (m_index == 1)
            {
            a2_s = m_box.getLatticeVector(2);
            a3_s = m_box.getLatticeVector(0);
            }
        else if (m_index == 2)
            {
            a2_s = m_box.getLatticeVector(0);
            a3_s = m_box.getLatticeVector(1);
            }

        ForceReal3 a2 = make_forcereal3(ForceReal(a2_s.x), ForceReal(a2_s.y), ForceReal(a2_s.z));
        ForceReal3 a3 = make_forcereal3(ForceReal(a3_s.x), ForceReal(a3_s.y), ForceReal(a3_s.z));

        ForceReal3 b = ForceReal(2.0 * M_PI)
                    * make_forcereal3(a2.y * a3.z - a2.z * a3.y,
                                   a2.z * a3.x - a2.x * a3.z,
                                   a2.x * a3.y - a2.y * a3.x)
                    / V_box;
        ForceReal clipParameter, arg, clipcos, tanH, sechSq;

        ForceReal3 q = b * ForceReal(m_periodicity);
        clipParameter = ForceReal(1.0) / ForceReal(2.0 * M_PI) / (ForceReal(m_periodicity) * ForceReal(m_interfaceWidth));
        arg = dot(m_pos, q);
        clipcos = clipParameter * fast::cos(arg);
        tanH = slow::tanh(clipcos);
        sechSq = (ForceReal(1.0) - tanH * tanH);

        F = ForceReal(m_orderParameter) * sechSq * clipParameter * fast::sin(arg) * q;
        energy = ForceReal(m_orderParameter) * tanH;
        }

#ifndef __HIPCC__
    //! Get the name of this potential
    /*! \returns The potential name.
     */
    static std::string getName()
        {
        return std::string("periodic");
        }
#endif

    protected:
    ForceReal3 m_pos; //!< particle position
    BoxDim m_box;     //!< box dimensions
    unsigned int
        m_index; //!< cartesian index of direction along which the lamellae should be oriented
    Scalar m_orderParameter;    //!< ordering parameter
    Scalar m_interfaceWidth;    //!< width of interface between lamellae (relative to box length)
    unsigned int m_periodicity; //!< number of lamellae of each type
    };

    } // end namespace md
    } // end namespace hoomd

#endif // __EVALUATOR_EXTERNAL_LAMELLAR_H__
