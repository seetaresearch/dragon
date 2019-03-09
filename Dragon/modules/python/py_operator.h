/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_PYTHON_PY_OPERATOR_H_
#define DRAGON_PYTHON_PY_OPERATOR_H_

#include "py_dragon.h"

namespace dragon {

namespace python {

void AddOperatorMethods(pybind11::module& m) {
    /*! \brief Return all the registered operators */
    m.def("RegisteredOperators", []() { return CPUOperatorRegistry()->keys(); });

    /*! \brief Return all the operators without gradients */
    m.def("NoGradientOperators", []() { return NoGradientRegistry()->keys(); });

    /*! \brief Run a operator from the def reference */
    m.def("RunOperator", [](
        OperatorDef*        def,
        const bool          verbose) {
        pybind11::gil_scoped_release g;
        if (verbose) {
            // It is not a good design to print the debug string
            std::cout << def->DebugString() << std::endl;
        }
        ws()->RunOperator(*def);
    });

    /*! \brief Run a operator from the serialized def */
    m.def("RunOperator", [](
        const string&       serialized,
        const bool          verbose) {
        OperatorDef def;
        CHECK(def.ParseFromString(serialized));
        pybind11::gil_scoped_release g;
        if (verbose) {
            // It is not a good design to print the debug string
            std::cout << def.DebugString() << std::endl;
        }
        ws()->RunOperatorOnce(def);
    });
}

}  // namespace python

}  // namespace dragon

#endif  // DRAGON_PYTHON_PY_OPERATOR_H_