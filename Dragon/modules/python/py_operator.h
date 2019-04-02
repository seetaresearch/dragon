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
    /*! \brief Return the registered operators */
    m.def("RegisteredOperators", []() {
        return CPUOperatorRegistry()->keys(); 
    });

    /*! \brief Return the non-gradient operators */
    m.def("NoGradientOperators", []() {
        return NoGradientRegistry()->keys(); 
    });
}

}  // namespace python

}  // namespace dragon

#endif  // DRAGON_PYTHON_PY_OPERATOR_H_