/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *    <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_MODULES_PYTHON_OPERATOR_H_
#define DRAGON_MODULES_PYTHON_OPERATOR_H_

#include "dragon/modules/python/common.h"
#include "dragon/utils/eigen_utils.h"

namespace dragon {

namespace python {

namespace ops {

void RegisterModule(py::module& m) {
  /*! \brief Return the registered operators */
  m.def("RegisteredOperators", []() { return CPUOperatorRegistry()->keys(); });

  /*! \brief Return the non-gradient operators */
  m.def("NoGradientOperators", []() { return NoGradientRegistry()->keys(); });
}

} // namespace ops

} // namespace python

} // namespace dragon

#endif // DRAGON_MODULES_PYTHON_OPERATOR_H_
