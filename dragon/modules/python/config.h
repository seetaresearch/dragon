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

#ifndef DRAGON_MODULES_PYTHON_CONFIG_H_
#define DRAGON_MODULES_PYTHON_CONFIG_H_

#include "dragon/modules/python/common.h"

namespace dragon {

namespace python {

namespace config {

void RegisterModule(py::module& m) {
  m.def("SetLoggingLevel", [](const string& severity) {
    SetLogDestination(severity);
  });
}

} // namespace config

} // namespace python

} // namespace dragon

#endif // DRAGON_MODULES_PYTHON_CONFIG_H_
