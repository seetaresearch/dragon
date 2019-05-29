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

#ifndef DRAGON_PYTHON_PY_CONFIG_H_
#define DRAGON_PYTHON_PY_CONFIG_H_

#include "py_dragon.h"

namespace dragon {

namespace python {

void AddConfigMethods(pybind11::module& m) {
    m.def("SetLoggingLevel", [](const string& level) {
        SetLogDestination(StrToLogSeverity(level));
    });
}

}  // namespace python

}  // namespace dragon

#endif  // DRAGON_PYTHON_PY_CONFIG_H_