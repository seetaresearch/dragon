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

#ifndef DRAGON_PYTHON_PY_IO_H_
#define DRAGON_PYTHON_PY_IO_H_

#include "py_dragon.h"

namespace dragon {

namespace python {

void AddIOMethods(pybind11::module& m) {
    m.def("Snapshot", [](
        const string&       filename,
        vector<string>&     names,
        const int           format) {
        vector<Tensor*> tensors;
        switch (format) {
            case 0:  // Pickle
                LOG(FATAL) << "Format depends on Pickle. "
                              "Can't be used in C++.";
                break;
            case 1:  // CaffeModel
                for (const auto& e : names)
                    tensors.emplace_back(ws()->GetTensor(e));
                SavaCaffeModel(filename, tensors);
                break;
            default:
                LOG(FATAL) << "Unknwon format, code: " << format;
        }
    });

    m.def("Restore", [](
        const string&       filename,
        const int           format) {
        switch (format) {
            case 0:  // Pickle
                LOG(FATAL) << "Format depends on Pickle. "
                    "Can't be used in C++.";
                break;
            case 1:  // CaffeModel
                LoadCaffeModel(filename, ws());
                break;
            default: 
                LOG(FATAL) << "Unknwon format, code: " << format;
        }
    });
}

}  // namespace python

}  // namespace dragon

#endif  // DRAGON_PYTHON_PY_IO_H_