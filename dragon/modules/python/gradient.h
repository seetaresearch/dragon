/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_MODULES_PYTHON_GRADIENT_H_
#define DRAGON_MODULES_PYTHON_GRADIENT_H_

#include <dragon/core/gradient.h>

#include "dragon/modules/python/common.h"

namespace dragon {

namespace python {

void RegisterModule_gradient(py::module& m) {
  /*! \brief GradientTape class */
  py::class_<GradientTape>(m, "GradientTape")
      /*! \brief Default constructor */
      .def(py::init())

      /*! \brief Pickle support */
      .def(py::pickle(
          [](GradientTape* self) { return py::bytes(); },
          [](py::bytes data) { return GradientTape(); }))

      /*! \brief Create gradient defs */
      .def(
          "CreateGradientDefs",
          [](GradientTape* self,
             const vector<string>& op_raw_defs,
             const vector<string>& targets,
             const vector<string>& grad_targets) {
            vector<OperatorDef> op_parsed_defs(op_raw_defs.size());
            vector<OperatorDef*> op_defs(op_raw_defs.size());
            for (size_t i = 0; i < op_raw_defs.size(); ++i) {
              op_parsed_defs[i].ParseFromString(op_raw_defs[i]);
              op_defs[i] = &op_parsed_defs[i];
            }
            self->CreateGradientDefs(op_defs, targets, grad_targets);
            vector<py::bytes> grad_raw_defs(self->def().op_size());
            for (size_t i = 0; i < grad_raw_defs.size(); ++i) {
              grad_raw_defs[i] = self->def().op(i).SerializeAsString();
            }
            return grad_raw_defs;
          });
}

} // namespace python

} // namespace dragon

#endif // DRAGON_MODULES_PYTHON_GRADIENT_H_
