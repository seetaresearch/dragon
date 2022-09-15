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

#ifndef DRAGON_MODULES_PYTHON_PROTO_H_
#define DRAGON_MODULES_PYTHON_PROTO_H_

#include <dragon/core/common.h>

#include "dragon/modules/python/common.h"

namespace dragon {

namespace python {

void RegisterModule_proto(py::module& m) {
  py::class_<OperatorDef>(m, "OperatorDef")
      /*! \brief Default constructor */
      .def(py::init())

      /*! \brief Copy from a def */
      .def(
          "CopyFrom",
          [](OperatorDef* self, OperatorDef* other) { self->CopyFrom(*other); })

      /*! \brief Decode from a serialized string */
      .def(
          "ParseFrom",
          [](OperatorDef* self, const string& serialized) {
            self->ParseFromString(serialized);
          })

      /*! \brief Encode to a serialized string */
      .def(
          "SerializeAs",
          [](OperatorDef* self) {
            return py::bytes(self->SerializeAsString());
          })

      /*! \brief Return a new def with different input and output */
      .def(
          "DeriveTo",
          [](OperatorDef* self,
             const vector<string>& inputs,
             const vector<string>& outputs) {
            auto* new_def = new OperatorDef(*self);
            *(new_def->mutable_input()) = {inputs.begin(), inputs.end()};
            *(new_def->mutable_output()) = {outputs.begin(), outputs.end()};
            return new_def;
          },
          py::return_value_policy::take_ownership)

      /*! \brief Return a string of data */
      .def("__repr__", [](OperatorDef* self) { return self->DebugString(); })

      /*! \brief Add an input */
      .def(
          "add_input",
          [](OperatorDef* self, const string& input) {
            self->add_input(input);
          })

      /*! \brief Add an output */
      .def(
          "add_output",
          [](OperatorDef* self, const string& output) {
            self->add_output(output);
          })

      /*! \brief Property "name" */
      .def_property(
          "name",
          [](OperatorDef* self) { return self->name(); },
          [](OperatorDef* self, const string& name) { self->set_name(name); })

      /*! \brief Property "type" */
      .def_property(
          "type",
          [](OperatorDef* self) { return self->type(); },
          [](OperatorDef* self, const string& type) { self->set_type(type); })

      /*! \brief Property "input" */
      .def_property(
          "input",
          [](OperatorDef* self) -> vector<string> {
            return {self->input().begin(), self->input().end()};
          },
          [](OperatorDef* self, const vector<string>& input) {
            *(self->mutable_input()) = {input.begin(), input.end()};
          })

      /*! \brief Property "output" */
      .def_property(
          "output",
          [](OperatorDef* self) -> vector<string> {
            return {self->output().begin(), self->output().end()};
          },
          [](OperatorDef* self, const vector<string>& output) {
            *(self->mutable_output()) = {output.begin(), output.end()};
          });
}

} // namespace python

} // namespace dragon

#endif // DRAGON_MODULES_PYTHON_PROTO_H_
