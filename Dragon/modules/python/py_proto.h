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

#ifndef DRAGON_PYTHON_PY_PROTO_H_
#define DRAGON_PYTHON_PY_PROTO_H_

#include "py_dragon.h"

namespace dragon {

namespace python {

void AddProtoMethods(pybind11::module& m) {
    /*! \brief Extented C-Style OperatorDef */
    pybind11::class_<OperatorDef>(m, "OperatorDef")
        .def(pybind11::init())
        .def("CopyFrom", [](
            OperatorDef*            self,
            OperatorDef*            other) {
            self->CopyFrom(*other);
      }).def("ParseFrom", [](
            OperatorDef*            self,
            const string&           serialized) {
            self->ParseFromString(serialized);
      }).def("SerializeAs", [](
            OperatorDef*            self) {
            return pybind11::bytes(self->SerializeAsString());
      }).def("add_input", [](
            OperatorDef*            self,
            const string&           input) {
          self->add_input(input);
      }).def("add_output", [](
            OperatorDef*            self,
            const string&           output) {
          self->add_output(output);
      }).def_property("name",
          [](OperatorDef* self) {
              return self->name(); },
          [](OperatorDef* self, const string& name) {
              self->set_name(name);
      }).def_property("type",
          [](OperatorDef* self) {
            return self->type(); },
          [](OperatorDef* self, const string& type) {
              self->set_type(type);
      }).def_property("input",
          [](OperatorDef* self) -> vector<string> {
              return { self->input().begin(), self->input().end() }; },
          [](OperatorDef* self, const vector<string>& input) {
              *(self->mutable_input()) = { input.begin(), input.end() };
      }).def_property("output",
          [](OperatorDef* self) -> vector<string> {
             return{ self->output().begin(), self->output().end() }; },
          [](OperatorDef* self, const vector<string>& output) {
              *(self->mutable_output()) = { output.begin(), output.end() };
      });
}

}  // namespace python

}  // namespace dragon

#endif DRAGON_PYTHON_PY_PROTO_H_