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

#ifndef DRAGON_PYTHON_PY_TENSOR_H_
#define DRAGON_PYTHON_PY_TENSOR_H_

#include "py_dragon.h"

namespace dragon {

namespace python {

void AddTensorMethods(pybind11::module& m) {
    /*! \brief Export the Tensor class */
    pybind11::class_<Tensor>(m, "Tensor")
        /*! \brief Return the number of dimensions */
        .def_property_readonly("ndim", &Tensor::ndim)

        /*! \brief Return all the dimensions */
        .def_property_readonly("dims", &Tensor::dims)

        /*! \brief Return the total number of elements */
        .def_property_readonly("size", &Tensor::size)

        /*! \brief Return the data type */
        .def_property_readonly("dtype", [](Tensor* self) {
            return TypeMetaToString(self->meta());
        })
        
        /*! \brief Return the device information */
        .def_property_readonly("device", [](Tensor* self) {
            if (self->has_memory()) {
                auto mem_info = self->memory()->info();
                return std::tuple<string, int>(
                    mem_info["device_type"], atoi(
                        mem_info["device_id"].c_str()));
            } else {
                return std::tuple<string, int>("Unknown", 0);
            }
        })

        /*! \brief Switch the memory to the cpu context */
        .def("ToCPU", [](Tensor* self) {
            CHECK(self->has_memory())
                << "\nTensor(" << self->name() << ") "
                << "does not initialize or had been reset.";
            self->memory()->ToCPU();
        })

        /*! \brief Switch the memory to the cuda context */
        .def("ToCUDA", [](Tensor* self, int device_id) {
#ifdef WITH_CUDA
           CHECK(self->has_memory()) 
               << "\nTensor(" << self->name() << ") "
               << "does not initialize or had been reset.";
           self->memory()->SwitchToCUDADevice(device_id);
#else
           CUDA_NOT_COMPILED;
#endif
     });
}

}  // namespace python

}  // namespace dragon

#endif  // DRAGON_PYTHON_PY_TENSOR_H_