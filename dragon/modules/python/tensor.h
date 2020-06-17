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

#ifndef DRAGON_MODULES_PYTHON_TENSOR_H_
#define DRAGON_MODULES_PYTHON_TENSOR_H_

#include "dragon/modules/python/common.h"

namespace dragon {

namespace python {

namespace tensor {

void RegisterModule(py::module& m) {
  /*! \brief Export the Tensor class */
  py::class_<Tensor>(m, "Tensor")
      /*! \brief Return the number of dimensions */
      .def_property_readonly("ndim", &Tensor::ndim)

      /*! \brief Return all the dimensions */
      .def_property_readonly("dims", &Tensor::dims)

      /*! \brief Return the total number of elements */
      .def_property_readonly("size", &Tensor::size)

      /*! \brief Return the total number of bytes */
      .def_property_readonly("nbytes", &Tensor::nbytes)

      /*! \brief Return the data type */
      .def_property_readonly(
          "dtype",
          [](Tensor* self) { return ::dragon::types::to_string(self->meta()); })

      /*! \brief Return the device information */
      .def_property_readonly(
          "device",
          [](Tensor* self) {
            if (self->has_memory()) {
              auto mem_info = self->memory()->info();
              return std::tuple<string, int>(
                  mem_info["device_type"], atoi(mem_info["device_id"].c_str()));
            } else {
              return std::tuple<string, int>("Unknown", 0);
            }
          })

      /*! \brief Return the raw const data pointer */
      .def(
          "data",
          [](Tensor* self, const string& device_type) {
            intptr_t pointer = 0;
            if (device_type == "cpu") {
              pointer = (intptr_t)self->raw_data<CPUContext>();
            } else if (device_type == "cuda") {
              pointer = (intptr_t)self->raw_data<CUDAContext>();
            } else if (device_type == "cnml") {
              pointer = (intptr_t)self->raw_data<CNMLContext>();
            } else {
              LOG(FATAL) << "Unknown device type: " << device_type;
            }
            return pointer;
          })

      /*! \brief Return the raw mutable data pointer */
      .def(
          "mutable_data",
          [](Tensor* self, const string& device_type) {
            intptr_t pointer = 0;
            if (device_type == "cpu") {
              pointer = (intptr_t)self->raw_mutable_data<CPUContext>();
            } else if (device_type == "cuda") {
              pointer = (intptr_t)self->raw_mutable_data<CUDAContext>();
            } else if (device_type == "cnml") {
              pointer = (intptr_t)self->raw_mutable_data<CNMLContext>();
            } else {
              LOG(FATAL) << "Unknown device type: " << device_type;
            }
            return pointer;
          })

      /*! \brief Reshape to the given dimensions */
      .def(
          "Reshape",
          [](Tensor* self, vec64_t& shape) {
            self->Reshape(shape);
            return self->has_memory();
          })

      /*! \brief Copy data from the another tensor */
      .def(
          "CopyFrom",
          [](Tensor* self,
             Tensor* other,
             const string& dest_dev,
             const string& src_dev) {
            DeviceOption src_opt, dest_opt;
            src_opt.ParseFromString(src_dev);
            dest_opt.ParseFromString(dest_dev);
            self->set_meta(other->meta())->ReshapeLike(*other);
            if (dest_opt.device_type() == PROTO_CUDA) {
              CUDADeviceGuard guard(dest_opt.device_id());
              if (src_opt.device_type() == PROTO_CUDA) {
                // CUDA <- CUDA
                CUDAContext::Memcpy<CUDAContext, CUDAContext>(
                    other->nbytes(),
                    self->raw_mutable_data<CUDAContext>(),
                    other->raw_data<CUDAContext>(),
                    dest_opt.device_id());
              } else {
                // CUDA <- CPU
                CUDAContext::Memcpy<CUDAContext, CPUContext>(
                    other->nbytes(),
                    self->raw_mutable_data<CUDAContext>(),
                    other->raw_data<CPUContext>(),
                    dest_opt.device_id());
              }
            } else {
              if (src_opt.device_type() == PROTO_CUDA) {
                // CPU <- CUDA
                CUDADeviceGuard guard(src_opt.device_id());
                CUDAContext::Memcpy<CPUContext, CUDAContext>(
                    other->nbytes(),
                    self->raw_mutable_data<CPUContext>(),
                    other->raw_data<CUDAContext>(),
                    src_opt.device_id());
              } else {
                // CPU <- CPU
                CPUContext::Memcpy<CUDAContext, CUDAContext>(
                    other->nbytes(),
                    self->raw_mutable_data<CPUContext>(),
                    other->raw_data<CPUContext>());
              }
            }
          })

      /*! \brief Construct from the shape and data type */
      .def(
          "FromShape",
          [](Tensor* self, const vector<int64_t>& dims, const string& dtype) {
            const auto& meta = ::dragon::types::to_meta(dtype);
            CHECK(meta.id() != 0)
                << "\nUnsupported tensor type: " + dtype + ".";
            self->set_meta(meta)->Reshape(dims);
            self->raw_mutable_data<CPUContext>();
            return self;
          },
          py::return_value_policy::reference_internal)

      /*! \brief Construct from an external pointer */
      .def(
          "FromPointer",
          [](Tensor* self,
             const vector<int64_t>& dims,
             const string& dtype,
             const string& dev,
             const intptr_t pointer) {
            const auto& meta = ::dragon::types::to_meta(dtype);
            DeviceOption opt;
            opt.ParseFromString(dev);
            auto* data = (void*)pointer;
            self->Reset(); // Reduce the risk taking dangling pointer
            self->set_meta(meta)->Reshape(dims);
            auto nbytes = self->nbytes();
            auto* memory = self->memory();
            if (memory == nullptr) memory = new UnifiedMemory();
            switch (opt.device_type()) {
              case PROTO_CPU:
                memory->set_cpu_data(data, nbytes);
                break;
              case PROTO_CUDA:
                memory->set_cuda_data(data, nbytes, opt.device_id());
                break;
              case PROTO_CNML:
              default:
                LOG(FATAL) << "Unsupported pointer device: "
                           << opt.device_type();
            }
            self->set_memory(memory);
            if (self->ExternalDeleter) self->ExternalDeleter();
            return self;
          },
          py::return_value_policy::reference_internal)

      /*! \brief Construct from a numpy array */
      .def(
          "FromNumpy",
          [](Tensor* self, py::object object) {
            auto* array = PyArray_GETCONTIGUOUS(
                reinterpret_cast<PyArrayObject*>(object.ptr()));
            const auto& meta = types::from_npy(PyArray_TYPE(array));
            if (meta.id() == 0) {
              LOG(FATAL) << "Unsupported numpy array type.";
            }
            auto ndim = PyArray_NDIM(array);
            auto* npy_dims = PyArray_DIMS(array);
            auto* data = static_cast<void*>(PyArray_DATA(array));
            vector<int64_t> dims;
            for (int i = 0; i < ndim; i++)
              dims.push_back(npy_dims[i]);
            self->set_meta(meta)->Reshape(dims);
            auto* memory = self->memory();
            if (memory == nullptr) memory = new UnifiedMemory();
            memory->set_cpu_data(data, self->nbytes());
            self->set_memory(memory);
            if (self->ExternalDeleter) self->ExternalDeleter();
            self->ExternalDeleter = [array]() -> void { Py_XDECREF(array); };
            return self;
          },
          py::return_value_policy::reference_internal)

      /*! \brief Construct from a dlpack tensor */
      .def(
          "FromDLPack",
          [](Tensor* self, py::object object) {
            return DLPackWrapper(self).From(object);
          },
          py::return_value_policy::reference_internal)

      /*! \brief Switch memory to the cpu context */
      .def(
          "ToCPU",
          [](Tensor* self) {
            CHECK(self->has_memory())
                << "\nTensor(" << self->name() << ") "
                << "does not initialize or had been reset.";
            self->memory()->ToCPU();
          })

      /*! \brief Switch memory to the cuda context */
      .def(
          "ToCUDA",
          [](Tensor* self, int device_id) {
#ifdef USE_CUDA
            CHECK(self->has_memory())
                << "\nTensor(" << self->name() << ") "
                << "does not initialize or had been reset.";
            self->memory()->SwitchToCUDADevice(device_id);
#else
       CUDA_NOT_COMPILED;
#endif
          })

      /*! \brief Return a numpy array sharing the data */
      .def(
          "ToNumpy",
          [](Tensor* self, bool readonly) {
            CHECK_GT(self->count(), 0)
                << "\nShare the data of an empty tensor.";
            auto npy_type = types::to_npy(self->meta());
            if (npy_type == -1) {
              LOG(FATAL) << "Tensor(" + self->name() + ") with type "
                         << ::dragon::types::to_string(self->meta())
                         << " is not supported by numpy array.";
            }
            vector<npy_intp> dims;
            for (auto dim : self->dims())
              dims.push_back(dim);
            auto* data = readonly
                ? const_cast<void*>(self->raw_data<CPUContext>())
                : self->raw_mutable_data<CPUContext>();
            auto* array = PyArray_SimpleNewFromData(
                self->ndim(), dims.data(), npy_type, data);
            return py::reinterpret_steal<py::object>(array);
          })

      /*! \brief Return a dlpack tensor sharing the data */
      .def("ToDLPack", [](Tensor* self, const string& dev, bool readonly) {
        CHECK_GT(self->count(), 0) << "\nShare the data of an empty tensor.";
        DeviceOption opt;
        opt.ParseFromString(dev);
        return DLPackWrapper(self).To(opt);
      });
}

} // namespace tensor

} // namespace python

} // namespace dragon

#endif // DRAGON_MODULES_PYTHON_TENSOR_H_
