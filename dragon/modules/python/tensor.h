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

#ifndef DRAGON_MODULES_PYTHON_TENSOR_H_
#define DRAGON_MODULES_PYTHON_TENSOR_H_

#include "dragon/modules/python/dlpack.h"
#include "dragon/modules/python/numpy.h"

namespace dragon {

namespace python {

void RegisterModule_tensor(py::module& m) {
  /*! \brief Tensor class */
  py::class_<Tensor>(m, "Tensor")
      /*! \brief Return the tensor name */
      .def_property_readonly("name", &Tensor::name)

      /*! \brief Return the number of dimensions */
      .def_property_readonly("ndim", &Tensor::ndim)

      /*! \brief Return the dimensions */
      .def_property_readonly("dims", &Tensor::dims)

      /*! \brief Return the number of elements */
      .def_property_readonly("size", &Tensor::size)

      /*! \brief Return the byte length of one element */
      .def_property_readonly(
          "itemsize", [](Tensor* self) { return self->meta().itemsize(); })

      /*! \brief Return the byte length of all elements */
      .def_property_readonly("nbytes", &Tensor::nbytes)

      /*! \brief Return the byte length of allocated memory */
      .def_property_readonly("capacity", &Tensor::capacity)

      /*! \brief Return the data type */
      .def_property_readonly(
          "dtype",
          [](Tensor* self) {
            return ::dragon::dtypes::to_string(self->meta());
          })

      /*! \brief Return the device information */
      .def_property_readonly(
          "device",
          [](Tensor* self) {
            if (self->has_memory()) {
              auto info = self->memory()->info();
              return std::tuple<string, int>(
                  info["device_type"], atoi(info["device_id"].c_str()));
            } else {
              return std::tuple<string, int>("unknown", 0);
            }
          })

      /*! \brief Return the raw const data pointer */
      .def(
          "data",
          [](Tensor* self, const string& device_type) {
            intptr_t pointer = 0;
            if (device_type == "cpu") {
              pointer = (intptr_t)self->raw_data<CPUContext>();
            }
#ifdef USE_CUDA
            else if (device_type == "cuda") {
              pointer = (intptr_t)self->raw_data<CUDAContext>();
            }
#endif
            else {
              LOG(FATAL) << "Unsupported device type: " << device_type;
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
            }
#ifdef USE_CUDA
            else if (device_type == "cuda") {
              pointer = (intptr_t)self->raw_mutable_data<CUDAContext>();
            }
#endif
            else {
              LOG(FATAL) << "Unsupported device type: " << device_type;
            }
            return pointer;
          })

      /*! \brief Reset to an empty tensor */
      .def("Reset", [](Tensor* self) { self->Reset(); })

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
#ifdef USE_CUDA
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
#else
              CUDA_NOT_COMPILED;
#endif
            } else {
              if (src_opt.device_type() == PROTO_CUDA) {
#ifdef USE_CUDA
                // CPU <- CUDA
                CUDADeviceGuard guard(src_opt.device_id());
                CUDAContext::Memcpy<CPUContext, CUDAContext>(
                    other->nbytes(),
                    self->raw_mutable_data<CPUContext>(),
                    other->raw_data<CUDAContext>(),
                    src_opt.device_id());
#else
              CUDA_NOT_COMPILED;
#endif
              } else {
                // CPU <- CPU
                CPUContext::Memcpy<CPUContext, CPUContext>(
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
            const auto& meta = ::dragon::dtypes::to_meta(dtype);
            CHECK(meta.id() != 0)
                << "\nUnsupported tensor type: " + dtype + ".";
            self->set_meta(meta)->Reshape(dims);
            self->raw_mutable_data<CPUContext>();
            return self;
          },
          py::return_value_policy::reference)

      /*! \brief Construct from an external pointer */
      .def(
          "FromPointer",
          [](Tensor* self,
             const vector<int64_t>& dims,
             const string& dtype,
             const string& dev,
             const intptr_t pointer) {
            const auto& meta = ::dragon::dtypes::to_meta(dtype);
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
              default:
                LOG(FATAL) << "Unsupported pointer device: "
                           << opt.device_type();
            }
            self->set_memory(memory);
            if (self->ExternalDeleter) self->ExternalDeleter();
            return self;
          },
          py::return_value_policy::reference)

      /*! \brief Construct from a numpy array */
      .def(
          "FromNumpy",
          [](Tensor* self, py::object object, bool copy) {
            return NumpyWrapper(self).From(object, copy);
          },
          py::return_value_policy::reference,
          py::arg("array"),
          py::arg("copy") = false)

      /*! \brief Construct from a dlpack tensor */
      .def(
          "FromDLPack",
          [](Tensor* self, py::object object) {
            return DLPackWrapper(self).From(object);
          },
          py::return_value_policy::reference)

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

      /*! \brief Switch memory to the cuda context */
      .def(
          "ToMPS",
          [](Tensor* self, int device_id) {
#ifdef USE_MPS
            CHECK(self->has_memory())
                << "\nTensor(" << self->name() << ") "
                << "does not initialize or had been reset.";
            self->memory()->SwitchToMPSDevice(device_id);
#else
       MPS_NOT_COMPILED;
#endif
          })

      /*! \brief Convert tensor into a numpy array */
      .def(
          "ToNumpy",
          [](Tensor* self, bool copy) { return NumpyWrapper(self).To(copy); },
          py::arg("copy") = false)

      /*! \brief Convert tensor into a dlpack tensor */
      .def("ToDLPack", [](Tensor* self, const string& device_str) {
        CHECK_GT(self->count(), 0) << "\nConvert an empty tensor.";
        DeviceOption device;
        device.ParseFromString(device_str);
        return DLPackWrapper(self).To(device);
      });
}

} // namespace python

} // namespace dragon

#endif // DRAGON_MODULES_PYTHON_TENSOR_H_
