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

#ifndef DRAGON_MODULES_PYTHON_DLPACK_H_
#define DRAGON_MODULES_PYTHON_DLPACK_H_

#include <dragon/core/tensor.h>

#include "dragon/modules/python/common.h"
#include "dragon/modules/python/types.h"

namespace dragon {

namespace python {

class DLPackWrapper {
 public:
  DLPackWrapper(Tensor* tensor) : tensor_(tensor) {}

  py::object To(const DeviceOption& opt, bool readonly = false) {
    void* data = nullptr;
    auto* memory = tensor_->memory(true);
    auto* dtype_ptr = dtypes::to_dlpack(tensor_->meta());
    CHECK(dtype_ptr) << "\nUnsupported "
                     << ::dragon::dtypes::to_string(tensor_->meta())
                     << " tensor to dlpack.";
    DLContext ctx;
    auto nbytes = tensor_->nbytes();
    switch (opt.device_type()) {
      case PROTO_CPU: {
        if (readonly) {
          data = const_cast<void*>(memory->cpu_data(nbytes));
        } else {
          data = memory->mutable_cpu_data(nbytes);
        }
        ctx.device_id = 0;
        ctx.device_type = DLDeviceType::kDLCPU;
        break;
      }
      case PROTO_CUDA: {
        if (readonly) {
          data = const_cast<void*>(memory->cuda_data(nbytes));
        } else {
          data = memory->mutable_cuda_data(nbytes);
        }
        ctx.device_id = memory->device();
        ctx.device_type = DLDeviceType::kDLGPU;
        break;
      }
      case PROTO_MPS:
      default:
        LOG(FATAL) << "Unsupported dlpack device.";
    }
    auto* managed_tensor = new DLManagedTensor;
    managed_tensor->dl_tensor.data = data;
    managed_tensor->dl_tensor.ctx = ctx;
    managed_tensor->dl_tensor.ndim = tensor_->ndim();
    managed_tensor->dl_tensor.dtype = *dtype_ptr;
    managed_tensor->dl_tensor.shape =
        const_cast<int64_t*>(tensor_->dims().data());
    managed_tensor->dl_tensor.strides = nullptr;
    managed_tensor->dl_tensor.byte_offset = 0;
    managed_tensor->manager_ctx = nullptr;
    managed_tensor->deleter = [](DLManagedTensor*) {};
    return py::reinterpret_steal<py::object>(
        PyCapsule_New(managed_tensor, "dltensor", nullptr));
  }

  Tensor* From(py::object obj) {
    CHECK(PyCapsule_CheckExact(obj.ptr())) << "\nExpected DLPack capsule.";
    auto* managed_tensor =
        (DLManagedTensor*)PyCapsule_GetPointer(obj.ptr(), "dltensor");
    CHECK(managed_tensor) << "\nInvalid DLPack capsule";
    auto* tensor = &managed_tensor->dl_tensor;
    const auto& meta = dtypes::from_dlpack(tensor->dtype);
    if (meta.id() == 0) {
      LOG(FATAL) << "Unsupported DLDataType: "
                 << "code = " << tensor->dtype.code
                 << ", bits = " << tensor->dtype.bits
                 << ", lanes = " << tensor->dtype.lanes;
    }
    vec64_t dims;
    for (int i = 0; i < tensor->ndim; ++i)
      dims.push_back(tensor->shape[i]);
    if (tensor->strides) {
      int64_t stride = 1;
      for (int i = (int)dims.size() - 1; i >= 0; --i) {
        CHECK_EQ(stride, tensor->strides[i])
            << "\nNon-contigous storage is not supported.";
        stride *= dims[i];
      }
    }
    auto device_id = tensor->ctx.device_id;
    auto* data = (void*)(((int8_t*)tensor->data) + tensor->byte_offset);
    tensor_->Reset(); // Reduce the risk taking dangling pointer
    tensor_->set_meta(meta)->Reshape(dims);
    auto nbytes = tensor_->nbytes();
    auto* memory = new UnifiedMemory();
    switch (tensor->ctx.device_type) {
      case DLDeviceType::kDLCPU:
      case DLDeviceType::kDLCPUPinned:
        memory->set_cpu_data(data, nbytes);
        break;
      case DLDeviceType::kDLGPU:
        memory->set_cuda_data(data, nbytes, device_id);
        break;
      case DLDeviceType::kDLOpenCL:
      case DLDeviceType::kDLVulkan:
      case DLDeviceType::kDLVPI:
      case DLDeviceType::kDLROCM:
      case DLDeviceType::kDLExtDev:
      default:
        LOG(FATAL) << "Unsupported dlpack device.";
    }
    tensor_->set_memory(memory);
    if (tensor_->ExternalDeleter) tensor_->ExternalDeleter();
    /*!
     * Currently not to invoke the deleter.
     * As the GC managing the orginal tensor is beyond controll.
     */
    return tensor_;
  }

 private:
  Tensor* tensor_;
};

} // namespace python

} // namespace dragon

#endif // DRAGON_MODULES_PYTHON_DLPACK_H_
