#ifdef WITH_CUDNN

#include "core/types.h"
#include "core/tensor.h"
#include "utils/cudnn_device.h"

namespace dragon {

float CuDNNType<float>::oneval = 1.f;
float CuDNNType<float>::zeroval = 0.f;
const void* CuDNNType<float>::one =
static_cast<void *>(&CuDNNType<float>::oneval);
const void* CuDNNType<float>::zero =
static_cast<void *>(&CuDNNType<float>::zeroval);

double CuDNNType<double>::oneval = 1.0;
double CuDNNType<double>::zeroval = 0.0;
const void* CuDNNType<double>::one =
static_cast<void *>(&CuDNNType<double>::oneval);
const void* CuDNNType<double>::zero =
static_cast<void *>(&CuDNNType<double>::zeroval);

float CuDNNType<float16>::oneval = 1.f;
float CuDNNType<float16>::zeroval = 0.f;
const void* CuDNNType<float16>::one =
static_cast<void*>(&CuDNNType<float16>::oneval);
const void* CuDNNType<float16>::zero =
static_cast<void*>(&CuDNNType<float16>::zeroval);

template <typename T>
void CuDNNSetTensorDesc(
    cudnnTensorDescriptor_t*        desc,
    const vec64_t&                  dims) {
    int ndim = (int)dims.size();
    int* dimA = new int[ndim];
    int* strideA = new int[ndim];
    int64_t stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strideA[i] = (int)stride;
        dimA[i] = (int)dims[i];
        stride *= dimA[i];
    }
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(
        *desc,
        CuDNNType<T>::type,
        ndim,
        dimA,
        strideA
    ));
    delete[] dimA;
    delete[] strideA;
}

template <typename T>
void CuDNNSetTensor4dDesc(
    cudnnTensorDescriptor_t*        desc,
    const string&                   data_format,
    const vec64_t&                  dims) {
    if (data_format == "NCHW") {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            *desc,
            CUDNN_TENSOR_NCHW,
            CuDNNType<T>::type,
            dims[0],
            dims[1],
            dims[2],
            dims[3]
        ));
    } else if (data_format == "NHWC") {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            *desc,
            CUDNN_TENSOR_NHWC,
            CuDNNType<T>::type,
            dims[0],
            dims[3],
            dims[1],
            dims[2]
        ));
    }
}

template <typename T>
void CuDNNSetTensor4dDescWithGroup(
    cudnnTensorDescriptor_t*        desc,
    const string&                   data_format,
    const vec64_t&                  dims,
    const int64_t                   group) {
    if (data_format == "NCHW") {
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
            *desc,
            CuDNNType<T>::type,
            dims[0],
            dims[1] / group,
            dims[2],
            dims[3],
            dims[1] * dims[2] * dims[3],
            dims[2] * dims[3],
            dims[3],
            1
        ));
    } else if (data_format == "NHWC") {
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
            *desc,
            CuDNNType<T>::type,
            dims[0],
            dims[3] / group,
            dims[1],
            dims[2],
            dims[1] * dims[2] * dims[3],
            1,
            dims[2] * dims[3], dims[3]
        ));
    }
}

template <typename T>
void CuDNNSetTensor5dDesc(
    cudnnTensorDescriptor_t*        desc,
    const string&                   data_format,
    const vec64_t&                  dims) {
    if (data_format == "NCHW") {
        CuDNNSetTensorDesc<T>(desc, dims);
    } else if (data_format == "NHWC") {
        const int N = (int)dims[0];
        const int C = (int)dims[4];
        const int H = (int)dims[1];
        const int W = (int)dims[2];
        const int D = (int)dims[3];
        vec32_t fake_dims = { N, C, H, W, D };
        vec32_t fake_strides = {
            H * W * D * C, 1,
                W * D * C,
                    D * C,
                        C };
        CUDNN_CHECK(cudnnSetTensorNdDescriptor(
            *desc,
            CuDNNType<T>::type,
            5,
            fake_dims.data(),
            fake_strides.data()
        ));
    }
}

template <typename T>
void CuDNNSetTensor3dDesc(
    cudnnTensorDescriptor_t*        desc,
    const string&                   data_format,
    const vec64_t&                  dims) {
    vec64_t fake_dims = dims;
    if (data_format == "NCHW") {
        // NCH -> NCHXX
        fake_dims.push_back(1);
        fake_dims.push_back(1);
    } else if (data_format == "NHWC") {
        // NHC -> NHXXC
        fake_dims.insert(fake_dims.begin() + 2, 1);
        fake_dims.insert(fake_dims.begin() + 2, 1);
    }
    CuDNNSetTensor5dDesc<T>(desc, data_format, fake_dims);
}

template <typename T>
void CuDNNSetTensorDesc(
    cudnnTensorDescriptor_t*        desc,
    const vec64_t&                  dims,
    const vec64_t&                  strides) {
    CHECK_EQ(dims.size(), strides.size());
    CHECK(dims.size() >= 3 && dims.size() <= 8);
    vec32_t dimA(dims.begin(), dims.end());
    vec32_t strideA(strides.begin(), strides.end());
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(
        *desc,
        CuDNNType<T>::type,
        (int)dimA.size(),
        dimA.data(),
        strideA.data()
    ));
}

template <typename T>
void CuDNNSetTensorDesc(
    cudnnTensorDescriptor_t*        desc,
    Tensor*                         tensor) {
    // CuDNN requires ndimensions from 3 to 8
    // Exapnd or Squeeze dimensions to pass check
    vec64_t fake_dims(tensor->dims());
    if (fake_dims.size() < 3) {
        fake_dims.resize(3, 1);
    } else if (fake_dims.size() > 8) {
        fake_dims = { tensor->count(), 1, 1 };
    } CuDNNSetTensorDesc<T>(desc, fake_dims);
}

template <typename T>
void CuDNNSetTensor4dDesc(
    cudnnTensorDescriptor_t*        desc,
    const string&                   data_format,
    Tensor*                         tensor) {
    CHECK_EQ(tensor->ndim(), 4)
        << "\nThe num of dimensions of Tensor("
        << tensor->name() << ") "
        << "should be 4, but got " << tensor->ndim() << ".";
    CuDNNSetTensor4dDesc<T>(desc, data_format, tensor->dims());
}

template <typename T>
void CuDNNSetTensor5dDesc(
    cudnnTensorDescriptor_t*        desc,
    const string&                   data_format,
    Tensor*                         tensor) {
    CHECK_EQ(tensor->ndim(), 5)
        << "\nThe num of dimensions of Tensor("
        << tensor->name() << ") "
        << "should be 5, but got " << tensor->ndim() << ".";
    CuDNNSetTensor5dDesc<T>(desc, data_format, tensor->dims());
}

template <typename T>
void CuDNNSetTensor3dDesc(
    cudnnTensorDescriptor_t*        desc,
    const string&                   data_format,
    Tensor*                         tensor) {
    CHECK_EQ(tensor->ndim(), 3)
        << "\nThe num of dimensions of Tensor("
        << tensor->name() << ") "
        << "should be 3, but got " << tensor->ndim() << ".";
    CuDNNSetTensor3dDesc<T>(desc, data_format, tensor->dims());
}

template void CuDNNSetTensorDesc<float>(
    cudnnTensorDescriptor_t*,
    Tensor*);

template void CuDNNSetTensor4dDesc<float>(
    cudnnTensorDescriptor_t*,
    const string&,
    Tensor*);

template void CuDNNSetTensor5dDesc<float>(
    cudnnTensorDescriptor_t*,
    const string&,
    Tensor*);

template void CuDNNSetTensor3dDesc<float>(
    cudnnTensorDescriptor_t*,
    const string&,
    Tensor*);

template void CuDNNSetTensorDesc<float>(
    cudnnTensorDescriptor_t*,
    const vec64_t&);

template void CuDNNSetTensor4dDesc<float>(
    cudnnTensorDescriptor_t*,
    const string&,
    const vec64_t&);

template void CuDNNSetTensor5dDesc<float>(
    cudnnTensorDescriptor_t*,
    const string&,
    const vec64_t&);

template void CuDNNSetTensor3dDesc<float>(
    cudnnTensorDescriptor_t*,
    const string&,
    const vec64_t&);

template void CuDNNSetTensor4dDescWithGroup<float>(
    cudnnTensorDescriptor_t*,
    const string&,
    const vec64_t&,
    const int64_t);

template void CuDNNSetTensorDesc<float>(
    cudnnTensorDescriptor_t*,
    const vec64_t&,
    const vec64_t&);

template void CuDNNSetTensorDesc<double>(
    cudnnTensorDescriptor_t*,
    Tensor*);

template void CuDNNSetTensor4dDesc<double>(
    cudnnTensorDescriptor_t*,
    const string&,
    Tensor*);

template void CuDNNSetTensor5dDesc<double>(
    cudnnTensorDescriptor_t*,
    const string&,
    Tensor*);

template void CuDNNSetTensor3dDesc<double>(
    cudnnTensorDescriptor_t*,
    const string&,
    Tensor*);

template void CuDNNSetTensorDesc<double>(
    cudnnTensorDescriptor_t*,
    const vec64_t&);

template void CuDNNSetTensor4dDesc<double>(
    cudnnTensorDescriptor_t*,
    const string&,
    const vec64_t&);

template void CuDNNSetTensor5dDesc<double>(
    cudnnTensorDescriptor_t*,
    const string&,
    const vec64_t&);

template void CuDNNSetTensor3dDesc<double>(
    cudnnTensorDescriptor_t*,
    const string&,
    const vec64_t&);

template void CuDNNSetTensor4dDescWithGroup<double>(
    cudnnTensorDescriptor_t*,
    const string&,
    const vec64_t&,
    const int64_t);

template void CuDNNSetTensorDesc<double>(
    cudnnTensorDescriptor_t*,
    const vec64_t&,
    const vec64_t&);

template void CuDNNSetTensorDesc<float16>(
    cudnnTensorDescriptor_t*,
    Tensor*);

template void CuDNNSetTensor4dDesc<float16>(
    cudnnTensorDescriptor_t*,
    const string&,
    Tensor*);

template void CuDNNSetTensor5dDesc<float16>(
    cudnnTensorDescriptor_t*,
    const string&,
    Tensor*);

template void CuDNNSetTensor3dDesc<float16>(
    cudnnTensorDescriptor_t*,
    const string&,
    Tensor*);

template void CuDNNSetTensorDesc<float16>(
    cudnnTensorDescriptor_t*,
    const vec64_t&);

template void CuDNNSetTensor4dDesc<float16>(
    cudnnTensorDescriptor_t*,
    const string&,
    const vec64_t&);

template void CuDNNSetTensor5dDesc<float16>(
    cudnnTensorDescriptor_t*,
    const string&,
    const vec64_t&);

template void CuDNNSetTensor3dDesc<float16>(
    cudnnTensorDescriptor_t*,
    const string&,
    const vec64_t&);

template void CuDNNSetTensor4dDescWithGroup<float16>(
    cudnnTensorDescriptor_t*,
    const string&,
    const vec64_t&,
    const int64_t);

template void CuDNNSetTensorDesc<float16>(
    cudnnTensorDescriptor_t*,
    const vec64_t&,
    const vec64_t&);

}  // namespace dragon

#endif  // WITH_CUDNN