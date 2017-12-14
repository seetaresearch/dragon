#ifdef WITH_CUDNN

#include "core/types.h"
#include "core/tensor.h"
#include "utils/cudnn_device.h"

namespace dragon {

float CUDNNType<float>::oneval = 1.0;
float CUDNNType<float>::zeroval = 0.0;
const void* CUDNNType<float>::one =
static_cast<void *>(&CUDNNType<float>::oneval);
const void* CUDNNType<float>::zero =
static_cast<void *>(&CUDNNType<float>::zeroval);

double CUDNNType<double>::oneval = 1.0;
double CUDNNType<double>::zeroval = 0.0;
const void* CUDNNType<double>::one =
static_cast<void *>(&CUDNNType<double>::oneval);
const void* CUDNNType<double>::zero =
static_cast<void *>(&CUDNNType<double>::zeroval);

#ifdef WITH_CUDA_FP16

float CUDNNType<float16>::oneval = 1.0;
float CUDNNType<float16>::zeroval = 0.0;
const void* CUDNNType<float16>::one =
static_cast<void*>(&CUDNNType<float16>::oneval);
const void* CUDNNType<float16>::zero =
static_cast<void*>(&CUDNNType<float16>::zeroval);

#endif

template <typename T>
void cudnnSetTensorDesc(cudnnTensorDescriptor_t* desc, const vector<TIndex>& dims) {
    int ndim = (int)dims.size();
    int* dimA = new int[ndim];
    int* strideA = new int[ndim];
    TIndex stride = 1;
    for (int i = ndim - 1; i >= 0; i--) {
        strideA[i] = stride;
        dimA[i] = dims[i];
        stride *= dimA[i];
    }
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(*desc, CUDNNType<T>::type, ndim, dimA, strideA));
    delete[] dimA;
    delete[] strideA;
}

template <typename T>
void cudnnSetTensor4dDesc(cudnnTensorDescriptor_t* desc,
                          const string& data_format,
                          const vector<TIndex>& dims) {
    if (data_format == "NCHW") {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(*desc, CUDNN_TENSOR_NCHW,
                                                     CUDNNType<T>::type,
                                                                dims[0],
                                                                dims[1],
                                                                dims[2],
                                                              dims[3]));
    } else if (data_format == "NHWC") {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(*desc, CUDNN_TENSOR_NHWC,
                                                     CUDNNType<T>::type,
                                                                dims[0],
                                                                dims[3],
                                                                dims[1],
                                                              dims[2]));
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template <typename T>
void cudnnSetTensor4dDescWithGroup(cudnnTensorDescriptor_t* desc,
                                   const string& data_format,
                                   const vector<TIndex>& dims,
                                   const TIndex group) {
    if (data_format == "NCHW") {
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(*desc, CUDNNType<T>::type,
                                                                   dims[0],
                                                           dims[1] / group,
                                                                   dims[2],
                                                                   dims[3],
                                               dims[1] * dims[2] * dims[3],
                                                         dims[2] * dims[3],
                                                                   dims[3],
                                                                       1));
    } else if (data_format == "NHWC") {
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(*desc, CUDNNType<T>::type,
                                                                   dims[0],
                                                           dims[3] / group,
                                                                   dims[1],
                                                                   dims[2],
                                               dims[1] * dims[2] * dims[3],
                                                                         1,
                                                         dims[2] * dims[3],
                                                                 dims[3]));
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template <typename T>
void cudnnSetTensor5dDesc(cudnnTensorDescriptor_t* desc,
                          const string& data_format,
                          const vector<TIndex>& dims) {
    if (data_format == "NCHW") {
        cudnnSetTensorDesc<T>(desc, dims);
    } else if (data_format == "NHWC") {
        const int N = (int)dims[0];
        const int C = (int)dims[4];
        const int H = (int)dims[1];
        const int W = (int)dims[2];
        const int D = (int)dims[3];
        vector<int> fake_dims = { N, C, H, W, D };
        vector<int> fake_strides = { H * W * D * C, 1, W * D * C, D * C, C };
        CUDNN_CHECK(cudnnSetTensorNdDescriptor(*desc,
                                  CUDNNType<T>::type,
                                                   5,
                                    fake_dims.data(),
                               fake_strides.data()));
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template <typename T>
void cudnnSetTensor3dDesc(cudnnTensorDescriptor_t* desc,
                          const string& data_format,
                          const vector<TIndex>& dims) {
    vector<TIndex> fake_dims = dims;
    if (data_format == "NCHW") {
        //  NCH -> NCHXX
        fake_dims.push_back(1);
        fake_dims.push_back(1);
    } else if (data_format == "NHWC") {
        //  NHC -> NHXXC
        fake_dims.insert(fake_dims.begin() + 2, 1);
        fake_dims.insert(fake_dims.begin() + 2, 1);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
    cudnnSetTensor5dDesc<T>(desc, data_format, fake_dims);
}

template <typename T>
void cudnnSetTensorDesc(cudnnTensorDescriptor_t* desc,
                        const vector<TIndex>& dims,
                        const vector<TIndex>& strides) {
    CHECK_EQ(dims.size(), strides.size());
    CHECK(dims.size() >= 3 && dims.size() <= 8);
    int ndim = (int)dims.size();
    int* dimA = new int[ndim];
    int* strideA = new int[ndim];
    for (int i = ndim - 1; i >= 0; i--) {
        strideA[i] = strides[i];
        dimA[i] = dims[i];
    }
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(*desc, CUDNNType<T>::type, ndim, dimA, strideA));
    delete[] dimA;
    delete[] strideA;
}

template <typename T>
void cudnnSetTensorDesc(cudnnTensorDescriptor_t* desc, Tensor* tensor) {
    //  cuDNN requires ndim from 3 to 8
    //  we fake a reshaped dims to pass check
    vector<TIndex> fake_dims(tensor->dims());
    if (fake_dims.size() < 3 || fake_dims.size() > 8) {
        fake_dims.assign({ 1, 1 });
        fake_dims.push_back(tensor->count());
    }
    cudnnSetTensorDesc<T>(desc, fake_dims);
}

template <typename T>
void cudnnSetTensor4dDesc(cudnnTensorDescriptor_t* desc, const string& data_format, Tensor* tensor) {
    CHECK_EQ((int)tensor->ndim(), 4)
        << "\nThe num of dimensions of Tensor(" << tensor->name() << ") "
        << "should be 4, but got " << tensor->ndim() << ".";
    cudnnSetTensor4dDesc<T>(desc, data_format, tensor->dims());
}

template <typename T>
void cudnnSetTensor5dDesc(cudnnTensorDescriptor_t* desc, const string& data_format, Tensor* tensor) {
    CHECK_EQ((int)tensor->ndim(), 5)
        << "\nThe num of dimensions of Tensor(" << tensor->name() << ") "
        << "should be 5, but got " << tensor->ndim() << ".";
    cudnnSetTensor5dDesc<T>(desc, data_format, tensor->dims());
}

template <typename T>
void cudnnSetTensor3dDesc(cudnnTensorDescriptor_t* desc, const string& data_format, Tensor* tensor) {
    CHECK_EQ((int)tensor->ndim(), 3)
        << "\nThe num of dimensions of Tensor(" << tensor->name() << ") "
        << "should be 3, but got " << tensor->ndim() << ".";
    cudnnSetTensor3dDesc<T>(desc, data_format, tensor->dims());
}

template void cudnnSetTensorDesc<float>(cudnnTensorDescriptor_t*, Tensor*);
template void cudnnSetTensor4dDesc<float>(cudnnTensorDescriptor_t*, const string&, Tensor*);
template void cudnnSetTensor5dDesc<float>(cudnnTensorDescriptor_t*, const string&, Tensor*);
template void cudnnSetTensor3dDesc<float>(cudnnTensorDescriptor_t*, const string&, Tensor*);
template void cudnnSetTensorDesc<float>(cudnnTensorDescriptor_t*, const vector<TIndex>&);
template void cudnnSetTensor4dDesc<float>(cudnnTensorDescriptor_t*, const string&, const vector<TIndex>&);
template void cudnnSetTensor5dDesc<float>(cudnnTensorDescriptor_t*, const string&, const vector<TIndex>&);
template void cudnnSetTensor3dDesc<float>(cudnnTensorDescriptor_t*, const string&, const vector<TIndex>&);
template void cudnnSetTensor4dDescWithGroup<float>(cudnnTensorDescriptor_t*, const string&, const vector<TIndex>&, const TIndex);
template void cudnnSetTensorDesc<float>(cudnnTensorDescriptor_t*, const vector<TIndex>&, const vector<TIndex>&);


template void cudnnSetTensorDesc<double>(cudnnTensorDescriptor_t*, Tensor*);
template void cudnnSetTensor4dDesc<double>(cudnnTensorDescriptor_t*, const string&, Tensor*);
template void cudnnSetTensor5dDesc<double>(cudnnTensorDescriptor_t*, const string&, Tensor*);
template void cudnnSetTensor3dDesc<double>(cudnnTensorDescriptor_t*, const string&, Tensor*);
template void cudnnSetTensorDesc<double>(cudnnTensorDescriptor_t*, const vector<TIndex>&);
template void cudnnSetTensor4dDesc<double>(cudnnTensorDescriptor_t*, const string&, const vector<TIndex>&);
template void cudnnSetTensor5dDesc<double>(cudnnTensorDescriptor_t*, const string&, const vector<TIndex>&);
template void cudnnSetTensor3dDesc<double>(cudnnTensorDescriptor_t*, const string&, const vector<TIndex>&);
template void cudnnSetTensor4dDescWithGroup<double>(cudnnTensorDescriptor_t*, const string&, const vector<TIndex>&, const TIndex);
template void cudnnSetTensorDesc<double>(cudnnTensorDescriptor_t*, const vector<TIndex>&, const vector<TIndex>&);


#ifdef WITH_CUDA_FP16
template void cudnnSetTensorDesc<float16>(cudnnTensorDescriptor_t*, Tensor*);
template void cudnnSetTensor4dDesc<float16>(cudnnTensorDescriptor_t*, const string&, Tensor*);
template void cudnnSetTensor5dDesc<float16>(cudnnTensorDescriptor_t*, const string&, Tensor*);
template void cudnnSetTensor3dDesc<float16>(cudnnTensorDescriptor_t*, const string&, Tensor*);
template void cudnnSetTensorDesc<float16>(cudnnTensorDescriptor_t*, const vector<TIndex>&);
template void cudnnSetTensor4dDesc<float16>(cudnnTensorDescriptor_t*, const string&, const vector<TIndex>&);
template void cudnnSetTensor5dDesc<float16>(cudnnTensorDescriptor_t*, const string&, const vector<TIndex>&);
template void cudnnSetTensor3dDesc<float16>(cudnnTensorDescriptor_t*, const string&, const vector<TIndex>&);
template void cudnnSetTensor4dDescWithGroup<float16>(cudnnTensorDescriptor_t*, const string&, const vector<TIndex>&, const TIndex);
template void cudnnSetTensorDesc<float16>(cudnnTensorDescriptor_t*, const vector<TIndex>&, const vector<TIndex>&);
#endif

}    // namespace dragon

#endif    // WITH_CUDNN
