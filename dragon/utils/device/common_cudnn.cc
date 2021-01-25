#ifdef USE_CUDNN

#include "dragon/utils/device/common_cudnn.h"
#include "dragon/core/tensor.h"
#include "dragon/core/types.h"

namespace dragon {

float CuDNNType<float>::oneval = 1.f;
float CuDNNType<float>::zeroval = 0.f;
const void* CuDNNType<float>::one =
    static_cast<void*>(&CuDNNType<float>::oneval);
const void* CuDNNType<float>::zero =
    static_cast<void*>(&CuDNNType<float>::zeroval);

double CuDNNType<double>::oneval = 1.0;
double CuDNNType<double>::zeroval = 0.0;
const void* CuDNNType<double>::one =
    static_cast<void*>(&CuDNNType<double>::oneval);
const void* CuDNNType<double>::zero =
    static_cast<void*>(&CuDNNType<double>::zeroval);

float CuDNNType<float16>::oneval = 1.f;
float CuDNNType<float16>::zeroval = 0.f;
const void* CuDNNType<float16>::one =
    static_cast<void*>(&CuDNNType<float16>::oneval);
const void* CuDNNType<float16>::zero =
    static_cast<void*>(&CuDNNType<float16>::zeroval);

void CuDNNCreateTensorDesc(cudnnTensorDescriptor_t* desc) {
  CUDNN_CHECK(cudnnCreateTensorDescriptor(desc));
}

void CuDNNDestroyTensorDesc(cudnnTensorDescriptor_t* desc) {
  CUDNN_CHECK(cudnnDestroyTensorDescriptor(*desc));
}

template <typename T>
void CuDNNSetTensorDesc(
    cudnnTensorDescriptor_t* desc,
    const vec64_t& dims,
    const vec64_t& strides) {
  CHECK_EQ(dims.size(), strides.size());
  CHECK(dims.size() >= 3 && dims.size() <= 8);
  vec32_t dimA(dims.begin(), dims.end());
  vec32_t strideA(strides.begin(), strides.end());
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(
      *desc,
      CuDNNType<T>::type,
      (int)dimA.size(),
      dimA.data(),
      strideA.data()));
}

template <typename T>
void CuDNNSetTensorDesc(cudnnTensorDescriptor_t* desc, const vec64_t& dims) {
  // CuDNN requires ndimensions from 3 to 8
  // Expand or Squeeze dimensions to pass the check
  vec64_t dummy_dims(dims);
  if (dims.size() < 3) {
    dummy_dims.resize(3, 1);
  } else if (dims.size() > 8) {
    auto size = std::accumulate(
        dims.data(), dims.data() + dims.size(), 1, std::multiplies<int64_t>());
    dummy_dims = {size, 1, 1};
  }
  int ndim = (int)dummy_dims.size();
  int* dimA = new int[ndim];
  int* strideA = new int[ndim];
  int64_t stride = 1;
  for (int i = ndim - 1; i >= 0; i--) {
    strideA[i] = (int)stride;
    dimA[i] = (int)dummy_dims[i];
    stride *= dimA[i];
  }
  CUDNN_CHECK(cudnnSetTensorNdDescriptor(
      *desc, CuDNNType<T>::type, ndim, dimA, strideA));
  delete[] dimA;
  delete[] strideA;
}

template <typename T>
void CuDNNSetTensorDesc(
    cudnnTensorDescriptor_t* desc,
    const vec64_t& dims,
    const string& data_format) {
  const int N = dims[0];
  const int C = data_format == "NCHW" ? dims[1] : dims.back();
  int D, H, W;
  if (dims.size() == 3) {
    D = W = 1;
    if (data_format == "NCHW") {
      H = dims[2]; // NCH -> NCH1
      CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
          *desc, CuDNNType<T>::type, N, C, H, W, C * H * W, H * W, W, 1));
    } else if (data_format == "NHWC") {
      H = dims[1]; // NHC -> NH1C
      CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
          *desc, CuDNNType<T>::type, N, C, H, W, H * W * C, 1, W * C, C));
    }
  } else if (dims.size() == 4) {
    D = 1;
    if (data_format == "NCHW") {
      H = dims[2], W = dims[3];
      CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
          *desc, CuDNNType<T>::type, N, C, H, W, C * H * W, H * W, W, 1));
    } else if (data_format == "NHWC") {
      H = dims[1], W = dims[2];
      CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(
          *desc, CuDNNType<T>::type, N, C, H, W, H * W * C, 1, W * C, C));
    }
  } else if (dims.size() == 5) {
    vector<int> dims32, strides32;
    if (data_format == "NCHW") {
      D = dims[2], H = dims[3], W = dims[4];
      dims32 = {N, C, D, H, W};
      strides32 = {C * D * H * W, D * H * W, H * W, W, 1};
    } else if (data_format == "NHWC") {
      D = dims[1], H = dims[2], W = dims[3];
      dims32 = {N, C, D, H, W};
      strides32 = {D * H * W * C, 1, H * W * C, W * C, C};
    }
    CUDNN_CHECK(cudnnSetTensorNdDescriptor(
        *desc,
        CuDNNType<T>::type,
        (int)dims32.size(),
        dims32.data(),
        strides32.data()));
  } else {
    LOG(FATAL) << "Excepted 3d/4d/5d tensor, got " << dims.size() << "d.";
  }
}

template <typename T>
void CuDNNSetBiasDesc(
    cudnnTensorDescriptor_t* desc,
    const int num_dims,
    const int64_t num_elements,
    const std::string& data_format) {
  vec64_t dummy_dims((size_t)num_dims - 1, 1);
  if (data_format == "NCHW") {
    dummy_dims.insert(dummy_dims.begin() + 1, num_elements);
  } else {
    dummy_dims.insert(dummy_dims.end(), num_elements);
  }
  CuDNNSetTensorDesc<T>(desc, dummy_dims, data_format);
}

#define INSTANTIATE_API(T)                                       \
  template void CuDNNSetTensorDesc<T>(                           \
      cudnnTensorDescriptor_t*, const vec64_t&);                 \
  template void CuDNNSetTensorDesc<T>(                           \
      cudnnTensorDescriptor_t*, const vec64_t&, const vec64_t&); \
  template void CuDNNSetTensorDesc<T>(                           \
      cudnnTensorDescriptor_t*, const vec64_t&, const string&);  \
  template void CuDNNSetBiasDesc<T>(                             \
      cudnnTensorDescriptor_t*, const int, const int64_t, const string&);

INSTANTIATE_API(float16);
INSTANTIATE_API(float);
INSTANTIATE_API(double);
#undef INSTANTIATE_API

} // namespace dragon

#endif // USE_CUDNN
