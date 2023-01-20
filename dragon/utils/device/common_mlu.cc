#ifdef USE_MLU

#include "dragon/utils/device/common_mlu.h"

namespace dragon {

float CNNLType<float>::oneval = 1.f;
float CNNLType<float>::zeroval = 0.f;
const void* CNNLType<float>::one = static_cast<void*>(&CNNLType<float>::oneval);
const void* CNNLType<float>::zero =
    static_cast<void*>(&CNNLType<float>::zeroval);

double CNNLType<double>::oneval = 1.0;
double CNNLType<double>::zeroval = 0.0;
const void* CNNLType<double>::one =
    static_cast<void*>(&CNNLType<double>::oneval);
const void* CNNLType<double>::zero =
    static_cast<void*>(&CNNLType<double>::zeroval);

float CNNLType<float16>::oneval = 1.f;
float CNNLType<float16>::zeroval = 0.f;
const void* CNNLType<float16>::one =
    static_cast<void*>(&CNNLType<float16>::oneval);
const void* CNNLType<float16>::zero =
    static_cast<void*>(&CNNLType<float16>::zeroval);

const cnnlDataType_t& CNNLGetDataType(const TypeMeta& type) {
  static cnnlDataType_t unknown_type = CNNL_DTYPE_INVALID;
  static std::unordered_map<TypeId, cnnlDataType_t> m{
      {TypeMeta::Id<bool>(), CNNL_DTYPE_BOOL},
      {TypeMeta::Id<uint8_t>(), CNNL_DTYPE_UINT8},
      {TypeMeta::Id<int8_t>(), CNNL_DTYPE_INT8},
      {TypeMeta::Id<short>(), CNNL_DTYPE_INT16},
      {TypeMeta::Id<int>(), CNNL_DTYPE_INT32},
      {TypeMeta::Id<int64_t>(), CNNL_DTYPE_INT64},
      {TypeMeta::Id<float16>(), CNNL_DTYPE_HALF},
      {TypeMeta::Id<float>(), CNNL_DTYPE_FLOAT},
      {TypeMeta::Id<double>(), CNNL_DTYPE_DOUBLE},
  };
  auto it = m.find(type.id());
  return it != m.end() ? it->second : unknown_type;
}

void CNNLCreateTensorDesc(cnnlTensorDescriptor_t* desc) {
  CNNL_CHECK(cnnlCreateTensorDescriptor(desc));
}

void CNNLDestroyTensorDesc(cnnlTensorDescriptor_t desc) {
  CNNL_CHECK(cnnlDestroyTensorDescriptor(desc));
}

template <typename T>
void CNNLSetTensorDesc(cnnlTensorDescriptor_t desc, const vec64_t& dims) {
  CHECK(dims.size() >= 1 && dims.size() <= 8);
  CNNL_CHECK(cnnlSetTensorDescriptor(
      desc,
      CNNL_LAYOUT_ARRAY,
      CNNLGetDataType<T>(),
      int(dims.size()),
      vec32_t({dims.begin(), dims.end()}).data()));
}

template <typename T>
void CNNLSetTensorDesc(
    cnnlTensorDescriptor_t desc,
    const vec64_t& dims,
    const vec64_t& strides) {
  CHECK_EQ(dims.size(), strides.size());
  CHECK(dims.size() >= 1 && dims.size() <= 8);
  CNNL_CHECK(cnnlSetTensorDescriptorEx(
      desc,
      CNNL_LAYOUT_ARRAY,
      CNNLGetDataType<T>(),
      int(dims.size()),
      vec32_t({dims.begin(), dims.end()}).data(),
      vec32_t({strides.begin(), strides.end()}).data()));
}

template <typename T>
void CNNLSetTensorDesc(
    cnnlTensorDescriptor_t desc,
    const vec64_t& dims,
    const string& data_format) {
  if (dims.empty()) return;
  CHECK(dims.size() <= 8);
  auto layout = CNNL_LAYOUT_ARRAY;
  auto dims_v2 = dims;
  if (dims.size() == 1) {
    dims_v2 = vec64_t(4, 1);
    if (data_format == "NCHW") {
      // C -> 1C11
      dims_v2[1] = dims.back();
    } else if (data_format == "NHWC") {
      // C -> 111C
      dims_v2[3] = dims.back();
    }
  } else if (dims.size() == 2) {
    layout = CNNL_LAYOUT_NC;
  } else if (dims.size() == 3) {
    if (data_format == "NCHW") {
      // NCH -> NCH1
      dims_v2.push_back(1);
      layout = CNNL_LAYOUT_NCHW;
    } else if (data_format == "NHWC") {
      // NHC -> NH1C
      dims_v2.insert(dims_v2.begin() + 2, 1);
      layout = CNNL_LAYOUT_NHWC;
    } else if (data_format == "NLC") {
      layout = CNNL_LAYOUT_NLC;
    }
  } else if (dims.size() == 4) {
    if (data_format == "NCHW") {
      layout = CNNL_LAYOUT_NCHW;
    } else if (data_format == "NHWC") {
      layout = CNNL_LAYOUT_NHWC;
    }
  } else if (dims.size() == 5) {
    if (data_format == "NCHW") {
      layout = CNNL_LAYOUT_NCDHW;
    } else if (data_format == "NHWC") {
      layout = CNNL_LAYOUT_NDHWC;
    }
  } else {
    LOG(FATAL) << "Excepted tensor dimensions <=5, got " << dims_v2.size();
  }
  CNNL_CHECK(cnnlSetTensorDescriptor(
      desc,
      layout,
      CNNLGetDataType<T>(),
      int(dims_v2.size()),
      vec32_t({dims_v2.begin(), dims_v2.end()}).data()));
}

#define INSTANTIATE_API(T)                                                    \
  template void CNNLSetTensorDesc<T>(cnnlTensorDescriptor_t, const vec64_t&); \
  template void CNNLSetTensorDesc<T>(                                         \
      cnnlTensorDescriptor_t, const vec64_t&, const vec64_t&);                \
  template void CNNLSetTensorDesc<T>(                                         \
      cnnlTensorDescriptor_t, const vec64_t&, const string&);

INSTANTIATE_API(bool);
INSTANTIATE_API(uint8_t);
INSTANTIATE_API(int8_t);
INSTANTIATE_API(short);
INSTANTIATE_API(int);
INSTANTIATE_API(int64_t);
INSTANTIATE_API(float16);
INSTANTIATE_API(float);
INSTANTIATE_API(double);
#undef INSTANTIATE_API

} // namespace dragon

#endif // USE_MLU
