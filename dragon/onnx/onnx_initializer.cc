#include "dragon/onnx/onnx_backend.h"

namespace dragon {

namespace onnx {

template <class T>
bool TryConvertingTensorRawValues(
    const TensorProto& onnx_tensor,
    google::protobuf::RepeatedField<T>* field) {
  if (!onnx_tensor.has_raw_data()) return false;

  size_t raw_size = onnx_tensor.raw_data().size();
  CHECK_EQ(raw_size % sizeof(T), 0);
  size_t num_elements = raw_size / sizeof(T);

  field->Resize((int)num_elements, 0);
  auto* src = static_cast<const void*>(onnx_tensor.raw_data().data());
  auto* dest = static_cast<void*>(field->mutable_data());
  memcpy(dest, src, raw_size);
  return true;
}

template <class Tx, class Ty>
bool TryConvertingTensorRawValues_v2(
    const TensorProto& onnx_tensor,
    google::protobuf::RepeatedField<Ty>* field) {
  if (!onnx_tensor.has_raw_data()) return false;

  size_t raw_size = onnx_tensor.raw_data().size();
  CHECK_EQ(raw_size % sizeof(Tx), 0);
  size_t num_elements = raw_size / sizeof(Tx);

  field->Resize((int)num_elements, 0);
  auto* src = static_cast<const void*>(onnx_tensor.raw_data().data());
  auto* dest = static_cast<void*>(field->mutable_data());
  memcpy(dest, src, raw_size);
  return true;
}

template <typename T>
void ConvertIntegralValue(const TensorProto& onnx_tensor, Argument* values) {
  google::protobuf::RepeatedField<T> tmp;
  const google::protobuf::RepeatedField<T>* t_src = &tmp;
  bool converted = TryConvertingTensorRawValues<T>(onnx_tensor, &tmp);
  if (converted) {
    for (const auto i : *t_src)
      values->add_ints(i);
  } else {
    const auto* int32_src = &onnx_tensor.int32_data();
    for (const auto i : *int32_src)
      values->add_ints(i);
  }
}

template <>
void ConvertIntegralValue<google::protobuf::int64>(
    const TensorProto& onnx_tensor,
    Argument* values) {
  auto* ints = values->mutable_ints();
  if (!TryConvertingTensorRawValues<google::protobuf::int64>(
          onnx_tensor, ints)) {
    ints->CopyFrom(onnx_tensor.int64_data());
  }
}

template <>
void ConvertIntegralValue<google::protobuf::uint64>(
    const TensorProto& onnx_tensor,
    Argument* values) {
  google::protobuf::RepeatedField<google::protobuf::uint64> tmp;
  const auto* src = &tmp;
  if (!TryConvertingTensorRawValues<google::protobuf::uint64>(
          onnx_tensor, &tmp)) {
    src = &onnx_tensor.uint64_data();
  }
  for (const auto i : *src)
    values->add_ints(i);
}

void ONNXBackend::ONNXTensorToArgument(
    const TensorProto& onnx_tensor,
    Argument* dtype,
    Argument* values) {
  if (onnx_tensor.data_type() == TensorProto::FLOAT16) {
    /*! float16: raw_data = >floats */
    dtype->set_s("float16");
    auto* floats = values->mutable_floats();
    CHECK((TryConvertingTensorRawValues_v2<google::protobuf::uint16, float>(
        onnx_tensor, floats)))
        << "Excepted the raw data to store the FLOAT16.";
  } else if (onnx_tensor.data_type() == TensorProto::FLOAT) {
    /*! float32: float_data | raw_data => floats */
    dtype->set_s("float32");
    auto* floats = values->mutable_floats();
    if (!TryConvertingTensorRawValues<float>(onnx_tensor, floats)) {
      floats->CopyFrom(onnx_tensor.float_data());
    }
  } else if (onnx_tensor.data_type() == TensorProto::DOUBLE) {
    /*! float64: double_data | raw_data => floats */
    dtype->set_s("float64");
    google::protobuf::RepeatedField<double> tmp;
    const auto* src = &tmp;
    if (!TryConvertingTensorRawValues<double>(onnx_tensor, &tmp)) {
      src = &onnx_tensor.double_data();
    }
    for (const auto i : *src)
      values->add_floats(i);
  } else if (onnx_tensor.data_type() == TensorProto::INT64) {
    /*! int64: int64_data | raw_data => ints */
    dtype->set_s("int64");
    ConvertIntegralValue<google::protobuf::int64>(onnx_tensor, values);
  } else if (onnx_tensor.data_type() == TensorProto::UINT64) {
    /*! uint64: uint64_data | raw_data => ints */
    dtype->set_s("uint64");
    ConvertIntegralValue<google::protobuf::uint64>(onnx_tensor, values);
  } else if (onnx_tensor.data_type() == TensorProto::UINT32) {
    /*! uint32: uint64_data | raw_data => ints */
    dtype->set_s("uint32");
    ConvertIntegralValue<google::protobuf::uint64>(onnx_tensor, values);
  } else if (onnx_tensor.data_type() == TensorProto::BOOL) {
    /*! bool: int32_data | raw_data => ints */
    dtype->set_s("bool");
    ConvertIntegralValue<google::protobuf::int8>(onnx_tensor, values);
  } else if (onnx_tensor.data_type() == TensorProto::UINT8) {
    /*! uint8: int32_data | raw_data => ints */
    dtype->set_s("uint8");
    ConvertIntegralValue<google::protobuf::uint8>(onnx_tensor, values);
  } else if (onnx_tensor.data_type() == TensorProto::INT8) {
    /*! int8: int32_data | raw_data => ints */
    dtype->set_s("int8");
    ConvertIntegralValue<google::protobuf::int8>(onnx_tensor, values);
  } else if (onnx_tensor.data_type() == TensorProto::UINT16) {
    /*! uint16: int32_data | raw_data => ints */
    dtype->set_s("uint16");
    ConvertIntegralValue<google::protobuf::uint16>(onnx_tensor, values);
  } else if (onnx_tensor.data_type() == TensorProto::INT16) {
    /*! int16: int32_data | raw_data => ints */
    dtype->set_s("int16");
    ConvertIntegralValue<google::protobuf::int16>(onnx_tensor, values);
  } else if (onnx_tensor.data_type() == TensorProto::INT32) {
    /*! int32: int32_data | raw_data => ints */
    dtype->set_s("int32");
    ConvertIntegralValue<google::protobuf::int32>(onnx_tensor, values);
  } else if (onnx_tensor.data_type() == TensorProto::STRING) {
    /*! string: string_data => strings */
    dtype->set_s("string");
    auto* strings = values->mutable_strings();
    strings->CopyFrom(onnx_tensor.string_data());
  } else {
    LOG(FATAL) << "unrecognized tensor type: ", onnx_tensor.data_type();
  }
}

void ONNXBackend::BuildTensorFillOp(
    const TensorProto& onnx_tensor,
    OperatorDef* op_def) {
  auto fill_name = onnx_tensor.name();
  CHECK(!fill_name.empty());

  if (onnx_tensor.has_segment()) {
    LOG(FATAL) << "Currently not supporting loading segments.";
  }

  op_def->set_type("GivenTensorFill");
  op_def->set_name("Fill -> " + fill_name);
  op_def->add_output(fill_name);

  // Determine the shape
  auto* shape = op_def->add_arg();
  shape->set_name("shape");
  for (const auto d : onnx_tensor.dims())
    shape->add_ints(d);

  // Determine the data type and values
  auto* dtype = op_def->add_arg();
  dtype->set_name("dtype");
  auto* values = op_def->add_arg();
  values->set_name("values");
  ONNXTensorToArgument(onnx_tensor, dtype, values);
}

} // namespace onnx

} // namespace dragon
