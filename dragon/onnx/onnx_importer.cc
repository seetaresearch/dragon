#include "dragon/onnx/onnx_backend.h"
#include "dragon/utils/map_utils.h"

namespace dragon {

namespace onnx {

#define ONNX_FLOATS google::protobuf::RepeatedField<float>
#define ONNX_INTS google::protobuf::RepeatedField<google::protobuf::int64>

bool AlmostEqual(double a, double b) {
  const static double kEps = 1e-15;
  return (fabs(a - b) < kEps);
}

ONNXImporterReturns ONNXBackend::ATenImporter(
    ONNXNode* onnx_node,
    const ConversionContext& ctx) {
  auto node = NodeProto(onnx_node->node);
  auto onnx_node_v2 = ONNXNode(node);
  auto& attributes = onnx_node_v2.attributes;
  auto op_type = attributes.get<string>("op_type", "");
  if (op_type.empty()) {
    LOG(FATAL) << "op_type is required to evolve "
               << "to the specific operator.";
  }
  node.set_op_type(op_type);
  attributes.remove("op_type");
  return GenericImporter(&onnx_node_v2, ctx);
}

ONNXImporterReturns ONNXBackend::BatchNormImporter(
    ONNXNode* onnx_node,
    const ConversionContext& ctx) {
  auto node = NodeProto(onnx_node->node);
  auto onnx_node_v2 = ONNXNode(node);
  auto& attributes = onnx_node_v2.attributes;
  // Enforce to NCHW format
  attributes.AddRewrittenAttribute("axis")->set_i(1);
  // Remove dummy attributes
  attributes.remove("consumed_inputs");
  attributes.remove("is_test");
  attributes.remove("spatial");
  return GenericImporter(&onnx_node_v2, ctx);
}

ONNXImporterReturns ONNXBackend::CastImporter(
    ONNXNode* onnx_node,
    const ConversionContext& ctx) {
  auto& attributes = onnx_node->attributes;
  // Determine the dtype
  auto* dtype = attributes.AddRewrittenAttribute("dtype");
  auto onnx_dtype = attributes.get<int64_t>("to", TensorProto::UNDEFINED);
  auto supported_dtype = true;
  switch (onnx_dtype) {
    case ONNX_NAMESPACE::TensorProto::BOOL:
      dtype->set_s("bool");
      break;
    case ONNX_NAMESPACE::TensorProto::INT8:
      dtype->set_s("int8");
      break;
    case ONNX_NAMESPACE::TensorProto::UINT8:
      dtype->set_s("uint8");
      break;
    case ONNX_NAMESPACE::TensorProto::INT32:
      dtype->set_s("int32");
      break;
    case ONNX_NAMESPACE::TensorProto::INT64:
      dtype->set_s("int64");
      break;
    case ONNX_NAMESPACE::TensorProto::FLOAT16:
      dtype->set_s("float16");
      break;
    case ONNX_NAMESPACE::TensorProto::FLOAT:
      dtype->set_s("float32");
      break;
    case ONNX_NAMESPACE::TensorProto::DOUBLE:
      dtype->set_s("float64");
      break;
    case ONNX_NAMESPACE::TensorProto::INT16:
      dtype->set_s("int16");
      supported_dtype = false;
      break;
    case ONNX_NAMESPACE::TensorProto::UINT16:
      dtype->set_s("uint16");
      supported_dtype = false;
      break;
    case ONNX_NAMESPACE::TensorProto::UINT32:
      dtype->set_s("uint32");
      supported_dtype = false;
      break;
    case ONNX_NAMESPACE::TensorProto::UINT64:
      dtype->set_s("uint64");
      supported_dtype = false;
      break;
    case ONNX_NAMESPACE::TensorProto::STRING:
      dtype->set_s("string");
      supported_dtype = false;
      break;
    case ONNX_NAMESPACE::TensorProto::COMPLEX64:
      dtype->set_s("complex64");
      supported_dtype = false;
      break;
    case ONNX_NAMESPACE::TensorProto::COMPLEX128:
      dtype->set_s("complex128");
      supported_dtype = false;
      break;
    case ONNX_NAMESPACE::TensorProto::UNDEFINED:
      dtype->set_s("undefined");
      supported_dtype = false;
      break;
  };
  CHECK(supported_dtype) << "\nCasting to " << dtype->s()
                         << " is not supported.";
  attributes.remove("to");
  return GenericImporter(onnx_node, ctx);
}

ONNXImporterReturns ONNXBackend::ConvPoolImporter(
    ONNXNode* onnx_node,
    const ConversionContext& ctx) {
  auto& attributes = onnx_node->attributes;
  const auto onnx_op_type = onnx_node->node.op_type();
  // Determine the padding
  auto mode = attributes.get<string>("auto_pad");
  auto* padding = attributes.AddRewrittenAttribute("padding");
  // SAME, SAME_LOWER, or SAME_UPPER
  if (str::find(mode, "SAME")) {
    padding->set_s(mode);
  } else {
    padding->set_s("VALID"); // Use explicit pads
  }
  attributes.remove("auto_pad");
  // Determine the pooling mode
  if (onnx_op_type == "MaxPool") {
    attributes.AddRewrittenAttribute("mode")->set_s("MAX");
  } else if (onnx_op_type == "AveragePool") {
    attributes.AddRewrittenAttribute("mode")->set_s("AVG");
  } else if (onnx_op_type == "GlobalMaxPool") {
    attributes.AddRewrittenAttribute("mode")->set_s("MAX");
    attributes.AddRewrittenAttribute("global_pool")->set_i(1);
  } else if (onnx_op_type == "GlobalAveragePool") {
    attributes.AddRewrittenAttribute("mode")->set_s("AVG");
    attributes.AddRewrittenAttribute("global_pool")->set_i(1);
  }
  return GenericImporter(onnx_node, ctx);
}

ONNXImporterReturns ONNXBackend::GenericImporter(
    ONNXNode* onnx_node,
    const ConversionContext& ctx) {
  ONNXImporterReturns returns;
  auto* op_def = returns.AddOp();
  const auto& node = onnx_node->node;
  op_def->mutable_input()->MergeFrom(node.input());
  op_def->mutable_output()->MergeFrom(node.output());
  op_def->set_name(node.name());
  const auto onnx_op_type = node.op_type();
  op_def->set_type(
      get_default(get_renamed_nodes(), onnx_op_type, onnx_op_type));
  auto mapper = [&, this](const std::string& k) {
    const auto it = get_node_renamed_attrs().find(onnx_op_type);
    if (it != get_node_renamed_attrs().end()) {
      const auto it_op = it->second.find(k);
      if (it_op != it->second.end()) {
        return it_op->second;
      }
    }
    const auto it_global = get_renamed_attrs().find(k);
    if (it_global != get_renamed_attrs().end()) {
      return it_global->second;
    }
    return k;
  };
  op_def->mutable_arg()->MergeFrom(onnx_node->attributes.AttrToArg(mapper));
  return returns;
}

ONNXImporterReturns ONNXBackend::GemmImporter(
    ONNXNode* onnx_node,
    const ConversionContext& ctx) {
  auto& attributes = onnx_node->attributes;
  auto alpha = attributes.get<float>("alpha", 1.f);
  auto beta = attributes.get<float>("beta", 1.f);
  auto trans_a = attributes.get<int64_t>("transA", 0L);
  // Remove the unsupported attributes
  if (alpha != 1.f || beta != 1.f) {
    LOG(FATAL) << "alpha/beta can not be set currently.";
  }
  if (trans_a) {
    LOG(FATAL) << "Tranposed A is not supported currently.";
  }
  attributes.remove("alpha");
  attributes.remove("beta");
  attributes.remove("transA");
  return GenericImporter(onnx_node, ctx);
}

ONNXImporterReturns ONNXBackend::MaxRoiPoolImporter(
    ONNXNode* onnx_node,
    const ConversionContext& ctx) {
  auto& attributes = onnx_node->attributes;
  auto pooled_shape = attributes.get<ONNX_INTS>("pooled_shape");
  attributes.AddRewrittenAttribute("pool_h")->set_i(pooled_shape.Get(0));
  attributes.AddRewrittenAttribute("pool_w")->set_i(pooled_shape.Get(1));
  attributes.remove("pooled_shape");
  return GenericImporter(onnx_node, ctx);
}

ONNXImporterReturns ONNXBackend::ReshapeImporter(
    ONNXNode* onnx_node,
    const ConversionContext& ctx) {
  auto node = NodeProto(onnx_node->node);
  auto onnx_node_v2 = ONNXNode(node);
  auto& attributes = onnx_node_v2.attributes;
  attributes.remove("consumed_inputs");
  // Determine the dims
  auto* dims = attributes.AddRewrittenAttribute("dims");
  if (ctx.opset_version() < 5) {
    const auto& shape = attributes.get<ONNX_INTS>("shape");
    CHECK_GT(shape.size(), 0) << "\nExcepted the shape value";
    attributes.remove("shape");
    for (auto d : shape) {
      dims->add_ints(d);
    }
  } else {
    CHECK_EQ(node.input_size(), 2)
        << "\nExpectd 2 input in upsample after onnx version 5";
    node.mutable_input()->Clear();
    node.add_input(onnx_node->node.input(0));
    const auto& shape_name = onnx_node->node.input(1);
    const auto* shape_tensor = ctx.initializer().at(shape_name);
    Argument shape_dtype, shape_values;
    ONNXTensorToArgument(*shape_tensor, &shape_dtype, &shape_values);
    CHECK_GT(shape_values.ints_size(), 0) << "\nExcepted the shape value";
    for (auto d : shape_values.ints()) {
      dims->add_ints(d);
    }
  }
  return GenericImporter(&onnx_node_v2, ctx);
}

ONNXImporterReturns ONNXBackend::ResizeImporter(
    ONNXNode* onnx_node,
    const ConversionContext& ctx) {
  auto node = NodeProto(onnx_node->node);
  auto onnx_node_v2 = ONNXNode(node);
  auto& attributes = onnx_node_v2.attributes;
  auto coord_mode = attributes.get<string>("coordinate_transformation_mode");
  attributes.remove("coordinate_transformation_mode");
  if (coord_mode == "align_corners") {
    attributes.AddRewrittenAttribute("align_corners")->set_i(1);
  }
  if (ctx.opset_version() >= 9) {
    node.mutable_input()->Clear();
    node.add_input(onnx_node->node.input(0));
    int scales_idx = 1, sizes_idx = -1;
    if (onnx_node->node.input_size() > 2) scales_idx = 2;
    if (onnx_node->node.input_size() > 3) sizes_idx = 3;
    Argument scales_dtype, scale_values;
    const auto& scales_name = onnx_node->node.input(scales_idx);
    const auto* scales_tensor = ctx.initializer().at(scales_name);
    ONNXTensorToArgument(*scales_tensor, &scales_dtype, &scale_values);
    auto* scales = attributes.AddRewrittenAttribute("scales");
    for (auto d : scale_values.floats()) {
      scales->add_floats(d);
    }
    if (sizes_idx > 0) {
      Argument sizes_dtype, sizes_values;
      const auto& sizes_name = onnx_node->node.input(sizes_idx);
      const auto* sizes_tensor = ctx.initializer().at(sizes_name);
      ONNXTensorToArgument(*sizes_tensor, &sizes_dtype, &sizes_values);
      auto* sizes = attributes.AddRewrittenAttribute("sizes");
      for (auto d : sizes_values.floats()) {
        sizes->add_ints(d);
      }
    }
  } else {
    LOG(FATAL) << "Required opset >= 7";
  }
  return GenericImporter(&onnx_node_v2, ctx);
}

ONNXImporterReturns ONNXBackend::RoiAlignImporter(
    ONNXNode* onnx_node,
    const ConversionContext& ctx) {
  auto node = NodeProto(onnx_node->node);
  auto onnx_node_v2 = ONNXNode(node);
  // Remove the batch indices
  node.mutable_input()->Clear();
  node.add_input(onnx_node->node.input(0));
  node.add_input(onnx_node->node.input(1));
  return GenericImporter(&onnx_node_v2, ctx);
}

ONNXImporterReturns ONNXBackend::TileImporter(
    ONNXNode* onnx_node,
    const ConversionContext& ctx) {
  auto node = NodeProto(onnx_node->node);
  auto onnx_node_v2 = ONNXNode(node);
  auto& attributes = onnx_node_v2.attributes;
  if (ctx.opset_version() >= 6) {
    // Determine repeats from repeats
    auto* repeats = attributes.AddRewrittenAttribute("repeats");
    node.mutable_input()->Clear();
    node.add_input(onnx_node->node.input(0));
    const auto& repeats_name = onnx_node->node.input(1);
    const auto* repeats_tensor = ctx.initializer().at(repeats_name);
    Argument repeats_dtype, repeats_values;
    ONNXTensorToArgument(*repeats_tensor, &repeats_dtype, &repeats_values);
    CHECK_GT(repeats_values.ints_size(), 0) << "\nExcepted the repeats value";
    for (auto repeat : repeats_values.ints()) {
      repeats->add_ints(repeat);
    }
  } else {
    LOG(FATAL) << "Required opset >= 6";
  }
  return GenericImporter(&onnx_node_v2, ctx);
}

#undef ONNX_INTS
#undef ONNX_FLOATS

} // namespace onnx

} // namespace dragon
