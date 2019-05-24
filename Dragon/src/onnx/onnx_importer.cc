#include "utils/map_utils.h"
#include "onnx/onnx_backend.h"

namespace dragon {

namespace onnx {

#define ONNX_FLOATS google::protobuf::RepeatedField<float>
#define ONNX_INTS google::protobuf::RepeatedField<google::protobuf::int64>

bool AlmostEqual(double a, double b) {
    const static double kEps = 1e-15;
    return (fabs(a - b) < kEps);
}

/*! Common Operators */

ONNXImporterReturns ONNXBackend::CommonONNXNodeImporter(
    ONNXNode*                       onnx_node,
    const ConversionContext&        ctx) {
    ONNXImporterReturns returns;
    auto* op_def = returns.AddOp();
    const auto& node = onnx_node->node;
    op_def->mutable_input()->MergeFrom(node.input());
    op_def->mutable_output()->MergeFrom(node.output());
    op_def->set_name(node.name());
    const auto onnx_op_type = node.op_type();
    op_def->set_type(get_default(get_renamed_nodes(),
        onnx_op_type, onnx_op_type));

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
    op_def->mutable_arg()->MergeFrom(
        onnx_node->attributes.AttrToArg(mapper));
    return returns;
}

/*! Conv2d, ConvTranspose2d, Pool2d */

ONNXImporterReturns ONNXBackend::ConvPoolNodeImporter(
    ONNXNode*                       onnx_node,
    const ConversionContext&        ctx) {
    auto& attributes = onnx_node->attributes;
    const auto onnx_op_type = onnx_node->node.op_type();

    // Determine the padding
    auto mode = attributes.get<string>("auto_pad");
    auto* padding = attributes.AddRewrittenAttribute("padding");
    // SAME, SAME_LOWER, or SAME_UPPER
    if (str::find(mode, "SAME")) padding->set_s(mode);
    else padding->set_s("VALID");  // Use explicit pads
    attributes.remove("auto_pad");

    // Determine the pooling mode
    if (onnx_op_type == "MaxPool") {
        auto* attr = attributes.AddRewrittenAttribute("mode");
        attr->set_s("MAX");
    } else if (onnx_op_type == "AveragePool") {
        auto* attr = onnx_node->attributes.AddRewrittenAttribute("mode");
        attr->set_s("AVG");
    }

    auto returns = CommonONNXNodeImporter(onnx_node, ctx);

    // Determine the op type
    OperatorDef* op_def = returns.GetOp(0);
    auto ks = attributes.get<ONNX_INTS>("kernel_shape");
    *(op_def->mutable_type()) += (str::to(ks.size()) + "d");

    return returns;
}

/*! FullyConnected */

ONNXImporterReturns ONNXBackend::GemmNodeImporter(
    ONNXNode*                       onnx_node,
    const ConversionContext&        ctx) {
    auto& attributes = onnx_node->attributes;
    auto alpha = attributes.get<float>("alpha", 1.f);
    auto beta = attributes.get<float>("beta", 1.f);
    auto trans_a = attributes.get<int64_t>("transA", 0L);

    if (alpha != 1.f || beta != 1.f) {
        LOG(FATAL) << "alpha/beta can not be set currently.";
    } if (trans_a) {
        LOG(FATAL) << "Tranposed A is not supported currently.";
    }

    attributes.remove("alpha");
    attributes.remove("beta");
    attributes.remove("transA");

    return CommonONNXNodeImporter(onnx_node, ctx);
}

/*! FusedBatchNorm */

ONNXImporterReturns ONNXBackend::BatchNormNodeImporter(
    ONNXNode*                       onnx_node,
    const ConversionContext&        ctx) {
    auto node = NodeProto(onnx_node->node);
    auto onnx_node_v2 = ONNXNode(node);

    auto& attributes = onnx_node_v2.attributes;

    auto* axis = attributes.AddRewrittenAttribute("axis");
    axis->set_i(1);  // Enforce NCHW format

    attributes.remove("is_test");
    attributes.remove("consumed_inputs");

    if (attributes.HasAttribute("spatial")) {
        auto spatial = attributes.get<int64_t>("spatial");
        CHECK_EQ(spatial, 1) << "Excepted spatial should be 1.";
        attributes.remove("spatial");
    }

    // [scale, bias, mean, var] -> [mean, var, scale, bias]
    *node.mutable_input(1) = onnx_node->node.input(3);
    *node.mutable_input(2) = onnx_node->node.input(4);
    *node.mutable_input(3) = onnx_node->node.input(1);
    *node.mutable_input(4) = onnx_node->node.input(2);

    return CommonONNXNodeImporter(&onnx_node_v2, ctx);
}

/*! NNResize, BilinearResize */

ONNXImporterReturns ONNXBackend::UpsampleNodeImporter(
    ONNXNode*                       onnx_node,
    const ConversionContext&        ctx) {
    auto node = NodeProto(onnx_node->node);
    auto onnx_node_v2 = ONNXNode(node);
    auto& attributes = onnx_node_v2.attributes;

    // Determine the mode
    auto mode = attributes.get<string>("mode");
    if (mode == "nearest") node.set_op_type("NNResize");
    else if (mode == "bilinear") node.set_op_type("BilinearResize");
    else LOG(FATAL) << "Unsuported upsample mode: " << mode;

    attributes.remove("mode");

    // Determine the fy and fx
    auto* fy = attributes.AddRewrittenAttribute("fy");
    auto* fx = attributes.AddRewrittenAttribute("fx");

    if (ctx.opset_version() >= 7 && ctx.opset_version() < 9) {
        const auto& scales = attributes.get<ONNX_FLOATS>("scales");
        if (scales.size() != 4) {
            LOG(FATAL) << "The scales argument should have size 4";
        } else if (!AlmostEqual(scales.Get(0), 1) || 
                   !AlmostEqual(scales.Get(1), 1))  {
            LOG(FATAL) << "The first two elements in the scales must be 1";
        } 
        attributes.remove("scales");
        fy->set_f(scales.Get(2));
        fx->set_f(scales.Get(3));
    } else if (ctx.opset_version() >= 9) {
        CHECK_EQ(node.input_size(), 2)
            << "\nExpectd 2 input in upsample after onnx version 9";
        node.mutable_input()->Clear();
        node.add_input(onnx_node->node.input(0));
        const auto& scales_name = onnx_node->node.input(1);
        const auto* scales_tensor = ctx.initializer().at(scales_name);
        Argument scales_dtype, scale_values;
        ONNXTensorToArgument(*scales_tensor, &scales_dtype, &scale_values);
        if (scale_values.floats_size() != 4) {
            LOG(FATAL) << "The scales argument should have size 4";
        } else if (!AlmostEqual(scale_values.floats(0), 1) ||
                   !AlmostEqual(scale_values.floats(1), 1)) {
            LOG(FATAL) << "The first two elements in the scales must be 1";
        }
        fy->set_f(scale_values.floats(2));
        fx->set_f(scale_values.floats(3));
    } else { LOG(FATAL) << "Required opset >= 7"; }

    return CommonONNXNodeImporter(&onnx_node_v2, ctx);
}

/*! Reshape */

ONNXImporterReturns ONNXBackend::ReshapeNodeImporter(
    ONNXNode*                       onnx_node,
    const ConversionContext&        ctx) {
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
        for (auto d : shape) dims->add_ints(d);
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
        for (auto d : shape_values.ints()) dims->add_ints(d);
    }

    return CommonONNXNodeImporter(&onnx_node_v2, ctx);
}

/*! Internal Dragon Operators */

ONNXImporterReturns ONNXBackend::ATenNodeImporter(
    ONNXNode*                       onnx_node,
    const ConversionContext&        ctx) {
    auto node = NodeProto(onnx_node->node);
    auto onnx_node_v2 = ONNXNode(node);
    auto& attributes = onnx_node_v2.attributes;

    auto op_type = attributes.get<string>("op_type", "");

    if (op_type.empty()) {
        LOG(FATAL) << "op_type is required to evolve "
                      "to the specific operator.";
    }

    node.set_op_type(op_type);
    attributes.remove("op_type");

    return CommonONNXNodeImporter(&onnx_node_v2, ctx);
}

/*! ROIPool */

ONNXImporterReturns ONNXBackend::MaxRoiPoolNodeImporter(
    ONNXNode*                       onnx_node,
    const ConversionContext&        ctx) {
    auto& attributes = onnx_node->attributes;
    auto pooled_shape = attributes.get<ONNX_INTS>("pooled_shape");

    auto* pool_h = attributes.AddRewrittenAttribute("pool_h");
    auto* pool_w = attributes.AddRewrittenAttribute("pool_w");
    pool_h->set_i(pooled_shape.Get(0)); pool_w->set_i(pooled_shape.Get(1));

    attributes.remove("pooled_shape");

    return CommonONNXNodeImporter(onnx_node, ctx);
}

/*! AsType */

ONNXImporterReturns ONNXBackend::CastNodeImporter(
    ONNXNode*                       onnx_node,
    const ConversionContext&        ctx) {
    auto& attributes = onnx_node->attributes;

    // Determine the dtype
    auto* dtype = attributes.AddRewrittenAttribute("dtype");
    auto onnx_dtype = attributes.get<int64_t>("to", TensorProto::UNDEFINED);
    auto supported_dtype = true;

    switch (onnx_dtype) {
        case ONNX_NAMESPACE::TensorProto::BOOL:
            dtype->set_s("bool"); break;
        case ONNX_NAMESPACE::TensorProto::INT8:
            dtype->set_s("int8"); break;
        case ONNX_NAMESPACE::TensorProto::UINT8:
            dtype->set_s("uint8"); break;
        case ONNX_NAMESPACE::TensorProto::INT32:
            dtype->set_s("int32"); break;
        case ONNX_NAMESPACE::TensorProto::INT64:
            dtype->set_s("int64"); break;
        case ONNX_NAMESPACE::TensorProto::FLOAT16:
            dtype->set_s("float16"); break;
        case ONNX_NAMESPACE::TensorProto::FLOAT:
            dtype->set_s("float32"); break;
        case ONNX_NAMESPACE::TensorProto::DOUBLE:
            dtype->set_s("float64"); break;
        case ONNX_NAMESPACE::TensorProto::INT16:
            dtype->set_s("int16"); supported_dtype = false; break;
        case ONNX_NAMESPACE::TensorProto::UINT16:
            dtype->set_s("uint16"); supported_dtype = false; break;
        case ONNX_NAMESPACE::TensorProto::UINT32:
            dtype->set_s("uint32"); supported_dtype = false; break;
        case ONNX_NAMESPACE::TensorProto::UINT64:
            dtype->set_s("uint64"); supported_dtype = false; break;
        case ONNX_NAMESPACE::TensorProto::STRING:
            dtype->set_s("string"); supported_dtype = false; break;
        case ONNX_NAMESPACE::TensorProto::COMPLEX64:
            dtype->set_s("complex64"); supported_dtype = false; break;
        case ONNX_NAMESPACE::TensorProto::COMPLEX128:
            dtype->set_s("complex128"); supported_dtype = false; break;
        case ONNX_NAMESPACE::TensorProto::UNDEFINED:
            dtype->set_s("undefined"); supported_dtype = false; break;
    };

    CHECK(supported_dtype) << "\nCasting to "
        << dtype->s() << " is not supported.";
    attributes.remove("to");

    return CommonONNXNodeImporter(onnx_node, ctx);
}

/*! L2Norm */

ONNXImporterReturns ONNXBackend::LpNormNodeImporter(
    ONNXNode*                       onnx_node,
    const ConversionContext&        ctx) {
    auto node = NodeProto(onnx_node->node);
    auto onnx_node_v2 = ONNXNode(node);
    auto& attributes = onnx_node_v2.attributes;

    // Determine the "p", i.e. op type
    auto p = attributes.get<int64_t>("p", 2);
    node.set_op_type("L" + str::to(p) + "Norm");
    attributes.remove("p");

    auto* num_axes = attributes.AddRewrittenAttribute("num_axes");
    num_axes->set_i(1);  // ONNX does not support multiple axes

    return CommonONNXNodeImporter(&onnx_node_v2, ctx);
}

/*! ArgReduce */

ONNXImporterReturns ONNXBackend::ArgReduceNodeImporter(
    ONNXNode*                       onnx_node,
    const ConversionContext&        ctx) {
    auto node = NodeProto(onnx_node->node);
    auto onnx_node_v2 = ONNXNode(node);
    auto& attributes = onnx_node_v2.attributes;

    // Determine the operation
    auto* operation = attributes.AddRewrittenAttribute("operation");
    if (onnx_node->node.op_type() == "ArgMax") operation->set_s("ARGMAX");
    else if (onnx_node->node.op_type() == "ArgMin") operation->set_s("ARGMIN");
    node.add_output("NULL");  // A dummy output("Value") is required

    return CommonONNXNodeImporter(&onnx_node_v2, ctx);
}

#undef ONNX_INTS
#undef ONNX_FLOATS

}  // namespace onnx

}  // namespace dragon