#include "core/operator_schema.h"
#include "utils/proto_utils.h"
#include "onnx/onnx_backend.h"

namespace dragon {

namespace onnx {

void ONNXBackend::Prepare(
    const string&                   onnx_model_path,
    GraphDef*                       init_graph,
    GraphDef*                       pred_graph) {
    ModelProto onnx_model;
    CHECK(ReadProtoFromBinaryFile(
        onnx_model_path.c_str(), &onnx_model))
            << "\nFailed to parse the onnx model.";

    int opset_version = -1;
    for (const auto& imp : onnx_model.opset_import()) {
        if ((!imp.has_domain()) || imp.domain().empty()) {
            opset_version = imp.version();
            if (opset_version > kKnownOpsetVersion) {
                std::cout
                    << "This version of onnx-dragon targets ONNX operator set version "
                    << kKnownOpsetVersion
                    << ", but the model we are trying to import uses version "
                    << opset_version << ".  We will try to import it anyway, "
                    << "but if the model uses operators which had BC-breaking changes "
                       "in the intervening versions, import will fail."
                    << std::endl;
            }
        } else {
            std::cout << "Unrecognized operator set "
                      << opset_version << std::endl;
        }
    }

    if (opset_version < 0) {
        if (onnx_model.ir_version() >= 0x00000003) {
            LOG(FATAL) << "Model with IR version >= 3 "
                "did not specify ONNX operator set version.";
        } else {
            opset_version = 1;
        }
    }

    ONNXToDragon(onnx_model, opset_version,
        true, init_graph, pred_graph);
}

void ONNXBackend::ONNXToDragon(
    const ModelProto&               onnx_model,
    const int                       opset_version,
    const bool                      include_initializers,
    GraphDef*                       init_graph,
    GraphDef*                       pred_graph) {
    ModelProto init_model = ModelProto();
    ModelProto pred_model = onnx_model;

    pred_graph->set_name(onnx_model.graph().name());
    init_graph->set_name(onnx_model.graph().name() + "/init");

    ValueInfoMap graph_value_infos{};
    InitializerMap graph_initializer{};

    for (const auto& vi : onnx_model.graph().input())
        graph_value_infos[vi.name()].CopyFrom(vi);

    for (const auto& vi : onnx_model.graph().output())
        graph_value_infos[vi.name()].CopyFrom(vi);

    for (const auto& vi : onnx_model.graph().value_info())
        graph_value_infos[vi.name()].CopyFrom(vi);

    for (const auto& tensor : onnx_model.graph().initializer()) {
        if (include_initializers) {
            auto* op_def = init_graph->add_op();
            init_graph->add_output(tensor.name());
            BuildTensorFillOp(tensor, op_def);
        }
        graph_initializer[tensor.name()] = &tensor;
    }

    auto converter = [&](
        const ModelProto& model, GraphDef* graph) mutable {
        for (const auto& node : model.graph().node()) {
            ValueInfoMap value_infos{};
            InitializerMap initializer{};
            for (const auto& name : node.input()) {
                if (graph_value_infos.count(name))
                    value_infos[name].CopyFrom(graph_value_infos[name]);
                if (graph_initializer.count(name))
                    initializer[name] = graph_initializer[name];
            }
            auto onnx_node = ONNXNode(node);
            auto returns = ONNXNodeToOps(init_model, pred_model,
                { value_infos, initializer, opset_version }, &onnx_node);
            for (const auto& op : returns.ops) {
                pred_graph->add_op()->CopyFrom(op);
            }
        }
    };

    converter(pred_model, pred_graph);

    // Set(Initializer) + Set(Placehoders) = Set(Inputs)
    Set<string> initializer;
    for (const auto& e : onnx_model.graph().initializer()) {
        initializer.insert(e.name());
    }

    // Add External Inputs
    for (const auto& e : onnx_model.graph().input()) {
        if (initializer.count(e.name()) == 0) {
            pred_graph->add_input(e.name());
        }
    }

    // Add External Outputs
    for (const auto& e : onnx_model.graph().output()) {
        pred_graph->add_output(e.name());
    }
}

ONNXImporterReturns ONNXBackend::ONNXNodeToOps(
    const ModelProto&               init_model,
    const ModelProto&               pred_model,
    const ConversionContext&        ctx,
    ONNXNode*                       onnx_node) {
    ONNXImporterReturns returns;
    if (get_special_nodes().count(onnx_node->node.op_type())) {
        returns = (this->*get_special_nodes().at(
            onnx_node->node.op_type()))(onnx_node, ctx);
    } else {
        returns = CommonONNXNodeImporter(onnx_node, ctx);
    }
    for (const auto& op: returns.ops) {
        const auto* schema = OpSchemaRegistry::Schema(op.type());
        if (!schema) {
            LOG(FATAL) << "Dragon has no such operator, "
                "could not find schema for " << op.type();
        }
    }
    return returns;
}

const Map<string, string>& ONNXBackend::get_renamed_nodes() const {
    const static Map<string, string>
        kRenamedNodes {
            { "MaxPool", "Pool" },
            { "AveragePool", "Pool" },
            { "Gemm", "FullyConnected" },
            { "BatchNormalization", "FusedBatchNorm" },
            { "MaxRoiPool", "ROIPool" },
            { "Cast", "AsType" },
            { "ArgMax", "ArgReduce" },
            { "ArgMin", "ArgReduce" },
            { "Identity", "Copy" },
    }; return kRenamedNodes;
}

const Map<string, ONNXBackend::SpecialNodeConverter>&
ONNXBackend::get_special_nodes() const {
    const static Map<string, ONNXBackend::SpecialNodeConverter>
        kSpecialNodes = {
            { "Conv", &ONNXBackend::ConvPoolNodeImporter },
            { "ConvTranspose", &ONNXBackend::ConvPoolNodeImporter },
            { "MaxPool", &ONNXBackend::ConvPoolNodeImporter },
            { "AveragePool", &ONNXBackend::ConvPoolNodeImporter },
            { "Gemm", &ONNXBackend::GemmNodeImporter },
            { "BatchNormalization", &ONNXBackend::BatchNormNodeImporter },
            { "Upsample", &ONNXBackend::UpsampleNodeImporter },
            { "Reshape", &ONNXBackend::ReshapeNodeImporter },
            { "ATen", &ONNXBackend::ATenNodeImporter },
            { "MaxRoiPool", &ONNXBackend::MaxRoiPoolNodeImporter },
            { "Cast", &ONNXBackend::CastNodeImporter },
            { "LpNormalization", &ONNXBackend::LpNormNodeImporter },
            { "ArgMax", &ONNXBackend::ArgReduceNodeImporter },
            { "ArgMin", &ONNXBackend::ArgReduceNodeImporter },
    }; return kSpecialNodes;
}

const Map< string, Map<string, string> >&
ONNXBackend::get_node_renamed_attrs() const {
    const static Map< string, Map<string, string> >
        kPerNodeRenamedAttrs = {
            { "Gemm", { { "transB", "transW" } } },
            { "BatchNormalization", { { "epsilon", "eps" } } },
    }; return kPerNodeRenamedAttrs;
}

const Map<string, string>& ONNXBackend::get_renamed_attrs() const {
    const static Map<string, string>
        kRenamedAttrs {
            { "keepdims", "keep_dims" },
    }; return kRenamedAttrs;
}

}  // namespace onnx

}  // namespace dragon