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

#ifndef DRAGON_ONNX_ONNX_BACKEND_H_
#define DRAGON_ONNX_ONNX_BACKEND_H_

#include "dragon/core/common.h"
#include "dragon/proto/onnx.pb.h"

#define ONNX_NAMESPACE onnx_dragon

namespace dragon {

namespace onnx {

const int kKnownOpsetVersion = 11;

using ONNX_NAMESPACE::AttributeProto;
using ONNX_NAMESPACE::GraphProto;
using ONNX_NAMESPACE::ModelProto;
using ONNX_NAMESPACE::NodeProto;
using ONNX_NAMESPACE::TensorProto;
using ONNX_NAMESPACE::ValueInfoProto;

using ValueInfoMap = Map<string, ValueInfoProto>;
using InitializerMap = Map<string, const TensorProto*>;

class ConversionContext {
 public:
  ConversionContext(
      const ValueInfoMap& value_infos,
      const InitializerMap& initializer,
      const int opset_version)
      : value_infos_(value_infos),
        initializer_(initializer),
        opset_version_(opset_version) {}

  const ValueInfoMap& value_infos() const {
    return value_infos_;
  }

  const InitializerMap& initializer() const {
    return initializer_;
  }

  const int opset_version() const {
    return opset_version_;
  }

 private:
  const ValueInfoMap& value_infos_;
  const InitializerMap& initializer_;
  const int opset_version_;
};

struct ONNXImporterReturns {
  vector<OperatorDef> ops;

  OperatorDef* AddOp() {
    ops.emplace_back(OperatorDef());
    return &ops.back();
  }

  OperatorDef* GetOp(int index) {
    CHECK_LT(index, ops.size());
    return &ops[index];
  }
};

class ONNXAttributes {
 public:
  ONNXAttributes(const NodeProto& node);

  bool HasAttribute(const string& key) const {
    return onnx_attrs_.count(key) > 0;
  }

  AttributeProto* AddRewrittenAttribute(const string& key) {
    auto tmp = rewritten_onnx_attrs_.emplace(key, AttributeProto());
    auto& attr = tmp.first->second;
    attr.set_name(key);
    return &attr;
  }

  google::protobuf::RepeatedPtrField<Argument> AttrToArg(
      std::function<string(const string&)> mapper) const;

  template <typename T>
  T get(const string& key) const;

  template <typename T>
  T get(const string& key, const T& default_value) const {
    if (onnx_attrs_.count(key)) {
      return get<T>(key);
    } else {
      return default_value;
    }
  }

  const AttributeProto* remove(const string& key) {
    const AttributeProto* result = nullptr;
    auto iter = onnx_attrs_.find(key);
    if (iter != onnx_attrs_.end()) {
      result = iter->second;
      onnx_attrs_.erase(iter);
    }
    return result;
  }

 private:
  Map<string, const AttributeProto*> onnx_attrs_;
  Map<string, AttributeProto> rewritten_onnx_attrs_;
};

template <>
int64_t ONNXAttributes::get(const string& key) const;
template <>
float ONNXAttributes::get(const string& key) const;

template <>
google::protobuf::RepeatedPtrField<string> ONNXAttributes::get(
    const string& key) const;

template <>
google::protobuf::RepeatedField<google::protobuf::int64> ONNXAttributes::get(
    const string& key) const;

template <>
google::protobuf::RepeatedField<float> ONNXAttributes::get(
    const string& key) const;

template <>
const TensorProto* ONNXAttributes::get(const std::string& key) const;

struct ONNXNode {
  ONNXNode(const NodeProto& node_in) : node(node_in), attributes(node_in) {}

  const NodeProto& node;
  ONNXAttributes attributes;
};

class ONNXBackend {
 public:
  void BuildTensorFillOp(const TensorProto& onnx_tensor, OperatorDef* op_def);

  ONNXImporterReturns ATenImporter(
      ONNXNode* onnx_node,
      const ConversionContext& ctx);

  ONNXImporterReturns BatchNormImporter(
      ONNXNode* onnx_node,
      const ConversionContext& ctx);

  ONNXImporterReturns CastImporter(
      ONNXNode* onnx_node,
      const ConversionContext& ctx);

  ONNXImporterReturns ConvPoolImporter(
      ONNXNode* onnx_node,
      const ConversionContext& ctx);

  ONNXImporterReturns GemmImporter(
      ONNXNode* onnx_node,
      const ConversionContext& ctx);

  ONNXImporterReturns GenericImporter(
      ONNXNode* onnx_node,
      const ConversionContext& ctx);

  ONNXImporterReturns MaxRoiPoolImporter(
      ONNXNode* onnx_node,
      const ConversionContext& ctx);

  ONNXImporterReturns ReshapeImporter(
      ONNXNode* onnx_node,
      const ConversionContext& ctx);

  ONNXImporterReturns ResizeImporter(
      ONNXNode* onnx_node,
      const ConversionContext& ctx);

  ONNXImporterReturns RoiAlignImporter(
      ONNXNode* onnx_node,
      const ConversionContext& ctx);

  ONNXImporterReturns TileImporter(
      ONNXNode* onnx_node,
      const ConversionContext& ctx);

  DRAGON_API void Prepare(
      const string& onnx_model_path,
      GraphDef* init_graph,
      GraphDef* pred_graph);

  void ONNXTensorToArgument(
      const TensorProto& onnx_tensor,
      Argument* dtype,
      Argument* values);

  ONNXImporterReturns ONNXNodeToOps(
      const ModelProto& init_model,
      const ModelProto& pred_model,
      const ConversionContext& ctx,
      ONNXNode* onnx_node);

  void ONNXToDragon(
      const ModelProto& onnx_model,
      const int opset_version,
      const bool include_initializers,
      GraphDef* init_graph,
      GraphDef* pred_graph);

  using SpecialNodeConverter =
      ONNXImporterReturns (ONNXBackend::*)(ONNXNode*, const ConversionContext&);

  const Map<string, string>& get_renamed_nodes() const;
  const Map<string, SpecialNodeConverter>& get_special_nodes() const;
  const Map<string, string>& get_renamed_attrs() const;
  const Map<string, Map<string, string>>& get_node_renamed_attrs() const;
};

} // namespace onnx

} // namespace dragon

#endif // DRAGON_ONNX_ONNX_BACKEND_H_
