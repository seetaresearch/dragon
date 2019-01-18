#include "onnx/onnx_backend.h"

namespace dragon {

namespace onnx {

ONNXAttributes::ONNXAttributes(const NodeProto& node) {
    for (const auto& attr : node.attribute()) 
        onnx_attrs_.emplace(attr.name(), &attr);
}

/*! float <- get(key) */

template <> float ONNXAttributes::get(const string& key) const {
    float value = 0.0;
    const auto it = onnx_attrs_.find(key);
    if (it != onnx_attrs_.end()) {
        const AttributeProto& attr = *it->second;
        value = attr.f();
    }
    return value;
}

/*! floats <- get(key) */

template <> google::protobuf::RepeatedField<float>
ONNXAttributes::get(const string& key) const {
    google::protobuf::RepeatedField<float> value;
    const auto it = onnx_attrs_.find(key);
    if (it != onnx_attrs_.end()) {
        const AttributeProto& attr = *it->second;
        value.CopyFrom(attr.floats());
    }
    return value;
}

/*! int64 <- get(key) */

template <> int64_t ONNXAttributes::get(const string& key) const {
    int64_t value = 0;
    const auto it = onnx_attrs_.find(key);
    if (it != onnx_attrs_.end()) {
        const AttributeProto& attr = *it->second;
        value = attr.i();
    }
    return value;
}

/*! string <- get(key) */

template <> string ONNXAttributes::get(const string& key) const {
    string value;
    const auto it = onnx_attrs_.find(key);
    if (it != onnx_attrs_.end()) {
        const AttributeProto& attr = *it->second;
        value = attr.s();
    }
    return value;
}

/*! strings <- get(key) */

template <> google::protobuf::RepeatedPtrField<string>
ONNXAttributes::get(const string& key) const {
    google::protobuf::RepeatedPtrField<std::string> value;
    const auto it = onnx_attrs_.find(key);
    if (it != onnx_attrs_.end()) {
        const AttributeProto& attr = *it->second;
        value.CopyFrom(attr.strings());
    } return value;
}

/*! ints <- get(key) */

template <> google::protobuf::RepeatedField<google::protobuf::int64>
ONNXAttributes::get(const string& key) const {
    google::protobuf::RepeatedField<google::protobuf::int64> value;
    const auto it = onnx_attrs_.find(key);
    if (it != onnx_attrs_.end()) {
        const AttributeProto& attr = *it->second;
        value.CopyFrom(attr.ints());
    } return value;
}

/*! argument <- attribute */

void CopyAttrValueToArg(
    Argument*                   arg,
    const AttributeProto&       attr) {
    if (attr.has_f()) {
        arg->set_f(attr.f());
    } else if (attr.has_i()) {
        arg->set_i(attr.i());
    } else if (attr.has_s()) {
        arg->set_s(attr.s());
    } else if (attr.has_t()) {
        // For proto, we convert it to serialized string
        std::string buffer;
        attr.t().SerializeToString(&buffer);
        arg->set_s(buffer);
    } else if (attr.floats_size()) {
        arg->mutable_floats()->CopyFrom(attr.floats());
    } else if (attr.ints_size()) {
        arg->mutable_ints()->CopyFrom(attr.ints());
    } else if (attr.strings_size()) {
        arg->mutable_strings()->CopyFrom(attr.strings());
    } else {
        LOG(FATAL) << "Unsupported ONNX attribute: " << attr.name();
    }
}

google::protobuf::RepeatedPtrField<Argument>
ONNXAttributes::AttrToArg(
    std::function<std::string(const std::string&)> mapper) const {
    google::protobuf::RepeatedPtrField<Argument> args;
    for (const auto& kv : onnx_attrs_) {
        // If the attribute was rewritten, we use it instead. Note that the
        // rewritten attribute still has the unmapped name
        const auto& attr = rewritten_onnx_attrs_.count(kv.first)
            ? rewritten_onnx_attrs_.at(kv.first) : (*kv.second);
        auto* arg = args.Add();
        arg->set_name(mapper(attr.name()));
        CopyAttrValueToArg(arg, attr);
    }
    for (const auto& kv : rewritten_onnx_attrs_) {
        // If rewritten attribute doesn't appear in the original attributes, this is
        // a newlly added one and we need to add this to argument too
        if (!onnx_attrs_.count(kv.first)) {
            const auto& attr = kv.second;
            auto* arg = args.Add();
            arg->set_name(mapper(attr.name()));
            CopyAttrValueToArg(arg, attr);
        }
    }
    return args;
}

}  // namespace onnx

}  // namespace dragon