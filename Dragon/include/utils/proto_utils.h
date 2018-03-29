// ------------------------------------------------------------
// Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_UTILS_PROTO_UTILS_H_
#define DRAGON_UTILS_PROTO_UTILS_H_

#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>

#include "protos/dragon.pb.h"
#include "core/common.h"
#include "utils/logging.h"

namespace dragon {

using google::protobuf::Message;

template <class IterableInputs,class IterableOutputs,class IterableArgs>
inline OperatorDef MakeOperatorDef(const string& type,
                                   const string& name,
                                   const IterableInputs& inputs,
                                   const IterableOutputs& outputs,
                                   const IterableArgs& args,
                                   const DeviceOption& device_option,
                                   const string& engine) {
    OperatorDef def;
    def.set_type(type);
    def.set_name(name);
    for (const string& in : inputs) def.add_input(in);
    for (const string& out : outputs) def.add_output(out);
    for (const Argument& arg : args) def.add_arg()->CopyFrom(arg);
    if (device_option.has_device_type()) def.mutable_device_option()->CopyFrom(device_option);
    return def;
}

template <class IterableInputs, class IterableOutputs, class IterableArgs>
inline OperatorDef MakeOperatorDef(const string& type,
                                   const string& name,
                                   const IterableInputs& inputs,
                                   const IterableOutputs& outputs,
                                   const IterableArgs& args) {
    return MakeOperatorDef(type, name, inputs, outputs, args, DeviceOption(), "");
}

template <class IterableInputs, class IterableOutputs>
inline OperatorDef MakeOperatorDef(const string& type,
                                   const string& name,
                                   const IterableInputs& inputs,
                                   const IterableOutputs& outputs) {
    return MakeOperatorDef(type, name, inputs, outputs, vector<Argument>(), DeviceOption(), "");
}

inline void ParseProtoFromText(string text, Message* proto) {
    google::protobuf::TextFormat::ParseFromString(text, proto);
}

}    // namespace dragon

#endif    // DRAGON_UTILS_PROTO_UTILS_H_
