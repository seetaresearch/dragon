#include "dragon/core/operator.h"
#include "dragon/core/workspace.h"

namespace dragon {

OperatorBase::OperatorBase(const OperatorDef& def, Workspace* ws)
    : def_(def),
      workspace_(ws),
      phase_("TRAIN"),
      name_(def.name()),
      data_type_("float32"),
      data_format_("NCHW") {
  // Set arguments.
  for (auto& arg : def_.arg()) {
    CHECK_GT(arg.name().size(), 0);
    CHECK_EQ(args_.count(arg.name()), 0);
    args_[arg.name()] = &arg;
    if (arg.name() == "name") {
      name_ = arg.s();
    } else if (arg.name() == "dtype") {
      data_type_ = arg.s();
    } else if (arg.name() == "data_format") {
      data_format_ = arg.s();
    }
  }
  // Set inputs and outputs.
  for (const auto& input : def.input()) {
    inputs_.push_back(ws->GetTensor(input));
  }
  for (const auto& output : def.output()) {
    outputs_.push_back(ws->CreateTensor(output));
  }
}

Tensor& OperatorBase::Input(int index) {
  CHECK_LT(index, InputSize());
  CHECK_GE(index, -InputSize());
  if (index >= 0) return *inputs_[index];
  return *inputs_[index + inputs_.size()];
}

Tensor& OperatorBase::Input(const string& name) {
  return *workspace_->GetTensor(name_ + "/" + name);
}

Tensor* OperatorBase::Output(int index) {
  CHECK_LT(index, OutputSize());
  CHECK_GE(index, -OutputSize());
  if (index >= 0) return outputs_[index];
  return outputs_[index + outputs_.size()];
}

Tensor* OperatorBase::Output(int index, const vec32_t& inputs_at) {
  auto* output = Output(index);
  if (index >= outputs_from_.size()) return output;
  for (auto input_index : inputs_at) {
    auto& input = Input(input_index);
    if (outputs_from_[index].count(input.name())) {
      return output->ReshapeLike(input)->MapFrom(&input);
    }
  }
  return output;
}

Tensor* OperatorBase::Output(const string& name) {
  return workspace_->CreateTensor(name_ + "/" + name);
}

OperatorBase* OperatorBase::New(const OperatorDef& def, Workspace* ws) {
  const auto& op_type = def.type();
  auto* schema = OpSchemaRegistry::Schema(op_type);
  if (schema != nullptr) CHECK(schema->Verify(def));
  switch (def.device_option().device_type()) {
    case PROTO_CPU:
      return CPUOperatorRegistry()->Create(op_type, def, ws);
    case PROTO_CUDA:
#ifdef USE_CUDNN
      if (CUDNNOperatorRegistry()->Has(op_type) &&
          CUDAContext::objects().cudnn_enabled_) {
        return CUDNNOperatorRegistry()->Create(op_type, def, ws);
      }
#endif
      return CUDAOperatorRegistry()->Create(op_type, def, ws);
    case PROTO_MPS:
      return MPSOperatorRegistry()->Create(op_type, def, ws);
    default:
      LOG(FATAL) << "Unsupported device: " << def.device_option().device_type();
      return nullptr;
  }
}

OperatorBase* OperatorBase::DeriveFrom(const OperatorDef& def) {
  name_ = def.name();
  if (def.arg().size() > 1) {
    const auto& arg = *(def.arg().end() - 2);
    if (arg.name() == "name") name_ = arg.s();
  }
  inputs_.resize(def.input_size());
  outputs_.resize(def.output_size());
  for (int i = 0; i < inputs_.size(); i++) {
    inputs_[i] = workspace_->GetTensor(def.input(i));
  }
  for (int i = 0; i < outputs_.size(); i++) {
    outputs_[i] = workspace_->CreateTensor(def.output(i));
  }
  return this;
}

/* Operator Registry */

DEFINE_REGISTRY(
    CPUOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

DEFINE_REGISTRY(
    CUDAOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

DEFINE_REGISTRY(
    CUDNNOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

DEFINE_REGISTRY(
    MPSOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

/* Macros */

#define INSTANTIATE_GET_SINGLE_ARGUMENT(T, fieldname, default) \
  template <>                                                  \
  DRAGON_API T OperatorBase::GetArgument(                      \
      const string& name, const T& default_value) {            \
    if (args_.count(name) == 0) return default_value;          \
    CHECK(args_[name]->has_##fieldname());                     \
    return static_cast<T>(args_[name]->fieldname());           \
  }                                                            \
  template <>                                                  \
  DRAGON_API T OperatorBase::GetArgument(const string& name) { \
    return OperatorBase::GetArgument<T>(name, default);        \
  }

INSTANTIATE_GET_SINGLE_ARGUMENT(float, f, 0.f)
INSTANTIATE_GET_SINGLE_ARGUMENT(double, f, 0.);
INSTANTIATE_GET_SINGLE_ARGUMENT(int, i, 0);
INSTANTIATE_GET_SINGLE_ARGUMENT(bool, i, false);
INSTANTIATE_GET_SINGLE_ARGUMENT(int64_t, i, int64_t(0));
INSTANTIATE_GET_SINGLE_ARGUMENT(string, s, "");
#undef INSTANTIATE_GET_SINGLE_ARGUMENT

#define INSTANTIATE_GET_REPEATED_ARGUMENT(T, fieldname)             \
  template <>                                                       \
  vector<T> DRAGON_API OperatorBase::GetArgument<vector<T>>(        \
      const string& name, const vector<T>& default_value) {         \
    if (args_.count(name) == 0) return default_value;               \
    vector<T> values;                                               \
    for (const auto& v : args_[name]->fieldname()) {                \
      values.emplace_back(static_cast<T>(v));                       \
    }                                                               \
    return values;                                                  \
  }                                                                 \
  template <>                                                       \
  vector<T> DRAGON_API OperatorBase::GetArgument<vector<T>>(        \
      const string& name) {                                         \
    return OperatorBase::GetArgument<vector<T>>(name, vector<T>()); \
  }

INSTANTIATE_GET_REPEATED_ARGUMENT(float, floats);
INSTANTIATE_GET_REPEATED_ARGUMENT(double, floats);
INSTANTIATE_GET_REPEATED_ARGUMENT(int, ints);
INSTANTIATE_GET_REPEATED_ARGUMENT(bool, ints);
INSTANTIATE_GET_REPEATED_ARGUMENT(int64_t, ints);
INSTANTIATE_GET_REPEATED_ARGUMENT(string, strings);
#undef INSTANTIATE_GET_REPEATED_ARGUMENT

template class Operator<CPUContext>;
#ifdef USE_CUDA
template class Operator<CUDAContext>;
#endif
#ifdef USE_MPS
template class Operator<MPSContext>;
#endif

} // namespace dragon
