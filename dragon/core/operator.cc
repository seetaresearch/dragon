#include "dragon/core/operator.h"
#include "dragon/core/workspace.h"

namespace dragon {

OperatorBase::OperatorBase(const OperatorDef& def, Workspace* ws)
    : def_(def),
      ws_(ws),
      phase_("TRAIN"),
      handle_(def.name()),
      dtype_("float32"),
      data_format_("NCHW") {
  // Scan the defined arguments
  for (auto& arg : def_.arg()) {
    CHECK_GT(arg.name().size(), 0);
    CHECK_EQ(args_.count(arg.name()), 0);
    args_[arg.name()] = &arg;
    if (arg.name() == "handle") {
      handle_ = arg.s();
    } else if (arg.name() == "dtype") {
      dtype_ = arg.s();
    } else if (arg.name() == "data_format") {
      data_format_ = arg.s();
    }
  }

  // Set the inputs and outputs
  size_t version_pos;
  for (const auto& input : def.input()) {
    string name = input;
    if ((version_pos = input.find("/ver:")) != string::npos) {
      name = input.substr(0, version_pos);
    }
    inputs_.push_back(ws->GetTensor(name));
  }
  for (const auto& output : def.output()) {
    string name = output;
    if ((version_pos = output.find("/ver:")) != string::npos) {
      name = output.substr(0, version_pos);
    }
    outputs_.push_back(ws->CreateTensor(name));
  }
}

Tensor& OperatorBase::Input(int i) {
  CHECK_LT(i, (int)inputs_.size());
  CHECK_GE(i, -(int)inputs_.size());
  if (i >= 0) return *inputs_[i];
  return *inputs_[i + inputs_.size()];
}

Tensor* OperatorBase::Output(int i) {
  CHECK_LT(i, (int)outputs_.size());
  CHECK_GE(i, -(int)outputs_.size());
  if (i >= 0) return outputs_[i];
  return outputs_[i + outputs_.size()];
}

Tensor* OperatorBase::Output(int i, const vec32_t& inputs) {
  auto* Y = Output(i);
  if (i < output_aliases_.size()) {
    for (auto j : inputs) {
      const auto& X = Input(j);
      if (output_aliases_[i].count(X.name())) {
        Output(i)->ReshapeLike(X)->Share(X.memory());
        return Y;
      }
    }
  }
  Y->Share(nullptr);
  return Y;
}

Tensor* OperatorBase::Buffer(const string& name) {
  return workspace()->CreateTensor(handle_ + "/" + name);
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
    default:
      LOG(FATAL) << "Unknown device: " << def.device_option().device_type();
      return nullptr;
  }
}

OperatorBase* OperatorBase::DeriveFrom(const OperatorDef& def) {
  handle_ = def.name();
  if (def.arg().size() > 1) {
    const auto& arg = *(def.arg().end() - 2);
    if (arg.name() == "handle") handle_ = arg.s();
  }
  inputs_.resize(def.input_size());
  outputs_.resize(def.output_size());
  for (int i = 0; i < inputs_.size(); i++) {
    inputs_[i] = workspace()->GetTensor(def.input(i));
  }
  for (int i = 0; i < outputs_.size(); i++) {
    outputs_[i] = workspace()->CreateTensor(def.output(i));
  }
  return this;
}

template <class Context>
void Operator<Context>::Prepare() {
  for (int i = 0; i < InputSize(); i++) {
    if (Input(i).version() >= 0) {
      const auto& name = def().input(i);
      auto ver_pos = name.find("/ver:");
      auto version = std::atoi(name.substr(ver_pos + 5).c_str());
      if (version == Input(i).version()) continue;
      LOG(DEBUG) << "Excepted version of Tensor(" + Input(i).name() + ") "
                 << "is " << version << ", got " << Input(i).version()
                 << ". Recompute.";
      Tensor* flag = workspace()->GetTensor("flagged/recomp");
      flag->mutable_data<bool, CPUContext>()[0] = true;
      vector<OperatorBase*>& chain = subgraph()[name];
      for (auto* op : chain) {
        op->Run(ctx()->stream());
      }
      flag->mutable_data<bool, CPUContext>()[0] = false;
    }
  }
}

template <class Context>
void Operator<Context>::Release() {
  for (int i = 0; i < OutputSize(); i++) {
    if (Output(i)->version() >= 0) {
      const auto& name = def().output(i);
      auto ver_pos = name.find("/ver:");
      auto version = std::atoi(name.substr(ver_pos + 5).c_str());
      Output(i)->set_version(version);
    }
  }
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
template class Operator<CUDAContext>;

} // namespace dragon
