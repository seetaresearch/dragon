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
  string name;
  size_t ver_pos;
  for (const auto& e : def.input()) {
    name = e;
    if ((ver_pos = e.find("/ver:")) != string::npos)
      name = e.substr(0, ver_pos);
    inputs_.push_back(ws->GetTensor(name));
  }
  for (const auto& e : def.output()) {
    name = e;
    if ((ver_pos = e.find("/ver:")) != string::npos)
      name = e.substr(0, ver_pos);
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
  return workspace()->CreateTensor("/share/buffer/" + handle_ + "/" + name);
}

string OperatorBase::MessageForUnsupported(
    const string& value,
    const vector<string>& support_values,
    const string& entry) const {
  std::stringstream ss;
  ss << "Unsupported " << entry << ": " << value << "\n";
  ss << "<" << type() << "Op>"
     << " supports the following " << entry << "(s): {\n";
  for (const auto& support_value : support_values) {
    ss << "  * " << support_value << ",\n";
  }
  ss << "}";
  return ss.str();
}

OperatorBase* OperatorBase::UpdateFrom(const OperatorDef& def) {
  handle_ = def.name();
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
      Tensor* flag = workspace()->GetTensor("/share/flag/recomputing");
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

OperatorBase*
TryCreateOperator(const string& key, const OperatorDef& def, Workspace* ws) {
  switch (def.device_option().device_type()) {
    case PROTO_CPU:
      return CPUOperatorRegistry()->Create(key, def, ws);
    case PROTO_CUDA:
#ifdef USE_CUDNN
      if (CUDNNOperatorRegistry()->Has(key) &&
          CUDAContext::objects().cudnn_enabled_) {
        return CUDNNOperatorRegistry()->Create(key, def, ws);
      }
#endif
      return CUDAOperatorRegistry()->Create(key, def, ws);
    case PROTO_CNML:
      return CNMLOperatorRegistry()->Create(key, def, ws);
    default:
      LOG(FATAL) << "Unknown Device: " << def.device_option().device_type();
      return nullptr;
  }
}

OperatorBase* NewOperator(const OperatorDef& def, Workspace* ws) {
  auto* schema = OpSchemaRegistry::Schema(def.type());
  if (schema != nullptr) CHECK(schema->Verify(def));
  OperatorDef mutable_def(def);
  // Heuristically make each random seed slightly different
  static unsigned int seed_offset = 0;
  mutable_def.mutable_device_option()->set_random_seed(
      seed_offset + def.device_option().random_seed());
  seed_offset = (seed_offset + 1) % UINT32_MAX;
  return TryCreateOperator(def.type(), mutable_def, ws);
}

GradientPack MakeGradientForOp(
    const OperatorDef& def,
    const vector<string>& grad_outputs) {
  CHECK(GradientRegistry()->Has(def.type()))
      << "\nNo GradientMaker registered for " << def.type() << "Op.";
  OperatorDef reference_def(def);
  unique_ptr<GradientMakerBase> maker(
      GradientRegistry()->Create(def.type(), def, grad_outputs));
  GradientPack pack = maker->Make();
  // Copy cache key
  if (reference_def.has_cache_key()) {
    for (int i = 0; i < pack.grad_defs.size(); ++i) {
      pack.grad_defs[i].set_cache_key(
          reference_def.cache_key() + "/grad" +
          (i > 0 ? (":" + str::to(i)) : ""));
    }
  }
  // Copy device option and arguments
  if (maker->CopyDeviceOption() && def.has_device_option()) {
    for (auto& grad_def : pack.grad_defs) {
      grad_def.mutable_device_option()->CopyFrom(def.device_option());
    }
  }
  // Copy arguments
  if (maker->CopyArguments() && def.arg_size()) {
    for (auto& grad_def : pack.grad_defs) {
      grad_def.mutable_arg()->MergeFrom(reference_def.arg());
    }
  }
  return pack;
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
    CNMLOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

DEFINE_REGISTRY(
    GradientRegistry,
    GradientMakerBase,
    const OperatorDef&,
    const vector<string>&);

DEFINE_REGISTRY(
    NoGradientRegistry,
    GradientMakerBase,
    const OperatorDef&,
    const vector<string>&);

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
      values.push_back(static_cast<T>(v));                          \
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
template class Operator<CNMLContext>;

} // namespace dragon
