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

#ifndef DRAGON_CORE_OPERATOR_H_
#define DRAGON_CORE_OPERATOR_H_

#include "dragon/core/context.h"
#include "dragon/core/gradient.h"
#include "dragon/core/operator_schema.h"
#include "dragon/core/registry.h"
#include "dragon/core/tensor.h"
#include "dragon/utils/conversions.h"

namespace dragon {

class Workspace;

class DRAGON_API OperatorBase {
 public:
  typedef Map<string, vector<OperatorBase*>> SubGraph;

  /*! \brief Constructor with the def and workspace */
  OperatorBase(const OperatorDef&, Workspace*);

  /*! \brief Destructor */
  virtual ~OperatorBase() {}

  /*! \brief Create a new operator */
  static OperatorBase* New(const OperatorDef& def, Workspace* ws);

  /*! \brief Derive a new operator from the base */
  OperatorBase* DeriveFrom(const OperatorDef& def);

  /*! \brief Fusion operator into the given graph */
  virtual void Fuse(void* graph) {
    NOT_IMPLEMENTED;
  }

  /*! \brief Run operator on the given stream */
  virtual void Run(int stream = 0) {
    NOT_IMPLEMENTED;
  }

  /*! \brief Switch to the given executing phase */
  void SwitchToPhase(const string& phase) {
    phase_ = phase;
  }

  /*! \brief Return the input tensor */
  Tensor& Input(int i);

  /*! \brief Return the output tensor */
  Tensor* Output(int i);

  /*! \brief Return the output tensor with input aliases */
  Tensor* Output(int i, const vec32_t& inputs);

  /*! \brief Return the buffer tensor */
  Tensor* Buffer(const string& name);

  /*! \brief Return the number of inputs */
  int InputSize() {
    return (int)inputs_.size();
  }

  /*! \brief Return the number of outputs */
  int OutputSize() {
    return (int)outputs_.size();
  }

  /*! \brief Return the value of argument */
  template <typename T>
  T GetArgument(const string& name);

  /*! \brief Return the value of argument with default */
  template <typename T>
  T GetArgument(const string& name, const T& default_value);

  /*! \brief Return the message for supported value */
  string MessageForUnsupported(
      const string& value,
      const vector<string>& support_values,
      const string& entry = "type") const {
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

  /*! \brief Return the specified argument */
  const Argument& arg(const string& name) {
    return *(args_[name]);
  }

  /*! \brief Return all the arguments */
  const Map<string, const Argument*>& args() {
    return args_;
  }

  /*! \brief Return the operator name */
  const string& name() const {
    return def_.name();
  }

  /*! \brief Return the operator type */
  const string& type() const {
    return def_.type();
  }

  /*! \brief Return the running phase */
  const string& phase() const {
    return phase_;
  }

  /*! \brief Return the data type */
  const string& dtype() const {
    return dtype_;
  }

  /*! \brief Return the data format */
  const string& data_format() const {
    return data_format_;
  }

  /*! \brief Return the execution handle */
  const string& handle() const {
    return handle_;
  }

  /*! \brief Return the operator def */
  const OperatorDef& def() const {
    return def_;
  }

  /*! \brief Return the recomputing subgraph */
  SubGraph& subgraph() {
    return subgraph_;
  }

  /*! \brief Return the parent workspace */
  Workspace* workspace() const {
    return ws_;
  }

  /*! \brief Set the subgraph for recomputing */
  void set_subgraph(SubGraph subgraph) {
    subgraph_ = subgraph;
  }

  /*! \brief Set the output aliases for in-place */
  void set_output_aliases(const Map<string, Set<string>>& alias_map) {
    output_aliases_.resize(outputs_.size());
    for (int i = 0; i < outputs_.size(); ++i) {
      const auto& it = alias_map.find(outputs_[i]->name());
      if (it != alias_map.end()) {
        output_aliases_[i] = it->second;
      } else {
        output_aliases_[i].clear();
      }
    }
  }

 protected:
  /*! \brief The parent workspace */
  Workspace* ws_;

  /*! \brief The operator def */
  OperatorDef def_;

  /*! \brief The recomputing subgraph */
  SubGraph subgraph_;

  /*! \brief The execution phase */
  string phase_;

  /*! \brief The execution handle */
  string handle_;

  /*! \brief The data type */
  string dtype_;

  /*! \brief The data format */
  string data_format_;

  /*! \brief The input and output tensors */
  vector<Tensor*> inputs_, outputs_;

  /*! \brief The candidate output aliases */
  vector<Set<string>> output_aliases_;

  /*! \brief The argument references */
  Map<string, const Argument*> args_;

  DISABLE_COPY_AND_ASSIGN(OperatorBase);
};

/*!
 * \brief The base operator class with context.
 */
template <class Context>
class DRAGON_API Operator : public OperatorBase {
 public:
  /*! \brief Constructor with the def and workspace */
  Operator(const OperatorDef& def, Workspace* ws)
      : OperatorBase(def, ws), ctx_(def.device_option()) {}

  /*! \brief Prepare the content of inputs */
  virtual void Prepare();

  /*! \brief Release the ownership of inputs */
  virtual void Release();

  /*! \brief The detailed execution on device */
  virtual void RunOnDevice() = 0;

  /*! \brief Run this operator */
  void Run(int stream = 0) final {
    Prepare();
    ctx()->SwitchToDevice(stream);
    RunOnDevice();
    Release();
  }

  /*! \brief Return the context */
  Context* ctx() {
    return &ctx_;
  }

 protected:
  /*! \brief The context */
  Context ctx_;
};

/* Macros */

#define SIMPLE_CTOR_DTOR(name)                                                \
  name(const OperatorDef& def, Workspace* ws) : Operator<Context>(def, ws) {} \
  virtual ~name() {}

#define USE_OPERATOR_BASE_FUNCTIONS          \
  using OperatorBase::SwitchToPhase;         \
  using OperatorBase::Input;                 \
  using OperatorBase::Output;                \
  using OperatorBase::Buffer;                \
  using OperatorBase::InputSize;             \
  using OperatorBase::OutputSize;            \
  using OperatorBase::MessageForUnsupported; \
  using OperatorBase::name;                  \
  using OperatorBase::type;                  \
  using OperatorBase::phase;                 \
  using OperatorBase::dtype;                 \
  using OperatorBase::data_format;           \
  using OperatorBase::handle;                \
  using OperatorBase::def;                   \
  using OperatorBase::workspace

#define USE_OPERATOR_FUNCTIONS \
  USE_OPERATOR_BASE_FUNCTIONS; \
  using Operator<Context>::ctx

#define SET_INPUT_SPEC(i)                                         \
  do {                                                            \
    auto& Xi = Input(i);                                          \
    workspace()                                                   \
        ->CreateTensor(handle() + "/X_spec:" + std::to_string(i)) \
        ->ReshapeLike(Xi)                                         \
        ->set_meta(Xi.meta());                                    \
  } while (0);

#define INPUT_SPEC(i) \
  *(workspace()->GetTensor(handle() + "/X_spec:" + std::to_string(i)))

/* Dispatchers */

template <typename Sizes, typename... Args>
struct DispatchHelper;

#define DEFINE_DTYPES_DISPATCHER(Func)                                      \
  template <typename T, typename... Types, typename... Args>                \
  struct DispatchHelper<dtypes::TypesBase<T, Types...>, Args...> {          \
    template <typename Op>                                                  \
    static void Call(Op* op, const TypeMeta& meta, string& types_str) {     \
      if (meta.Match<T>()) return op->template Func<T, Args...>();          \
      types_str += "  * " + dtypes::to_string<T>() + ",\n";                 \
      return DispatchHelper<dtypes::TypesBase<Types...>, Args...>::Call(    \
          op, meta, types_str);                                             \
    }                                                                       \
    template <typename Op>                                                  \
    static void Call(Op* op) {                                              \
      string types_str;                                                     \
      return Call(op, dtypes::to_meta(op->dtype()), types_str);             \
    }                                                                       \
    template <typename Op>                                                  \
    static void Call(Op* op, const Tensor& tensor) {                        \
      string types_str;                                                     \
      return Call(op, tensor.meta(), types_str);                            \
    }                                                                       \
  };                                                                        \
  template <typename... Args>                                               \
  struct DispatchHelper<dtypes::TypesBase<>, Args...> {                     \
    template <typename Op>                                                  \
    static void Call(Op* op, const TypeMeta& meta, string& types_str) {     \
      LOG(FATAL) << "Unsupported type: " << dtypes::to_string(meta) << "\n" \
                 << "<" << op->type() << "Op>"                              \
                 << " supports the following type(s): {\n"                  \
                 << types_str << "}";                                       \
    }                                                                       \
    template <typename Op>                                                  \
    static void Call(Op* op, const Tensor& tensor) {                        \
      return Call(op, tensor.meta(), "");                                   \
    }                                                                       \
  };

DEFINE_DTYPES_DISPATCHER(DoRunWithType);
#undef DEFINE_DTYPES_DISPATCHER

/* Initializers */

#define INITIALIZE_TENSOR_VIA_SPEC(tensor, kShape, kDType)                    \
  do {                                                                        \
    int64_t count = 1;                                                        \
    for (int i = 0; i < kShape.size(); i++) {                                 \
      count *= kShape[i];                                                     \
    }                                                                         \
    CHECK_EQ(count, tensor.count())                                           \
        << "\nExcepted size of tensor '" << tensor.name() << "' is " << count \
        << ", got " << tensor.count() << ".";                                 \
    CHECK(TypeMeta::Make<kDType>() == tensor.meta())                          \
        << "\nExcepted dtype of tensor '" << tensor.name() << "' is "         \
        << dtypes::to_string<kDType>() << ", got "                            \
        << dtypes::to_string(tensor.meta()) << ".";                           \
    tensor.Reshape(kShape);                                                   \
  } while (0)

/* Arguments */

#define OP_SINGLE_ARG(type, name, default) \
  OperatorBase::GetArgument<type>(name, (default))

#define OP_REPEATED_ARG(type, name) \
  OperatorBase::GetArgument<vector<type>>(name)

#define DECLARE_OP_SINGLE_ARG(type, arg) \
  type arg##_;                           \
  string arg##_desc_;                    \
  type arg()

#define DECLARE_OP_REPEATED_ARG(type, arg) \
  string arg##_desc_;                      \
  vector<type> arg##_;                     \
  type arg(int i, int* num = nullptr)

#define INITIALIZE_OP_SINGLE_ARG(type, arg, default_value) \
  arg##_ = OP_SINGLE_ARG(type, #arg, default_value);       \
  arg##_desc_ = OP_SINGLE_ARG(string, string(#arg) + "_desc", "")

#define INITIALIZE_OP_REPEATED_ARG(type, arg) \
  arg##_ = OP_REPEATED_ARG(type, #arg);       \
  arg##_desc_ = OP_SINGLE_ARG(string, string(#arg) + "_desc", "");

#define DEFINE_OP_SINGLE_ARG(type, classname, arg)                   \
  template <class Context>                                           \
  type classname<Context>::arg() {                                   \
    if (arg##_desc_.empty()) return arg##_;                          \
    auto* arg##_tensor = workspace()->GetTensor(                     \
        str::replace_first(arg##_desc_, "$HANDLE", handle()));       \
    CHECK_EQ(arg##_tensor->count(), 1)                               \
        << "\nArgument <" << #arg << "> should be a size-1 scalar."; \
    CHECK(arg##_tensor->template IsType<type>())                     \
        << "\nType of argument <" << #arg << "> should be "          \
        << dtypes::to_string<type>() << ".";                         \
    return arg##_tensor->template data<type, CPUContext>()[0];       \
  }

#define DEFINE_OP_REPEATED_ARG(type, classname, arg)             \
  template <class Context>                                       \
  type classname<Context>::arg(int i, int* num) {                \
    const type* data;                                            \
    int N;                                                       \
    if (!arg##_desc_.empty()) {                                  \
      auto* arg##_tensor = workspace()->GetTensor(               \
          str::replace_first(arg##_desc_, "$HANDLE", handle())); \
      CHECK(arg##_tensor->template IsType<type>())               \
          << "\nType of argument <" << #arg << "> should be "    \
          << dtypes::to_string<type>() << ".";                   \
      data = arg##_tensor->template data<type, CPUContext>();    \
      N = int(arg##_tensor->size());                             \
    } else {                                                     \
      data = arg##_.data(), N = int(arg##_.size());              \
    }                                                            \
    if (num != nullptr) *num = N;                                \
    return i < N ? data[i] : type(0);                            \
  }

#define GET_OP_AXIS_ARG(arg, num_axes, default_value)                  \
  auto arg = OP_SINGLE_ARG(int64_t, #arg, default_value);              \
  if (arg != INT_MAX) {                                                \
    arg = arg < 0 ? arg + num_axes : arg;                              \
    CHECK(arg >= 0 && arg < num_axes)                                  \
        << "\nExcepted the <" << #arg << "> in [-" << num_axes << ", " \
        << num_axes << "), got "                                       \
        << OP_SINGLE_ARG(int64_t, #arg, default_value) << ".";         \
  }

/* Registers */

DECLARE_REGISTRY(
    CPUOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

DECLARE_REGISTRY(
    CUDAOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

DECLARE_REGISTRY(
    CUDNNOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

#define INSTANTIATE_OPERATOR(name, context) template class name##Op<context>;

#define INSTANTIATE_CUDNN_OPERATOR(name) \
  template class CuDNN##name##Op<CUDAContext>;

#define REGISTER_CPU_OPERATOR(name, ...) \
  REGISTER_CLASS(CPUOperatorRegistry, name, __VA_ARGS__)

#define REGISTER_CUDA_OPERATOR(name, ...) \
  REGISTER_CLASS(CUDAOperatorRegistry, name, __VA_ARGS__)

#define REGISTER_CUDNN_OPERATOR(name, ...) \
  REGISTER_CLASS(CUDNNOperatorRegistry, name, __VA_ARGS__)

#define DEPLOY_CPU_OPERATOR(name)                    \
  REGISTER_CPU_OPERATOR(name, name##Op<CPUContext>); \
  INSTANTIATE_OPERATOR(name, CPUContext);

#define DEPLOY_CUDA_OPERATOR(name)                     \
  REGISTER_CUDA_OPERATOR(name, name##Op<CUDAContext>); \
  INSTANTIATE_OPERATOR(name, CUDAContext);

#define DEPLOY_CPU_CUDA_OPERATOR(name)                \
  REGISTER_CPU_OPERATOR(name, name##Op<CPUContext>);  \
  REGISTER_CUDA_OPERATOR(name, name##Op<CPUContext>); \
  INSTANTIATE_OPERATOR(name, CPUContext);

#define DEPLOY_CUDNN_OPERATOR(name)                            \
  REGISTER_CUDNN_OPERATOR(name, CuDNN##name##Op<CUDAContext>); \
  INSTANTIATE_CUDNN_OPERATOR(name);

} // namespace dragon

#endif // DRAGON_CORE_OPERATOR_H_
