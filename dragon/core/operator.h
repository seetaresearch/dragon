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
#include "dragon/core/operator_gradient.h"
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

  /*! \brief Update operator from the given def  */
  OperatorBase* UpdateFrom(const OperatorDef&);

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
      const string& entry = "type") const;

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

  /*! \brief Return the resource handle */
  const string& handle() const {
    return handle_;
  }

  /*! \brief Return the stored def */
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
  /*! \brief Store the parent workspace */
  Workspace* ws_;

  /*! \brief Store the def */
  OperatorDef def_;

  /*! \brief Store the recomputing subgraph */
  SubGraph subgraph_;

  /*! \brief Store the phase and handle */
  string phase_, handle_;

  /*! \brief Store the data type and format */
  string dtype_, data_format_;

  /*! \brief Store the pointer of inputs and outputs */
  vector<Tensor*> inputs_, outputs_;

  /*! \brief Store the candidate output aliases */
  vector<Set<string>> output_aliases_;

  /*! \brief Store the defined arguments */
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
      : OperatorBase(def, ws),
        ctx_(def.device_option()),
        do_sync_(OperatorBase::GetArgument<bool>("do_sync", false)) {}

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
    if (do_sync_) {
      ctx()->FinishDeviceComputation();
    }
    Release();
  }

  /*! \brief Return the context */
  Context* ctx() {
    return &ctx_;
  }

 protected:
  /*! \brief The context */
  Context ctx_;

  /*! \brief The executing flags */
  bool do_sync_;
};

/*! \brief New a operator from the raw def */
OperatorBase* NewOperator(const OperatorDef&, Workspace*);

/* Defines */

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

#define STORE_INPUT_SPEC(i)               \
  *(Buffer("X_spec:" + std::to_string(i)) \
        ->ReshapeLike(Input(i))           \
        ->set_meta(Input(i).meta()))

#define RESTORE_INPUT_SPEC(i) \
  *(workspace()->GetTensor(   \
      "/share/buffer/" + handle() + "/X_spec:" + std::to_string(i)))

/* Dispatchers */

template <typename... Types>
struct TensorTypes {};

using IntegralTensorTypes = TensorTypes<int8_t, uint8_t, int, int64_t>;

using FloatingTensorTypes = TensorTypes<float16, float, double>;

using NumericalTensorTypes =
    TensorTypes<int8_t, uint8_t, int, int64_t, float16, float, double>;

using BooleanIntegralTensorTypes =
    TensorTypes<bool, int8_t, uint8_t, int, int64_t, bool>;

using FullTensorTypes =
    TensorTypes<bool, int8_t, uint8_t, int, int64_t, float16, float, double>;

template <typename Sizes, typename... Args>
struct DispatchHelper;

#define DEFINE_TENSOR_TYPES_DISPATCHER(func)                               \
  template <typename T, typename... Types, typename... Args>               \
  struct DispatchHelper<TensorTypes<T, Types...>, Args...> {               \
    template <typename Op>                                                 \
    static void Call(Op* op, const TypeMeta& meta, string& types) {        \
      if (meta.Match<T>()) return op->template func<T, Args...>();         \
      types += "  * " + types::to_string<T>() + ",\n";                     \
      return DispatchHelper<TensorTypes<Types...>, Args...>::Call(         \
          op, meta, types);                                                \
    }                                                                      \
    template <typename Op>                                                 \
    static void Call(Op* op) {                                             \
      string types;                                                        \
      return Call(op, types::to_meta(op->dtype()), types);                 \
    }                                                                      \
    template <typename Op>                                                 \
    static void Call(Op* op, const Tensor& tensor) {                       \
      string types;                                                        \
      return Call(op, tensor.meta(), types);                               \
    }                                                                      \
  };                                                                       \
  template <typename... Args>                                              \
  struct DispatchHelper<TensorTypes<>, Args...> {                          \
    template <typename Op>                                                 \
    static void Call(Op* op, const TypeMeta& meta, string& types) {        \
      LOG(FATAL) << "Unsupported type: " << types::to_string(meta) << "\n" \
                 << "<" << op->type() << "Op>"                             \
                 << " supports the following type(s): {\n"                 \
                 << types << "}";                                          \
    }                                                                      \
    template <typename Op>                                                 \
    static void Call(Op* op, const Tensor& tensor) {                       \
      return Call(op, tensor.meta(), "");                                  \
    }                                                                      \
  };

DEFINE_TENSOR_TYPES_DISPATCHER(DoRunWithType);
#undef DEFINE_TENSOR_TYPES_DISPATCHER

/* Fillers */

#define TENSOR_FILL_WITH_TYPE(tensor, shape, type)                        \
  if (tensor.count() == 0) {                                              \
    auto* filler_info = workspace()->GetFillerInfo(tensor.name());        \
    CHECK(filler_info) << "\nTensor(" << tensor.name() << ") is empty.\n" \
                       << "May be specify a filler for it?";              \
    tensor.Reshape(shape);                                                \
    unique_ptr<Filler<type, Context>> filler(                             \
        CreateFiller<type, Context>(*filler_info));                       \
    filler->Fill(&tensor, ctx());                                         \
  } else {                                                                \
    int64_t count = 1;                                                    \
    for (int i = 0; i < shape.size(); i++)                                \
      count *= shape[i];                                                  \
    CHECK_EQ(count, tensor.count())                                       \
        << "\nExcepted Tensor(" << tensor.name() << ")'s "                \
        << "size is " << count << ", \n"                                  \
        << "but now is " << tensor.count() << ", "                        \
        << "did you feed the incorrect data before?";                     \
    tensor.Reshape(shape);                                                \
  }

#define TENSOR_FILL(tensor, shape)                                        \
  if (tensor.count() == 0) {                                              \
    auto* filler_info = workspace()->GetFillerInfo(tensor.name());        \
    CHECK(filler_info) << "\nTensor(" << tensor.name() << ") is empty.\n" \
                       << "May be specify a filler for it?";              \
    tensor.Reshape(shape);                                                \
    unique_ptr<Filler<T, Context>> filler(                                \
        CreateFiller<T, Context>(*filler_info));                          \
    filler->Fill(&tensor, ctx());                                         \
  } else {                                                                \
    int64_t count = 1;                                                    \
    for (int i = 0; i < shape.size(); i++)                                \
      count *= shape[i];                                                  \
    CHECK_EQ(count, tensor.count())                                       \
        << "\nExcepted Tensor(" << tensor.name() << ")'s "                \
        << "size is " << count << ", \n"                                  \
        << "but now is " << tensor.count() << ", "                        \
        << "did you feed the incorrect data before?";                     \
    tensor.Reshape(shape);                                                \
  }

/* Arguments */

#define OP_SINGLE_ARG(type, name, default) \
  OperatorBase::GetArgument<type>(name, (default))

#define OP_REPEATED_ARG(type, name) \
  OperatorBase::GetArgument<vector<type>>(name)

#define DECLARE_OP_SINGLE_ARG_WITH_DESC(type, arg) \
  type arg##_;                                     \
  string arg##_desc_;                              \
  type arg()

#define DECLARE_OP_REPEATED_ARG_WITH_DESC(type, arg) \
  string arg##_desc_;                                \
  vector<type> arg##_;                               \
  vector<string> arg##_descs_;                       \
  type arg(int i, int* num = nullptr)

#define INIT_OP_SINGLE_ARG_WITH_DESC(type, arg, default_value) \
  arg##_ = OP_SINGLE_ARG(type, #arg, default_value);           \
  arg##_desc_ = OP_SINGLE_ARG(string, string(#arg) + "_desc", "")

#define INIT_OP_REPEATED_ARG_WITH_DESC(type, arg)                  \
  arg##_ = OP_REPEATED_ARG(type, #arg);                            \
  arg##_desc_ = OP_SINGLE_ARG(string, string(#arg) + "_desc", ""); \
  arg##_descs_ = OP_REPEATED_ARG(string, string(#arg) + "_descs")

#define DEFINE_OP_SINGLE_ARG_WITH_DESC(type, classname, arg)      \
  template <class Context>                                        \
  type classname<Context>::arg() {                                \
    if (arg##_desc_.empty()) return arg##_;                       \
    auto* arg##_tensor = workspace()->GetTensor(                  \
        str::replace_first(arg##_desc_, "${HANDLE}", handle()));  \
    CHECK_EQ(arg##_tensor->count(), 1)                            \
        << "\nThe argument <" << #arg << "> should be a scalar."; \
    CHECK(arg##_tensor->template IsType<type>())                  \
        << "\nThe type of argument <" << #arg << "> should be "   \
        << types::to_string<type>() << ".";                       \
    return arg##_tensor->template data<type, CPUContext>()[0];    \
  }

#define DEFINE_OP_REPEATED_ARG_WITH_DESC(type, classname, arg)    \
  template <class Context>                                        \
  type classname<Context>::arg(int i, int* num) {                 \
    const type* data;                                             \
    string desc;                                                  \
    if (!arg##_desc_.empty()) {                                   \
      desc = arg##_desc_;                                         \
    } else if (!arg##_descs_.empty()) {                           \
      desc = arg##_descs_[i];                                     \
    }                                                             \
    if (!desc.empty()) {                                          \
      auto* arg##_tensor = workspace()->GetTensor(                \
          str::replace_first(desc, "${HANDLE}", handle()));       \
      CHECK(arg##_tensor->template IsType<type>())                \
          << "\nThe type of argument <" << #arg << "> should be " \
          << types::to_string<type>() << ".";                     \
      data = arg##_tensor->template data<type, CPUContext>();     \
      if (num != nullptr) {                                       \
        *num = arg##_desc_.empty() ? (int)arg##_descs_.size()     \
                                   : (int)arg##_tensor->size();   \
      }                                                           \
    } else {                                                      \
      data = arg##_.data();                                       \
      if (num != nullptr) {                                       \
        *num = (int)arg##_.size();                                \
      }                                                           \
    }                                                             \
    if (num != nullptr && (*num) == 0) return type(0);            \
    return arg##_descs_.empty() ? data[i] : data[0];              \
  }

#define CANONICALIZE_AXIS_WITH_TENSOR_AND_OFFSET(tensor, offset)         \
  auto axis = OP_SINGLE_ARG(int64_t, "axis", INT_MAX);                   \
  if (axis != INT_MAX) {                                                 \
    axis = axis < 0 ? axis + tensor.ndim() + offset : axis;              \
    CHECK(axis >= 0 && axis < tensor.ndim() + offset)                    \
        << "\nExcepted the axis in [-" << tensor.ndim() + offset << ", " \
        << tensor.ndim() + offset << "), got "                           \
        << OP_SINGLE_ARG(int64_t, "axis", INT_MAX) << ".";               \
  }

#define CANONICALIZE_AXIS_WITH_TENSOR(tensor) \
  CANONICALIZE_AXIS_WITH_TENSOR_AND_OFFSET(tensor, 0)

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

DECLARE_REGISTRY(
    CNMLOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

#define INSTANTIATE_OPERATOR(name, context) template class name##Op<context>;

#define INSTANTIATE_CUDNN_OPERATOR(name) \
  template class CuDNN##name##Op<CUDAContext>;

#define INSTANTIATE_CNML_OPERATOR(name) \
  template class CnML##name##Op<CNMLContext>;

#define REGISTER_CPU_OPERATOR(name, ...) \
  REGISTER_CLASS(CPUOperatorRegistry, name, __VA_ARGS__)

#define REGISTER_CUDA_OPERATOR(name, ...) \
  REGISTER_CLASS(CUDAOperatorRegistry, name, __VA_ARGS__)

#define REGISTER_CUDNN_OPERATOR(name, ...) \
  REGISTER_CLASS(CUDNNOperatorRegistry, name, __VA_ARGS__)

#define REGISTER_CNML_OPERATOR(name, ...) \
  REGISTER_CLASS(CNMLOperatorRegistry, name, __VA_ARGS__)

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

#define DEPLOY_CNML_OPERATOR(name)                           \
  REGISTER_CNML_OPERATOR(name, CnML##name##Op<CNMLContext>); \
  INSTANTIATE_CNML_OPERATOR(name);

} // namespace dragon

#endif // DRAGON_CORE_OPERATOR_H_
