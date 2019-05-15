/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_CORE_OPERATOR_H_
#define DRAGON_CORE_OPERATOR_H_

#include "core/registry.h"
#include "core/context.h"
#include "core/tensor.h"
#include "core/operator_gradient.h"
#include "core/operator_schema.h"
#include "utils/cast.h"

#ifdef WITH_MPI
#include <mpi.h>
#endif

namespace dragon {

class Workspace;

class OperatorBase {
 public:
    /*! \brief Default constructor */
    OperatorBase(const OperatorDef& def, Workspace* ws);

    /*! \brief Default deconstructor */
    virtual ~OperatorBase() {}

    /*! \brief Return the specified input tensor */
    Tensor& X(int i);

    /*! \brief Return the specified output tensor */
    Tensor* Y(int i);

    /*! \brief Return the number of inputs */
    int XSize() { return (int)inputs_.size(); }

    /*! \brief Return the number of outputs */
    int YSize() { return (int)outputs_.size(); }

    /*! \brief Modify this operator according to the given def  */
    void UpdateFrom(const OperatorDef& def);

    /*! \brief Switch the internal running phase */
    void SwitchToPhase(const string& phase) { phase_ = phase; }

    /*! \brief Run this operator on the specified stream */
    virtual void Run(int stream_id = 0) { NOT_IMPLEMENTED; }

    /*! \brief Fusion this operator into the specified graph */
    virtual void Fusion(void* graph) { NOT_IMPLEMENTED; }

    /*! \brief Return the operator name */
    const string& name() const { return def_.name(); }

    /*! \brief Return the operator type */
    const string& type() const { return def_.type(); }

    /*! \brief Return the current running phase */
    const string& phase() const { return phase_; }

    /*! \brief Return the anchor name of this operator */
    const string& anchor() const { return anchor_; }

    /*! \brief Return the data type of this operator */
    const string& dtype() const { return dtype_; }

    /*! \brief Return the data format of this operator */
    const string& data_format() const { return data_format_; }

    /*! \brief Return the unique name in this operator */
    const string unique_name(const string& name) const {
        return "/mnt/" + anchor_ + "/" + name;
    }

    /*! \brief Return the parent workspace */
    Workspace* ws() const { return ws_; }

    /*! \brief Return the value of the specified argument */
    template <typename T>
    T Arg(const string& name, const T& default_value);

    /*! \brief Return the values of the specified argument */
    template <typename T>
    vector<T> Args(const string& name);

    /*! \brief Return the argument map of this operator */
    const Map<std::string, const Argument*>& args() { return args_; }

    /*! \brief Return the specified argument */
    const Argument& arg(const string& name) { return *(args_[name]); }

    typedef Map<string, vector<OperatorBase*>> SubGraph;

    /*! \brief Return the recomputing subgraph of this operator */
    SubGraph& subgraph() { return subgraph_; }

    /*! \brief Set the given recomputing subgraph */
    void set_subgraph(SubGraph subgraph) {
        subgraph_ = subgraph;
    }

    /*! \brief Return the stored operator def */
    const OperatorDef& def() const { return def_; }

    /*! \brief Return the debug string of the stored operator def */
    string DebugString() const { return def_.DebugString(); }

    /*! \brief Return the dtype string according to given tensor */
    string DTypeString(
        const Tensor&           tensor,
        const Set<string>&      dtypes) const;

    /* \brief Return the dtype string according to given type */
    string DTypeString(
        const string&           dtype,
        const Set<string>&      dtypes) const;

 protected:
    Workspace* ws_;
    OperatorDef def_;
    SubGraph subgraph_;
    string phase_, anchor_;
    string dtype_, data_format_;
    vector<Tensor*> inputs_, outputs_;
    Map<string, const Argument*> args_;
};

template <class Context>
class Operator : public OperatorBase {
 public:
    /*! \brief Default constructor */
    Operator(const OperatorDef& def, Workspace* ws)
        : OperatorBase(def, ws),
          ctx_(def.device_option()),
          do_sync_(OperatorBase::Arg<bool>(
              "do_sync", false)),
          allow_recomp_(OperatorBase::Arg<bool>(
              "allow_recomp", false)) {
        allow_run_ = true;
        allow_run_ &= MPICheck();
        allow_run_ &= (!(YSize() == 1 &&
            Y(0)->name() == "NULL"));
    }

    /*! \brief Run this operator on the specified stream */
    void Run(int stream_id = 0) final {
        if (!allow_run_) return;
        if (allow_recomp_) PrepareResource();
        ctx()->SwitchToDevice(stream_id);
        MemorySwitch();
        RunOnDevice();
        if (do_sync_ || stream_id > 0) {
            // Sync the stream(0) at the specific time
            ctx()->FinishDeviceCompution();
        }
        if (allow_recomp_) ReleaseResource();
    }

    /*! \brief Prepare the content of inputs */
    virtual void PrepareResource();

    /*! \brief Release the ownership of inputs */
    virtual void ReleaseResource();

    /*! \brief Coordinate the context of inputs and outputs */
    virtual void MemorySwitch() {
        for (auto* e : inputs_)
            if(e->name() != "NULL")
                e->SwitchToDevice(ctx()->device_id());
        for (auto* e : outputs_)
            if(e->name() != "NULL")
                e->SwitchToDevice(ctx()->device_id());
    }

    /*! \brief Implement the detailed execution */
    virtual void RunOnDevice() = 0;

    /*! \brief Return the internal context */
    Context* ctx() { return &ctx_; }

    /*! \brief Whether this operator can be ignored */
    bool AllowRun() { return allow_run_; }

 protected:
    /*! \brief Store the internal context */
    Context ctx_;
    bool allow_run_, allow_recomp_, do_sync_;

 private:
    /*! \brief Check the MPI conditions */
    bool MPICheck() {
#ifndef WITH_MPI
        return true;
#else
        vec32_t allow_ranks =
            OperatorBase::Args<int>("mpi_ranks");
        if (allow_ranks.empty()) return true;
        int cur_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &cur_rank);
        for (auto mpi_rank : allow_ranks)
            if (cur_rank == mpi_rank) return true;
        return false;
#endif
    }
};

/*! \brief New a operator from the raw def */

OperatorBase* NewOperator(
    const OperatorDef&          def,
    Workspace*                  ws);

/* Macros */

#define OpArg OperatorBase::Arg
#define OpArgs OperatorBase::Args

#define SIMPLE_CTOR_DTOR(name) \
    name(const OperatorDef& def, Workspace* ws) \
        : Operator<Context>(def, ws) {} \
    virtual ~name() {}

#define USE_OPERATOR_BASE_FUNCTIONS \
    using OperatorBase::ws; \
    using OperatorBase::name; \
    using OperatorBase::type; \
    using OperatorBase::phase; \
    using OperatorBase::anchor; \
    using OperatorBase::dtype; \
    using OperatorBase::data_format; \
    using OperatorBase::unique_name; \
    using OperatorBase::def; \
    using OperatorBase::X; \
    using OperatorBase::Y; \
    using OperatorBase::XSize; \
    using OperatorBase::YSize; \
    using OperatorBase::DebugString; \
    using OperatorBase::DTypeString; \
    using OperatorBase::SwitchToPhase

#define USE_OPERATOR_FUNCTIONS \
    USE_OPERATOR_BASE_FUNCTIONS; \
    using Operator<Context>::ctx; \
    using Operator<Context>::AllowRun

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

/* NVIDIA's Accelerated Library - CUDNN */

DECLARE_REGISTRY(
    CUDNNOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

/* CAMBRICON's Accelerated Library - CNML */

DECLARE_REGISTRY(
    CNMLOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

/* Dispatcher for Runtime Typed-Implementation */

#define XIsType(x, dtype) \
    x.template IsType<dtype>()

template <typename... Types>
struct TensorTypes {};

template <typename Sizes, typename... Args>
struct DispatchHelper;

#define DEFINE_TENSOR_TYPES_DISPATCHER(TensorTypes, Impl) \
    template <typename T, typename... Types, typename... Args> \
    struct DispatchHelper<TensorTypes<T, Types...>, Args...> { \
        template <typename Op> \
        static void Call(Op* op, const TypeMeta& meta, string& types) { \
            if (meta.Match<T>()) return op->template Impl<T, Args...>(); \
            types += "    * " + TypeToString<T>() + ",\n"; \
            return DispatchHelper<TensorTypes<Types...>, Args...> \
                ::Call(op, meta, types); \
        } \
        template <typename Op> \
        static void Call(Op* op, const Tensor& tensor) { \
            string types; return Call(op, tensor.meta(), types); \
        } \
    }; \
    template <typename... Args> \
    struct DispatchHelper<TensorTypes<>, Args...> { \
        template <typename Op> \
        static void Call(Op* op, const TypeMeta& meta, string& types) { \
            LOG(FATAL) << "Unsupported DType: " \
                       << TypeMetaToString(meta) << "\n" \
                       << "<" << op->type() << "Op>" \
                       << " supports the following dtypes: {\n" \
                       << types << "}"; \
        } \
        template <typename Op> \
        static void Call(Op* op, const Tensor& tensor) { \
            return Call(op, tensor.meta(), ""); \
        } \
    };

DEFINE_TENSOR_TYPES_DISPATCHER(TensorTypes, RunImpl);
#undef DEFINE_TENSOR_TYPES_DISPATCHER

/* TensorFiller */

#define TENSOR_FILL_WITH_TYPE(tensor, shape, type) \
    if (tensor.count() == 0) { \
        CHECK(ws()->GetFiller(tensor.name())) \
            << "\nTensor(" << tensor.name() << ") is empty. \n" \
            << "may be specify a filler for it ?"; \
        tensor.Reshape(shape); \
        unique_ptr<Filler<type, Context>> filler(  \
            CreateFiller<type, Context>(*ws()->GetFiller(tensor.name()))); \
        filler->Fill(&tensor, ctx()); \
    } else { \
         int64_t count = 1; \
         for(int i = 0; i < shape.size(); i++) count *= shape[i]; \
         CHECK_EQ(count, tensor.count()) \
            << "\nModel request " << "Tensor(" << tensor.name() << ")'s " \
            << "size is " << count << ", \n" \
            << "but now is " << tensor.count() << ", " \
            << "did you feed the incorrect Tensor before ?"; \
        tensor.Reshape(shape); \
    }

#define TENSOR_FILL(tensor, shape) \
    if (tensor.count() == 0) { \
        CHECK(ws()->GetFiller(tensor.name())) \
            << "\nTensor(" << tensor.name() << ") is empty. \n" \
            << "may be specify a filler for it ?"; \
        tensor.Reshape(shape); \
        unique_ptr<Filler<T, Context>> filler(  \
            CreateFiller<T, Context>(*ws()->GetFiller(tensor.name()))); \
        filler->Fill(&tensor, ctx()); \
    } else { \
         int64_t count = 1; \
         for(int i = 0; i < shape.size(); i++) count *= shape[i]; \
         CHECK_EQ(count, tensor.count()) \
            << "\nModel request " << "Tensor(" << tensor.name() << ")'s " \
            << "size is " << count << ", \n" \
            << "but now is " << tensor.count() << ", " \
            << "did you feed the incorrect Tensor before ?"; \
        tensor.Reshape(shape); \
    }

/* Shared Multiplier */

#define DECLARE_MULTIPLIER(name, size) \
    const T* name; \
    { \
        auto* mp = ws()->CreateTensor("/share/multiplier/" \
            + TypeMetaToString(TypeMeta::Make<T>())); \
        if (size > mp->count()) { \
            mp->Reshape({ size }); \
            math::Set<T, Context>(size, cast::to<T>(1.f), \
                mp->template mutable_data<T, Context>(), ctx()); \
        } \
        name = mp->template data<T, Context>(); \
    }

/* Dynamic Arguments */

#define DECLARE_ARG_WITH_DESC(type, arg) \
    type arg##_; \
    string arg##_desc_; \
    type arg()

#define DECLARE_ARGS_WITH_DESC(type, arg) \
    vector<type> arg##_; \
    vector<string> arg##_desc_; \
    type arg(int i)

#define GET_ARG_WITH_DESC(type, arg, default_value) \
    arg##_ = OpArg<type>(#arg, default_value); \
    arg##_desc_ = OpArg<string>(string(#arg) + "_desc", "")

#define GET_ARGS_WITH_DESC(type, arg) \
    arg##_ = OpArgs<type>(#arg); \
    arg##_desc_ = OpArgs<string>(string(#arg) + "_desc")

#define DEFINE_ARG_WITH_DESC(type, classname, arg) \
    template <class Context> \
    type classname<Context>::arg() { \
        if (arg##_desc_.empty()) return arg##_; \
        auto* arg##T = ws()->GetTensor(arg##_desc_); \
        CHECK(arg##T->template IsType<type>()) \
            << "\nThe type of " << #arg << " should be " << #type << "."; \
        CHECK_EQ(arg##T->count(), 1) \
            << "\nThe argument of " << #arg << " should be a scalar."; \
        return arg##T->template data<type, CPUContext>()[0]; \
    }

#define DEFINE_ARGS_WITH_DESC(type, classname, arg) \
    template <class Context> \
    type classname<Context>::arg(int i) { \
        if (arg##_desc_.empty()) { \
            CHECK_LT(i, arg##_.size()) \
                << "\nExcepted the size of " << #arg \
                << " > " << i << ". (Got " \
                << arg##_.size() << ")."; \
            return arg##_[i]; \
        } \
        CHECK_LT(i, arg##_desc_.size()) \
            << "\nExcepted the size of " << #arg \
            << " > " << i << ". (Got " \
            << arg##_desc_.size() << ")."; \
        auto* arg##T = ws()->GetTensor( \
            str::replace_first(arg##_desc_[i], \
                "${ANCHOR}", anchor())); \
        CHECK(arg##T->template IsType<type>()) \
            << "\nThe type of " << #arg << " should be " << #type << "."; \
        CHECK_EQ(arg##T->count(), 1) \
            << "\nThe argument of " << #arg << " at pos(" \
            << i << ") should be a scalar."; \
        return arg##T->template data<type, CPUContext>()[0]; \
    }

#define GET_ARGS_SIZE(arg) \
    (int)std::max(arg##_.size(), arg##_desc_.size())

/* Registers */

#define INSTANTIATE_OPERATOR(name, context) \
    template class name##Op<context>;

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

#define DEPLOY_CPU(name) \
    REGISTER_CPU_OPERATOR(name, name##Op<CPUContext>); \
    INSTANTIATE_OPERATOR(name, CPUContext);

#define DEPLOY_CUDA(name) \
    REGISTER_CUDA_OPERATOR(name, name##Op<CUDAContext>); \
    INSTANTIATE_OPERATOR(name, CUDAContext); \

#define DEPLOY_CPU_CUDA(name) \
    REGISTER_CPU_OPERATOR(name, name##Op<CPUContext>); \
    REGISTER_CUDA_OPERATOR(name, name##Op<CPUContext>); \
    INSTANTIATE_OPERATOR(name, CPUContext); \

#define DEPLOY_CUDNN(name) \
    REGISTER_CUDNN_OPERATOR(name, CuDNN##name##Op<CUDAContext>); \
    INSTANTIATE_CUDNN_OPERATOR(name);

#define DEPLOY_CNML(name) \
    REGISTER_CNML_OPERATOR(name, CnML##name##Op<CNMLContext>); \
    INSTANTIATE_CNML_OPERATOR(name);

}  // namespace dragon

#endif  // DRAGON_CORE_OPERATOR_H_