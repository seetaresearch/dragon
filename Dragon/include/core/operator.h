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
    Tensor& Input(int idx);

    /*! \brief Return the specified output tensor */
    Tensor* Output(int idx);

    /*! \brief Return the number of inputs */
    int InputSize() { return (int)inputs_.size(); }

    /*! \brief Return the number of outputs */
    int OutputSize() { return (int)outputs_.size(); }

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

    /*! \brief Return the mount name in this operator */
    const string mount_name(const string& name) const {
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

    typedef Map<string, vector<OperatorBase*> > SubGraph;

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

    /*! \brief Return the debug DType string on given tensor */
    string DTypeHelper(
        const Tensor&           tensor,
        const Set<string>&      dtypes) const;

    /* \brief Return the debug DType string on given type */
    string DTypeHelper(
        const string&           dtype,
        const Set<string>&      dtypes) const;

 protected:
    string phase_, anchor_;
    Map<std::string, const Argument*> args_;
    SubGraph subgraph_;
    vector<Tensor*> inputs_, outputs_;
    OperatorDef def_;
    Workspace* ws_;
};

template <class Context>
class Operator : public OperatorBase {
 public:
    /*! \brief Default constructor */
    Operator(const OperatorDef& def, Workspace* ws)
        : OperatorBase(def, ws), ctx_(def.device_option()),
          allow_recomputing_(OperatorBase::Arg<bool>(
              "allow_recomputing", false)),
          do_sync_(OperatorBase::Arg<bool>(
              "do_sync", false)) {
        allow_run_ = true;
        allow_run_ &= MPICheck();
        allow_run_ &= (!(OutputSize() == 1 &&
            Output(0)->name() == "ignore"));
    }

    /*! \brief Run this operator on the specified stream */
    void Run(int stream_id = 0) final {
        if (!allow_run_) return;
        if (allow_recomputing_) PrepareResource();
        ctx()->SwitchToDevice(stream_id);
        MemorySwitch();
        RunOnDevice();
        if (do_sync_ || stream_id > 0) {
            // We will sync the stream 0 at the specific time
            ctx()->FinishDeviceCompution();
        }
        if (allow_recomputing_) ReleaseResource();
    }

    /*! \brief Prepare the content of inputs */
    virtual void PrepareResource();

    /*! \brief Release the ownership of inputs */
    virtual void ReleaseResource();

    /*! \brief Coordinate the context of inputs and outputs */
    virtual void MemorySwitch() {
        for (auto* e : inputs_)
            if(e->name() != "ignore")
                e->SwitchToDevice(ctx()->device_id());
        for (auto* e : outputs_)
            if(e->name() != "ignore")
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
    bool allow_run_, allow_recomputing_, do_sync_;

 private:
    /*! \brief Check the MPI conditions */
    bool MPICheck() {
#ifndef WITH_MPI
        return true;
#else
        vector<int> allow_ranks =
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

/*! Macros */

#define USE_SIMPLE_CTOR_DTOR(name) \
    name(const OperatorDef& def, Workspace* ws) \
        : Operator<Context>(def, ws) {} \
    virtual ~name() {}

#define USE_OPERATOR_BASE_FUNCTIONS \
    using OperatorBase::Input; \
    using OperatorBase::Output; \
    using OperatorBase::ws; \
    using OperatorBase::name; \
    using OperatorBase::type; \
    using OperatorBase::phase; \
    using OperatorBase::anchor; \
    using OperatorBase::mount_name; \
    using OperatorBase::def; \
    using OperatorBase::InputSize; \
    using OperatorBase::OutputSize; \
    using OperatorBase::DebugString; \
    using OperatorBase::DTypeHelper; \
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

/*! NVIDIA's Accelerated Library - CUDNN */

DECLARE_REGISTRY(
    CUDNNOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

/*! CAMBRICON's Accelerated Library - CNML */

DECLARE_REGISTRY(
    CNMLOperatorRegistry,
    OperatorBase,
    const OperatorDef&,
    Workspace*);

#define TENSOR_FILL_WITH_TYPE(tensor, shape, type) \
    if (tensor.count() == 0) { \
        CHECK(ws()->GetFiller(tensor.name())) \
            << "\nTensor(" << tensor.name() << ") is empty. \n" \
            << "may be specify a filler for it ?"; \
        tensor.Reshape(shape); \
        unique_ptr< Filler<type, Context> > filler(  \
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
        unique_ptr< Filler<T, Context> > filler(  \
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

#define DECLARE_ARGUMENT_WITH_DESC(type, argument) \
    type argument##_value; \
    string argument##_desc; \
    type argument()

#define DECLARE_ARGUMENTS_WITH_DESC(type, argument) \
    vector<type> argument##_value; \
    vector<string> argument##_desc; \
    type argument(int idx)

#define GET_ARGUMENT_WITH_DESC(type, argument, default_value) \
    argument##_value = OperatorBase::Arg<type>(#argument, default_value); \
    argument##_desc = OperatorBase::Arg<string>(string(#argument) + "_desc", "")

#define GET_ARGUMENTS_WITH_DESC(type, argument) \
    argument##_value = OperatorBase::Args<type>(#argument); \
    argument##_desc = OperatorBase::Args<string>(string(#argument) + "_desc")

#define DEFINE_ARGUMENT_WITH_DESC(type, classname, argument) \
    template <class Context> \
    type classname<Context>::argument() { \
        if (argument##_desc.empty()) return argument##_value; \
        Tensor* argument##_tensor = ws()->GetTensor(argument##_desc); \
        CHECK(argument##_tensor->IsType<type>()) \
            << "\nThe type of " << #argument << " should be " << #type << "."; \
        CHECK_EQ(argument##_tensor->count(), 1) \
            << "\nThe argument of " << #argument << " should be a scalar."; \
        return argument##_tensor->template data<type, CPUContext>()[0]; \
    }

#define DEFINE_ARGUMENTS_WITH_DESC(type, classname, argument) \
    template <class Context> \
    type classname<Context>::argument(int idx) { \
        if (argument##_desc.empty()) { \
            CHECK_LT(idx, argument##_value.size()) \
                << "\nExcepted the size of " << #argument \
                << " > " << idx << ". (Got " \
                << argument##_value.size() << ")."; \
            return argument##_value[idx]; \
        } \
        CHECK_LT(idx, argument##_desc.size()) \
            << "\nExcepted the size of " << #argument \
            << " > " << idx << ". (Got " \
            << argument##_desc.size() << ")."; \
        Tensor* argument##_tensor = ws()->GetTensor( \
            str::replace_first(argument##_desc[idx], \
                "${ANCHOR}", anchor())); \
        CHECK(argument##_tensor->IsType<type>()) \
            << "\nThe type of " << #argument << " should be " << #type << "."; \
        CHECK_EQ(argument##_tensor->count(), 1) \
            << "\nThe argument of " << #argument << " at pos(" \
            << idx << ") should be a scalar."; \
        return argument##_tensor->template data<type, CPUContext>()[0]; \
    }

#define GET_ARGUMENTS_SIZE(argument) \
    (int)std::max(argument##_value.size(), argument##_desc.size())

#define XIsType(x, dtype) \
    x.template IsType<dtype>()

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