// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_CORE_OPERATOR_H_
#define DRAGON_CORE_OPERATOR_H_

#include "core/registry.h"
#include "core/context.h"
#include "core/tensor.h"
#include "core/operator_gradient.h"
#include "core/operator_schema.h"
#include "utils/cast.h"

#ifdef WITH_MPI
#include <mpi/mpi.h>
#endif

namespace dragon {

class Workspace;

class OperatorBase {
 public:
    OperatorBase(const OperatorDef& op_def, Workspace* ws);
    virtual ~OperatorBase() {}

    Tensor& Input(int idx);
    Tensor* Output(int idx);

    inline size_t InputSize() { return inputs_.size(); }
    inline size_t OutputSize() { return outputs_.size(); }

    void MutableOp(const OperatorDef& op_def);
    void MutableOp(const vector<string>& inputs,
                   const vector<string>& outputs,
                   const string& anchor);

    inline void SwitchToPhase(const string& phase) { this->phase_ = phase; }
    virtual void Run() { NOT_IMPLEMENTED; }

    inline const string& name() const { return op_def_.name(); }
    inline const string& type() const { return op_def_.type(); }
    inline const string& phase() const { return phase_; }
    inline const string& anchor() { return anchor_; }
    inline Workspace* ws() const { return ws_; }

    template <typename T>
    T GetSingleArg(const string& name, const T& default_value);

    template <typename T>
    vector<T> GetRepeatedArg(const string& name);

    inline const Map<std::string, const Argument*>& args() { return args_; }
    inline const Argument& arg(const string& name) { return *(args_[name]); }

    typedef Map<string, vector<OperatorBase*> > RecomputeMap;
    inline RecomputeMap& recompute_map() { return recompute_map_; }
    void set_recompute_map(RecomputeMap recompute_map) { recompute_map_ = recompute_map; }

    inline const OperatorDef& op_def() const { return op_def_; }
    inline string DebugString() const { return op_def_.DebugString(); }
    string DTypeHelper(const Tensor& tensor, const Set<string>& dtypes) const;
    string DTypeHelper(const string& dtype, const Set<string>& dtypes) const;

 protected:
    string phase_, anchor_;
    Map<std::string, const Argument*> args_;
    Map<string, vector<OperatorBase*> > recompute_map_;
    vector<Tensor*> inputs_, outputs_;
    OperatorDef op_def_;
    Workspace* ws_;
};

template <class Context>
class Operator : public OperatorBase {
 public:
    Operator(const OperatorDef& op_def, Workspace* ws)
        : OperatorBase(op_def, ws), ctx_(op_def.device_option()),
          do_synchronize_(Operator::GetSingleArg<bool>("do_synchronize", false)),
          recomputing_aware_(Operator::GetSingleArg<bool>("recomputing_aware", false)) {
        allow_run_ = true;
        allow_run_ &= _MPICheck();
        allow_run_ &= (!(OutputSize() == 1 && Output(0)->name() == "ignore"));
    }

    virtual void Run() final {
        if (!allow_run_)  return;
        if (recomputing_aware_) MakeResource();
        ctx_.SwitchToDevice();
        MemorySwitch();
        RunOnDevice();
        if (do_synchronize_) ctx_.FinishDeviceCompution();
        if (recomputing_aware_) CleanResource();
    }

    virtual void ElimateCorruption();
    virtual void MakeResource();
    virtual void CleanResource();

    void MemorySwitch() {
        for (auto* I : inputs_) if(I->name() != "ignore") I->SwitchToDevice();
        for (auto* O : outputs_) if(O->name() != "ignore") O->SwitchToDevice();
    }

    virtual void RunOnDevice() = 0;

    inline Context& ctx() { return ctx_; }
    inline bool AllowRun() { return allow_run_; }

 protected:
    Context ctx_;
    bool allow_run_, recomputing_aware_, do_synchronize_;

 private:
    bool _MPICheck() {
#ifndef WITH_MPI
        return true;
#else
        vector<int> allow_ranks = Operator::GetRepeatedArg<int>("mpi_ranks");
        if (allow_ranks.empty()) return true;
        int cur_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &cur_rank);
        for (auto mpi_rank : allow_ranks)
            if (cur_rank == mpi_rank) return true;
        return false;
#endif
    }
};

OperatorBase* CreateOperator(const OperatorDef& op_def, Workspace* ws);

#define USE_SIMPLE_CTOR_DTOR(name) \
    name(const OperatorDef& op_def, Workspace* ws) \
        : Operator<Context>(op_def, ws) {} \
    virtual ~name() {}

#define USE_OPERATOR_BASE_FUNCTIONS \
    using OperatorBase::Input; \
    using OperatorBase::Output; \
    using OperatorBase::ws; \
    using OperatorBase::name; \
    using OperatorBase::type; \
    using OperatorBase::phase; \
    using OperatorBase::anchor; \
    using OperatorBase::op_def; \
    using OperatorBase::InputSize; \
    using OperatorBase::OutputSize; \
    using OperatorBase::DebugString; \
    using OperatorBase::DTypeHelper; \
    using OperatorBase::SwitchToPhase

#define USE_OPERATOR_FUNCTIONS \
    USE_OPERATOR_BASE_FUNCTIONS; \
    using Operator<Context>::ctx; \
    using Operator<Context>::AllowRun

DECLARE_REGISTRY(CPUOperatorRegistry, OperatorBase,const OperatorDef&, Workspace*);
DECLARE_REGISTRY(CUDAOperatorRegistry, OperatorBase, const OperatorDef&, Workspace*);
DECLARE_REGISTRY(CUDNNOperatorRegistry, OperatorBase, const OperatorDef&, Workspace*);

#define TENSOR_FILL(tensor, shape) \
    if (tensor.count() == 0) { \
        CHECK(ws()->GetFiller(tensor.name())) \
            << "\nTensor(" << tensor.name() << ") is empty. \n" \
            << "may be specify a filler for it ?"; \
        tensor.Reshape(shape); \
        unique_ptr< Filler<T, Context> > filler(  \
            CreateFiller<T, Context>(*ws()->GetFiller(tensor.name()))); \
        filler->Fill(&tensor); \
    } else { \
         TIndex count = 1; \
         for(int i = 0; i < shape.size(); i++) count *= shape[i]; \
         CHECK_EQ(count, tensor.count()) \
            << "\nModel request " << "Tensor(" << tensor.name() << ")'s " \
            << "size is " << count << ", \n" \
            << "but now is " << tensor.count() << ", " \
            << "did you feed the incorrect Tensor before ?"; \
        tensor.Reshape(shape); \
    }

#define INIT_MULTIPLIER(ptr_tensor, size) { \
    ptr_tensor = ws()->CreateTensor("/share/multiplier"); \
    if (size > ptr_tensor->count()) { \
        ptr_tensor->Reshape(vector<TIndex>(1, size)); \
        math::Set<T, Context>(size, dragon_cast<T, float>(1.f), \
            ptr_tensor->template mutable_data<T, Context>()); \
    } \
  }

#define DECLARE_MULTIPLIER(name, size) \
    const T* name; \
    { \
        Tensor* _auto_multiplier_; \
        INIT_MULTIPLIER(_auto_multiplier_, size); \
        name = _auto_multiplier_->template data<T, Context>(); \
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
    argument##_value = OperatorBase::GetSingleArg<type>(#argument, default_value); \
    argument##_desc = OperatorBase::GetSingleArg<string>(string(#argument) + "_desc", "")

#define GET_ARGUMENTS_WITH_DESC(type, argument) \
    argument##_value = OperatorBase::GetRepeatedArg<type>(#argument); \
    argument##_desc = OperatorBase::GetRepeatedArg<string>(string(#argument) + "_desc")

#define DEFINE_ARGUMENT_WITH_DESC(type, classname, argument) \
    template <class Context> \
    type classname<Context>::argument() { \
        if (argument##_desc.empty()) return argument##_value; \
        Tensor* argument##_tensor = ws()->GetTensor(argument##_desc); \
        CHECK(argument##_tensor->IsType<type>()) \
            << "\nThe type of " << #argument << " should be " << #type << "."; \
        CHECK_EQ(argument##_tensor->count(), 1) \
            << "\nThe argument of " << #argument << " should be a scalar"; \
        return argument##_tensor->template data<type, CPUContext>()[0]; \
    }

#define DEFINE_ARGUMENTS_WITH_DESC(type, classname, argument) \
    template <class Context> \
    type classname<Context>::argument(int idx) { \
        if (argument##_desc.empty()) { \
            CHECK_LT(idx, argument##_value.size()); \
            return argument##_value[idx]; \
        } \
        CHECK_LT(idx, argument##_desc.size()); \
        Tensor* argument##_tensor = ws()->GetTensor(argument##_desc[idx]); \
        CHECK(argument##_tensor->IsType<type>()) \
            << "\nThe type of " << #argument << " should be " << #type; \
        CHECK_EQ(argument##_tensor->count(), 1) \
            << "\nThe argument of " << #argument << " at pos(" \
            << idx << ") should be a scalar."; \
        return argument##_tensor->template data<type, CPUContext>()[0]; \
    }

#define GET_ARGUMENTS_SIZE(argument) \
    std::max(argument##_value.size(), argument##_desc.size())

#define XIsType(x, dtype) \
    x.template IsType<dtype>()

#define INSTANTIATE_OPERATOR(name, context) \
  template class name##Op<context>;

#define INSTANTIATE_CUDNN_OPERATOR(name) \
  template class CuDNN##name##Op<CUDAContext>;

#define REGISTER_CPU_OPERATOR(name, ...) \
    REGISTER_CLASS(CPUOperatorRegistry, name, __VA_ARGS__)

#define REGISTER_CUDA_OPERATOR(name, ...) \
    REGISTER_CLASS(CUDAOperatorRegistry, name, __VA_ARGS__)

#define REGISTER_CUDNN_OPERATOR(name, ...) \
    REGISTER_CLASS(CUDNNOperatorRegistry, name, __VA_ARGS__)

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

}    // namespace dragon

#endif    // DRAGON_CORE_OPERATOR_H_