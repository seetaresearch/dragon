// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

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

class OperatorBase{
 public:
    OperatorBase(const OperatorDef& op_def, Workspace* ws);

    inline Tensor& input(int idx) {
        CHECK_LT(idx, (int)inputs_.size());
        CHECK_GE(idx, -(int)inputs_.size());
        if (idx >= 0) return *inputs_[idx];
        else return *inputs_[idx + inputs_.size()];
    }

    inline Tensor* output(int idx) {
        CHECK_LT(idx, (int)outputs_.size());
        CHECK_GE(idx, -(int)outputs_.size());
        if (idx >= 0) return outputs_[idx];
        else return outputs_[idx + outputs_.size()];
    }

    inline size_t InputSize() { return inputs_.size(); }
    inline size_t OutputSize() { return outputs_.size(); }

    inline void SwitchToPhase(const string& phase) { this->phase_ = phase; }
    virtual void Run() { NOT_IMPLEMENTED; }


    inline const string& name() const { return op_def_.name(); }
    inline const string& type() const { return op_def_.type(); }
    inline const string& phase() const { return phase_; }
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
    inline const string debug_string() const { return op_def_.DebugString(); }

 protected:
    string phase_;
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
        : OperatorBase(op_def, ws), ctx_(op_def.device_option()) {
        allow_run_ = true;
        allow_run_ &= _MPICheck();
        allow_run_ &= (!(OutputSize() == 1 && output(0)->name() == "ignore"));
        allow_share_grads_ = (!op_def.debug_mode());
        allow_share_grads_ &= op_def.share_grads(); 
        allow_share_grads_ &= (type().find("Gradient") != string::npos);
    }

    virtual void Run() final {
        if (!allow_run_)  return;
        MakeResource();
        ctx_.SwitchToDevice();
        MemorySwitch();
        RunOnDevice();
        ctx_.FinishDeviceCompution();
        CleanResource();
    }

    virtual void ElimateCorruption();
    virtual void ShareGradient();

    virtual void MakeResource();
    virtual void CleanResource();

    void MemorySwitch() {
        for (int i = 0; i < InputSize(); i++)
            if (input(i).name() != "ignore") input(i).SwitchToDevice();
        for (int i = 0; i < OutputSize(); i++)
            if (output(i)->name() != "ignore") output(i)->SwitchToDevice();
    }

    virtual void RunOnDevice() = 0;

    inline Context& ctx() { return ctx_; }
    inline string anchor() { return GetSingleArg("anchor", name()); }
    inline bool allow_run() { return allow_run_; }

 protected:
    Context ctx_;
    bool allow_run_, allow_share_grads_;

 private:
    bool _MPICheck() {
#ifndef WITH_MPI
        return true;
#else
        vector<int> allow_ranks = Operator::GetRepeatedArg<int>("mpi_rank");
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

DECLARE_REGISTRY(CPUOperatorRegistry, OperatorBase,const OperatorDef&, Workspace*);
DECLARE_REGISTRY(CUDAOperatorRegistry, OperatorBase, const OperatorDef&, Workspace*);
DECLARE_REGISTRY(CUDNNOperatorRegistry, OperatorBase, const OperatorDef&, Workspace*);

#define  TENSOR_FILL(tensor, shape) \
    if (tensor.count() == 0) { \
        CHECK(ws()->GetFiller(tensor.name())) \
            << "Tensor(" << tensor.name() << ") is empty. \n" \
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
    ptr_tensor = ws()->CreateTensor("_t_multiplier"); \
    if (size > ptr_tensor->count()) { \
        ptr_tensor->Reshape(vector<TIndex>(1, size)); \
        math::Set<T, Context>(size, dragon_cast<T, float>(1.0f), \
            ptr_tensor->template mutable_data<T, Context>()); \
    } \
  }

#define DISABLE_SHARE_GRADIENT   \
    this->allow_share_grads_ = false

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
    REGISTER_CUDA_OPERATOR(name, name##Op<CPUContext>); \
    INSTANTIATE_OPERATOR(name, CPUContext); \

#define DEPLOY_CUDNN(name) \
    REGISTER_CUDNN_OPERATOR(name, CuDNN##name##Op<CUDAContext>); \
    INSTANTIATE_CUDNN_OPERATOR(name);
}    // namespace dragon

#endif    // DRAGON_CORE_OPERATOR_H_