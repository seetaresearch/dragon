#include "core/operator.h"
#include "core/workspace.h"
#include "utils/logging.h"

namespace dragon {

OperatorBase::OperatorBase(const OperatorDef& op_def, Workspace* ws) 
    : op_def_(op_def), ws_(ws) {
    for (auto& arg : this->op_def_.arg()) {
        CHECK_GT(arg.name().size(), 0);
        CHECK_EQ(args_.count(arg.name()), 0);
        args_[arg.name()] = &arg;
    }
    for (const std::string& input : this->op_def_.input()) {
        auto* tensor = ws->GetTensor(input);
        inputs_.push_back(tensor);
    }
    for (const std::string& output : this->op_def_.output()) {
        auto* tensor = ws->CreateTensor(output);
        outputs_.push_back(tensor);
    }
}

OperatorBase* TryCreateOperator(const string& key, const OperatorDef& op_def, Workspace* ws) {
    switch (op_def.device_option().device_type()) {
        case CPU:
            return CPUOperatorRegistry()->Create(key, op_def, ws);
        case CUDA:
            if (op_def.device_option().has_engine() && 
                op_def.device_option().engine() == "CUDNN" &&
                CUDNNOperatorRegistry()->Has(key))
                return CUDNNOperatorRegistry()->Create(key, op_def, ws);
            return CUDAOperatorRegistry()->Create(key, op_def, ws);
        default:
            LOG(FATAL) << "Unknown device type: " << op_def.device_option().device_type();
            return nullptr;
    }
}

OperatorBase* CreateOperator(const OperatorDef& op_def, Workspace* ws) {
    auto* schema = OpSchemaRegistry::Schema(op_def.type());
    CHECK(schema->Verify(op_def)) << "\nOperator failed to pass the schema checking.";
    OperatorBase* op = TryCreateOperator(op_def.type(), op_def, ws);
    return op;
}

Gradient MakeGradientForOp(const OperatorDef& def, const vector<string>& g_outputs) {
    unique_ptr<GradientMakerBase> maker(GradientRegistry()->Create(def.type(), def, g_outputs));
    if (maker.get() == nullptr) 
        LOG(FATAL) << "Gradient maker for operator " << def.type() << "not implemented.";
    Gradient grad = maker->Make();
    // copy device option, engine, and arguments if needed.
    if (maker->CopyDeviceOption() && def.has_device_option()) 
        for (auto& grad_def : grad.ops) 
            grad_def.mutable_device_option()->CopyFrom(def.device_option());
    // copy arguments if needed.
    if (maker->CopyArguments() && def.arg_size()) 
        for (auto& grad_def : grad.ops) grad_def.mutable_arg()->MergeFrom(def.arg());
    return grad;
}

template <class Context>
void Operator<Context>::ElimateCorruption() {
    Set<string> all_heads;
    queue<int> safe_heads;
    Tensor* head = ws()->GetTensor("_t_mirror_stage_head");
    string* head_data = head->mutable_data<string, CPUContext>();
    for (int i = 0; i < head->count(); i++) all_heads.insert(head_data[i]);
    //  sub-graph run
    for (int i = 0; i < InputSize(); i++) {
        if (input(i).is_corrupted())   {
            if (all_heads.count(input(i).name())) continue;
            LOG(DEBUG) << "Tensor(" << input(i).name() << ") is corrupted, recompute...  ";
            Tensor* recompute_flag = ws()->GetTensor("_t_global_recompute_flag");
            vector<OperatorBase*>& list = recompute_map()[input(i).name()];
            recompute_flag->mutable_data<bool, CPUContext>()[0] = true;
            for (int j = 0; j < list.size(); j++) list[j]->Run();
            recompute_flag->mutable_data<bool, CPUContext>()[0] = false;
        }
    }
    //  check available head
    all_heads.clear();
    for (int i = 0; i < head->count(); i++) {
        bool safe = true;
        for (int j = 0; j < InputSize(); j++) 
            if (head_data[i] == input(j).name()) safe = false;
        if (safe) safe_heads.push(i);
        all_heads.insert(head_data[i]);
    }
    //  pre-process
    for (int i = 0; i < OutputSize(); i++) {
        if (output(i)->is_corrupted()) {
            bool inplace_flag = false;
            for (int j = 0; j < InputSize(); j++)
                if (output(i)->name() == input(j).name()) inplace_flag = true;
            if (inplace_flag || all_heads.count(output(i)->name())) continue;    //  skip to use new buffer
            CHECK(!safe_heads.empty())
                << "\nAt most (" << safe_heads.size() << " [safe] / "
                << all_heads.size() << " [total] can be used for corrupted output in "
                << "(" << name() << ", " << type() << "), "
                << "\nadd WORKSPACE_MAX_CORRUPTED_SIZE for more powerful mirror stage ?";
            int idx = safe_heads.front();
            safe_heads.pop();
            Tensor* buffer = ws()->GetTensor("_t_mirror_stage_buffer_" + dragon_cast<string, int>(idx));
            output(i)->Move(buffer->memory());
            head_data[idx] = output(i)->name();
        }
    }
}

template <class Context>
void Operator<Context>::ShareGradient() {
    //  TODO(PhyscalX):  we preset input(-1)->output(0) to share
    if (output(0)->name() != "ignore") {
        Tensor* dX = ws()->GetBuffer("Grad");
        output(0)->Replace(*dX);
    }
}

template <class Context>
void Operator<Context>::MakeResource() {
    ElimateCorruption();
    if (allow_share_grads_) ShareGradient();
}

template <class Context>
void Operator<Context>::CleanResource() {
    //  post-process for mirror stage
    Map<string, int> head_to_idx;
    Tensor* head = ws()->GetTensor("_t_mirror_stage_head");
    string* head_data = head->mutable_data<string, CPUContext>();
    for (int i = 0; i < head->count(); i++) head_to_idx[head_data[i]] = i;
    for (int i = 0; i < OutputSize(); i++) {
        if (output(i)->is_corrupted() && head_to_idx.count(output(i)->name())) {
            string used = "_t_mirror_stage_buffer_" + dragon_cast<string, int>(head_to_idx[output(i)->name()]);
            Tensor* buffer = ws()->GetTensor(used);
            if (output(i)->memory() != buffer->memory()) buffer->Move(output(i)->memory());
        }
    } 
    if (allow_share_grads_) {
        //  TODO(PhyscalX):  we preset input(-1)->output(0) to share
        Tensor* dY = &input(-1);
        ws()->ReleaseBuffer(dY, "Grad");
    }
}

DEFINE_REGISTRY(CPUOperatorRegistry, OperatorBase,const OperatorDef&, Workspace*);
DEFINE_REGISTRY(CUDAOperatorRegistry, OperatorBase, const OperatorDef&, Workspace*);
DEFINE_REGISTRY(CUDNNOperatorRegistry, OperatorBase, const OperatorDef&, Workspace*);
DEFINE_REGISTRY(GradientRegistry, GradientMakerBase, const OperatorDef&, const vector<string>&);
DEFINE_REGISTRY(NoGradientRegistry, GradientMakerBase, const OperatorDef&, const vector<string>&);

#define INSTANTIATE_GET_SINGLE_ARGUMENT(T, fieldname) \
template <> T OperatorBase::GetSingleArg(const string& name, const T& default_value) { \
    if(args_.count(name) == 0) { \
        return default_value; \
    } \
    CHECK(args_[name]->has_##fieldname()); \
    return args_[name]->fieldname(); \
}

INSTANTIATE_GET_SINGLE_ARGUMENT(float, f)
INSTANTIATE_GET_SINGLE_ARGUMENT(int, i)
INSTANTIATE_GET_SINGLE_ARGUMENT(string, s)
INSTANTIATE_GET_SINGLE_ARGUMENT(bool, b);
INSTANTIATE_GET_SINGLE_ARGUMENT(int64_t, i64);


#define INSTANTIATE_GET_REPEATED_ARGUMENT(T, fieldname) \
template<> vector<T> OperatorBase::GetRepeatedArg<T>(const string& name) { \
    if(args_.count(name) == 0) return vector<T>(); \
    vector<T> values; \
    for(const auto& v : args_[name]->fieldname()) values.push_back(v); \
    return values; \
}

INSTANTIATE_GET_REPEATED_ARGUMENT(float, floats)
INSTANTIATE_GET_REPEATED_ARGUMENT(int, ints)
INSTANTIATE_GET_REPEATED_ARGUMENT(string, strings)
#undef INSTANTIATE_GET_REPEATED_ARGUMENT

template void Operator<CPUContext>::ElimateCorruption();
template void Operator<CUDAContext>::ElimateCorruption();
template void Operator<CPUContext>::ShareGradient();
template void Operator<CUDAContext>::ShareGradient();
template void Operator<CPUContext>::MakeResource();
template void Operator<CUDAContext>::MakeResource();
template void Operator<CPUContext>::CleanResource();
template void Operator<CUDAContext>::CleanResource();

}    // namespace dragon