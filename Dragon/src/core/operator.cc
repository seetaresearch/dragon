#include "core/operator.h"
#include "core/workspace.h"
#include "utils/logging.h"

namespace dragon {

OperatorBase::OperatorBase(
    const OperatorDef&          def,
    Workspace*                  ws)
        : def_(def), ws_(ws), anchor_(def.name()) {
    for (auto& arg : def_.arg()) {
        CHECK_GT(arg.name().size(), 0);
        CHECK_EQ(args_.count(arg.name()), 0);
        args_[arg.name()] = &arg;
        if (arg.name() == "anchor") anchor_ = arg.s();
    }
    for (auto& input : def.input()) {
        auto* tensor = ws->GetTensor(input);
        inputs_.push_back(tensor);
    }
    for (auto& output : def.output()) {
        auto* tensor = ws->CreateTensor(output);
        outputs_.push_back(tensor);
    }
}

inline Tensor& OperatorBase::Input(int idx) {
    CHECK_LT(idx, (int)inputs_.size());
    CHECK_GE(idx, -(int)inputs_.size());
    if (idx >= 0) return *inputs_[idx];
    else return *inputs_[idx + inputs_.size()];
}

inline Tensor* OperatorBase::Output(int idx) {
    CHECK_LT(idx, (int)outputs_.size());
    CHECK_GE(idx, -(int)outputs_.size());
    if (idx >= 0) return outputs_[idx];
    else return outputs_[idx + outputs_.size()];
}

string OperatorBase::DTypeHelper(
    const Tensor&               tensor,
    const Set<string>&          dtypes) const {
    std::stringstream ss;
    ss << "Unsupported DType of Input(" << tensor.name() << "): "
       << TypeMetaToString(tensor.meta()) << "\n";
    ss << "<" << type() << "Op>" << " supports the following dtypes: {\n";
    for (auto& dtype : dtypes) ss << "    * " << dtype << ",\n";
    ss << "}";
    return ss.str();
}

string OperatorBase::DTypeHelper(
    const string&               dtype,
    const Set<string>&          dtypes) const {
    std::stringstream ss;
    ss << "Unsupported DType: " << dtype << "\n";
    ss << "<" << type() << "Op>" << " supports the following dtypes: {\n";
    for (auto& dtype : dtypes) ss << "    * " << dtype << ",\n";
    ss << "}";
    return ss.str();
}

OperatorBase* TryCreateOperator(
    const string&               key,
    const OperatorDef&          def,
    Workspace*                  ws) {
    switch (def.device_option().device_type()) {
        case CPU:
            return CPUOperatorRegistry()->Create(key, def, ws);
        case CUDA:
            if (def.device_option().has_engine() &&
                def.device_option().engine() == "CUDNN" &&
                CUDNNOperatorRegistry()->Has(key))
                return CUDNNOperatorRegistry()->Create(key, def, ws);
            return CUDAOperatorRegistry()->Create(key, def, ws);
        case CNML:
            return CNMLOperatorRegistry()->Create(key, def, ws);
        default:
            LOG(FATAL) << "Unknown device type: "
                       << def.device_option().device_type();
            return nullptr;
    }
}

void OperatorBase::MutableOp(
    const vector<string>&       inputs,
    const vector<string>&       outputs,
    const string&               anchor) {
    inputs_.resize(inputs.size());
    outputs_.resize(outputs.size());
    for (int i = 0; i < inputs_.size(); i++)
        inputs_[i] = ws()->GetTensor(inputs[i]);
    for (int i = 0; i < outputs_.size(); i++)
        outputs_[i] = ws()->CreateTensor(outputs[i]);
    anchor_ = anchor;
}

void OperatorBase::MutableOp(const OperatorDef& def) {
    inputs_.resize(def.input_size());
    outputs_.resize(def.output_size());
    for (int i = 0; i < inputs_.size(); i++)
        inputs_[i] = ws()->GetTensor(def.input(i));
    for (int i = 0; i < outputs_.size(); i++)
        outputs_[i] = ws()->CreateTensor(def.output(i));
    anchor_ = def.name();
    for (auto& arg : def.arg())
        if (arg.name() == "anchor") anchor_ = arg.s();
}

OperatorBase* CreateOperator(
    const OperatorDef&          def,
    Workspace*                  ws) {
    auto* schema = OpSchemaRegistry::Schema(def.type());
    CHECK(schema->Verify(def))
        << "\nOperator failed to pass the schema checking.";
    OperatorDef mutable_def(def);
    // Heuristically makes each random seed slightly differnet
    static unsigned int op_seed_uuid = 0;
    mutable_def.mutable_device_option()->set_random_seed(
        op_seed_uuid + def.device_option().random_seed());
    op_seed_uuid = (op_seed_uuid + 1) % UINT32_MAX;
    return TryCreateOperator(def.type(), mutable_def, ws);
}

Gradient MakeGradientForOp(
    const OperatorDef&          def,
    const vector<string>&       g_outputs) {
    unique_ptr<GradientMakerBase> maker(
        GradientRegistry()->Create(def.type(), def, g_outputs));
    if (maker.get() == nullptr)
        LOG(FATAL) << "Gradient maker for operator "
                   << def.type() << "not implemented.";
    Gradient grad = maker->Make();
    OperatorDef reference_def(def);
    // Custom arguments
    for (int i = 0; i < reference_def.arg_size(); i++) {
        if (reference_def.arg(i).name() == "persistent_key") {
            string s = reference_def.arg(i).s();
            if (!s.empty()) *reference_def.mutable_arg(i)
                ->mutable_s() = s + "/grad";
        }
    }
    // Copy device option, engine, and arguments
    if (maker->CopyDeviceOption() && def.has_device_option())
        for (auto& grad_def : grad.ops)
            grad_def.mutable_device_option()->CopyFrom(
                def.device_option());
    // Copy arguments
    if (maker->CopyArguments() && def.arg_size())
        for (auto& grad_def : grad.ops)
            grad_def.mutable_arg()->MergeFrom(
                reference_def.arg());
    return grad;
}

template <class Context>
void Operator<Context>::ElimateCorruption() {
    Set<string> all_heads;
    queue<int> safe_heads;
    Tensor* head = ws()->GetTensor("/opt/mirror_stage/head");
    string* head_data = head->mutable_data<string, CPUContext>();
    for (int i = 0; i < head->count(); i++)
        all_heads.insert(head_data[i]);
    // Sub-graph run
    for (int i = 0; i < InputSize(); i++) {
        if (Input(i).is_corrupted())   {
            if (all_heads.count(Input(i).name())) continue;
            LOG(DEBUG) << "Tensor(" << Input(i).name()
                       << ") is corrupted, recompute...  ";
            Tensor* recompute_flag = ws()->GetTensor(
                "/opt/mirror_stage/recompute_flag");
            vector<OperatorBase*>& list = recompute_map()[Input(i).name()];
            recompute_flag->mutable_data<bool, CPUContext>()[0] = true;
            for (int j = 0; j < list.size(); j++) list[j]->Run();
            recompute_flag->mutable_data<bool, CPUContext>()[0] = false;
        }
    }
    // Check available head
    all_heads.clear();
    for (int i = 0; i < head->count(); i++) {
        bool safe = true;
        for (int j = 0; j < InputSize(); j++)
            if (head_data[i] == Input(j).name()) safe = false;
        if (safe) safe_heads.push(i);
        all_heads.insert(head_data[i]);
    }
    // Pre-process
    for (int i = 0; i < OutputSize(); i++) {
        if (Output(i)->is_corrupted()) {
            bool inplace_flag = false;
            for (int j = 0; j < InputSize(); j++)
                if (Output(i)->name() == Input(j).name()) inplace_flag = true;
            // Skip to use new buffer
            if (inplace_flag || all_heads.count(Output(i)->name())) continue;
            CHECK(!safe_heads.empty())
                << "\nAt most (" << safe_heads.size() << " [safe] / "
                << all_heads.size() << " [total] can be used for corrupted output in "
                << "(" << name() << ", " << type() << "), "
                << "\nadd WORKSPACE_MAX_CORRUPTED_SIZE for more powerful mirror stage ?";
            int idx = safe_heads.front();
            safe_heads.pop();
            Tensor* buffer = ws()->GetTensor(
                "/opt/mirror_stage/buffer_" 
                    + std::to_string(idx));
            Output(i)->Move(buffer->memory());
            head_data[idx] = Output(i)->name();
        }
    }
}

template <class Context>
void Operator<Context>::MakeResource() {
    ElimateCorruption();
}

template <class Context>
void Operator<Context>::CleanResource() {
    // Post-process for mirror stage
    Map<string, int> head_to_idx;
    Tensor* head = ws()->GetTensor("/opt/mirror_stage/head");
    string* head_data = head->mutable_data<string, CPUContext>();
    for (int i = 0; i < head->count(); i++) head_to_idx[head_data[i]] = i;
    for (int i = 0; i < OutputSize(); i++) {
        if (Output(i)->is_corrupted() &&
                head_to_idx.count(Output(i)->name())) {
            string used = "/opt/mirror_stage/buffer_" 
                + std::to_string(head_to_idx[Output(i)->name()]);
            Tensor* buffer = ws()->GetTensor(used);
            if (Output(i)->memory() != buffer->memory())
                buffer->Move(Output(i)->memory());
        }
    }
}

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

#define INSTANTIATE_GET_SINGLE_ARGUMENT(T, fieldname) \
template <> T OperatorBase::Arg( \
    const string& name, \
    const T& default_value) { \
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
template<> vector<T> OperatorBase::Args<T>(const string& name) { \
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
template void Operator<CNMLContext>::ElimateCorruption();
template void Operator<CPUContext>::MakeResource();
template void Operator<CUDAContext>::MakeResource();
template void Operator<CNMLContext>::MakeResource();
template void Operator<CPUContext>::CleanResource();
template void Operator<CUDAContext>::CleanResource();
template void Operator<CNMLContext>::CleanResource();

}  // namespace dragon