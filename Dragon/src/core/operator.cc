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
            LOG(FATAL) << "unknown device type: " << op_def.device_option().device_type();
            return nullptr;
    }
}

OperatorBase* CreateOperator(const OperatorDef& op_def, Workspace* ws) {
    auto* schema = OpSchemaRegistry::Schema(op_def.type());
    CHECK(schema->Verify(op_def)) << "operator failed to pass the schema checking.";
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

DEFINE_REGISTRY(CPUOperatorRegistry, OperatorBase,const OperatorDef&, Workspace*);
DEFINE_REGISTRY(CUDAOperatorRegistry, OperatorBase, const OperatorDef&, Workspace*);
DEFINE_REGISTRY(CUDNNOperatorRegistry, OperatorBase, const OperatorDef&, Workspace*);
DEFINE_REGISTRY(GradientRegistry, GradientMakerBase, const OperatorDef&, const vector<string>&);
DEFINE_REGISTRY(NoGradientRegistry, GradientMakerBase, const OperatorDef&, const vector<string>&);

#define INSTANTIATE_GET_SINGLE_ARGUMENT(T, fieldname) \
template <> T OperatorBase::GetSingleArg(const string& name, const T& default_value){ \
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
template<> vector<T> OperatorBase::GetRepeatedArg<T>(const string& name){ \
    if(args_.count(name) == 0) return vector<T>(); \
    vector<T> values; \
    for(const auto& v : args_[name]->fieldname()) values.push_back(v); \
    return values; \
}

INSTANTIATE_GET_REPEATED_ARGUMENT(float, floats)
INSTANTIATE_GET_REPEATED_ARGUMENT(int, ints)
INSTANTIATE_GET_REPEATED_ARGUMENT(string, strings)
#undef INSTANTIATE_GET_REPEATED_ARGUMENT

}    // namespace dragon