#include "core/operator.h"
#include "core/workspace.h"
#include "utils/logging.h"

namespace dragon {

/*! Default constructor of <OperatorBase> */

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
    string tensor_name; size_t ver_pos;
    for (auto& input : def.input()) {
        tensor_name = input;
        if ((ver_pos = input.find("/ver:")) != string::npos)
            tensor_name = input.substr(0, ver_pos);
        auto* tensor = ws->GetTensor(tensor_name);
        inputs_.push_back(tensor);
    }
    for (auto& output : def.output()) {
        tensor_name = output;
        if ((ver_pos = output.find("/ver:")) != string::npos)
            tensor_name = output.substr(0, ver_pos);
        auto* tensor = ws->CreateTensor(tensor_name);
        outputs_.push_back(tensor);
    }
}

/*! Return the specified input tensor */

Tensor& OperatorBase::Input(int idx) {
    CHECK_LT(idx, (int)inputs_.size());
    CHECK_GE(idx, -(int)inputs_.size());
    if (idx >= 0) return *inputs_[idx];
    else return *inputs_[idx + inputs_.size()];
}

/*! Return the specified output tensor */

Tensor* OperatorBase::Output(int idx) {
    CHECK_LT(idx, (int)outputs_.size());
    CHECK_GE(idx, -(int)outputs_.size());
    if (idx >= 0) return outputs_[idx];
    else return outputs_[idx + outputs_.size()];
}

/*! Return the debug DType string on given tensor */

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

/* Return the debug DType string on given type */

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

/*! Modify this operator according to the given def  */

void OperatorBase::UpdateFrom(const OperatorDef& def) {
    anchor_ = def.name();
    inputs_.resize(def.input_size());
    outputs_.resize(def.output_size());
    for (int i = 0; i < inputs_.size(); i++)
        inputs_[i] = ws()->GetTensor(def.input(i));
    for (int i = 0; i < outputs_.size(); i++)
        outputs_[i] = ws()->CreateTensor(def.output(i));
}

/*! Create a operator instance from the factory  */

OperatorBase* TryCreateOperator(
    const string&               key,
    const OperatorDef&          def,
    Workspace*                  ws) {
    switch (def.device_option().device_type()) {
        case PROTO_CPU:
            return CPUOperatorRegistry()->Create(key, def, ws);
        case PROTO_CUDA:
            if (def.device_option().has_engine() &&
                def.device_option().engine() == "CUDNN" &&
                CUDNNOperatorRegistry()->Has(key))
                return CUDNNOperatorRegistry()->Create(key, def, ws);
            return CUDAOperatorRegistry()->Create(key, def, ws);
        case PROTO_CNML:
            return CNMLOperatorRegistry()->Create(key, def, ws);
        default:
            LOG(FATAL) << "Unknown device type: "
                       << def.device_option().device_type();
            return nullptr;
    }
}

/*! New a operator from the raw def */

OperatorBase* NewOperator(
    const OperatorDef&          def,
    Workspace*                  ws) {
    auto* schema = OpSchemaRegistry::Schema(def.type());
    if (schema) {
        // Check the Inputs and Outputs if necessary
        CHECK(schema->Verify(def))
            << "\nOperator failed to pass the schema checking.";
    }
    OperatorDef mutable_def(def);
    // Heuristically make each random seed slightly different
    static unsigned int op_seed_uuid = 0;
    mutable_def.mutable_device_option()->set_random_seed(
        op_seed_uuid + def.device_option().random_seed());
    op_seed_uuid = (op_seed_uuid + 1) % UINT32_MAX;
    return TryCreateOperator(def.type(), mutable_def, ws);
}

/*! Make the gradient for the given def */

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
    // Map the UID
    if (reference_def.has_uid()) {
        for (int i = 0; i < grad.ops.size(); ++i) {
            grad.ops[i].set_uid(
                reference_def.uid() + "/grad" +
                    (i > 0 ? (":" + std::to_string(i)) : "")
            );
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

/*! Prepare the content of inputs */

template <class Context>
void Operator<Context>::PrepareResource() {
    string tensor_name;
    size_t ver_pos; int version;
    for (int i = 0; i < InputSize(); i++) {
        if (Input(i).version() >= 0) {
            tensor_name = def().input(i);
            ver_pos = tensor_name.find("/ver:");
            version = std::atoi(tensor_name.substr(ver_pos + 5).c_str());
            if (version == Input(i).version()) continue;
            LOG(DEBUG) << "Excepted version of Tensor(" + Input(i).name() + ") "
                       << "is " << version << ", got " << Input(i).version()
                       << ". Recompute.";
            Tensor* flag = ws()->GetTensor("/opt/recomputing_flag");
            flag->mutable_data<bool, CPUContext>()[0] = true;
            vector<OperatorBase*>& chain = subgraph()[tensor_name];
            for (auto* op : chain) op->Run(ctx()->stream_id());
            flag->mutable_data<bool, CPUContext>()[0] = false;
        }
    }
}

/*! Release the ownership of inputs */

template <class Context>
void Operator<Context>::ReleaseResource() {
    string tensor_name;
    size_t ver_pos; int version;
    for (int i = 0; i < OutputSize(); i++) {
        if (Output(i)->version() >= 0) {
            tensor_name = def().output(i);
            ver_pos = tensor_name.find("/ver:");
            version = std::atoi(tensor_name.substr(ver_pos + 5).c_str());
            Output(i)->set_version(version);
        }
    }
}

/*! Operator Registry */

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

/*! Macros */

#define INSTANTIATE_GET_SINGLE_ARGUMENT(T, fieldname) \
template <> T OperatorBase::Arg( \
    const string& name, \
    const T& default_value) { \
    if (args_.count(name) == 0) { \
        return default_value; \
    } \
    CHECK(args_[name]->has_##fieldname()); \
    return static_cast<T>(args_[name]->fieldname()); \
}

INSTANTIATE_GET_SINGLE_ARGUMENT(float, f)
INSTANTIATE_GET_SINGLE_ARGUMENT(int, i)
INSTANTIATE_GET_SINGLE_ARGUMENT(bool, i)
INSTANTIATE_GET_SINGLE_ARGUMENT(int64_t, i)
INSTANTIATE_GET_SINGLE_ARGUMENT(string, s)
#undef INSTANTIATE_GET_SINGLE_ARGUMENT

#define INSTANTIATE_GET_REPEATED_ARGUMENT(T, fieldname) \
template<> vector<T> OperatorBase::Args<T>(const string& name) { \
    if(args_.count(name) == 0) return vector<T>(); \
    vector<T> values; \
    for(const auto& v : args_[name]->fieldname()) \
        values.push_back(static_cast<T>(v)); \
    return values; \
}

INSTANTIATE_GET_REPEATED_ARGUMENT(float, floats)
INSTANTIATE_GET_REPEATED_ARGUMENT(double, floats)
INSTANTIATE_GET_REPEATED_ARGUMENT(int, ints)
INSTANTIATE_GET_REPEATED_ARGUMENT(bool, ints)
INSTANTIATE_GET_REPEATED_ARGUMENT(int64_t, ints)
INSTANTIATE_GET_REPEATED_ARGUMENT(string, strings)
#undef INSTANTIATE_GET_REPEATED_ARGUMENT

template void Operator<CPUContext>::PrepareResource();
template void Operator<CUDAContext>::PrepareResource();
template void Operator<CNMLContext>::PrepareResource();
template void Operator<CPUContext>::ReleaseResource();
template void Operator<CUDAContext>::ReleaseResource();
template void Operator<CNMLContext>::ReleaseResource();

}  // namespace dragon