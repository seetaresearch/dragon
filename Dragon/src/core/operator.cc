#include "core/operator.h"
#include "core/workspace.h"
#include "utils/logging.h"

namespace dragon {

/* Default constructor of <OperatorBase> */

OperatorBase::OperatorBase(
    const OperatorDef&          def,
    Workspace*                  ws)
        : def_(def), ws_(ws), handle_(def.name()),
          dtype_("float32"), data_format_("NCHW") {
    // Scan the defined arguments
    for (auto& arg : def_.arg()) {
        CHECK_GT(arg.name().size(), 0);
        CHECK_EQ(args_.count(arg.name()), 0);
        args_[arg.name()] = &arg;
        if (arg.name() == "handle") {
            handle_ = arg.s();
        } else if (arg.name() == "dtype") {
            dtype_ = arg.s();
        } else if (arg.name() == "data_format") {
            data_format_ = arg.s();
        }
    }

    // Set the inputs and outputs
    string name; size_t ver_pos;
    for (const auto& e : def.input()) {
        name = e;
        if ((ver_pos = e.find("/ver:")) != string::npos)
            name = e.substr(0, ver_pos);
        inputs_.push_back(ws->GetTensor(name));
    }
    for (const auto& e : def.output()) {
        name = e;
        if ((ver_pos = e.find("/ver:")) != string::npos)
            name = e.substr(0, ver_pos);
        outputs_.push_back(ws->CreateTensor(name));
    }
}

/* Return the specified input tensor */

Tensor& OperatorBase::X(int i) {
    CHECK_LT(i, (int)inputs_.size());
    CHECK_GE(i, -(int)inputs_.size());
    if (i >= 0) return *inputs_[i];
    else return *inputs_[i + inputs_.size()];
}

/* Return the specified output tensor */

Tensor* OperatorBase::Y(int i) {
    CHECK_LT(i, (int)outputs_.size());
    CHECK_GE(i, -(int)outputs_.size());
    if (i >= 0) return outputs_[i];
    else return outputs_[i + outputs_.size()];
}

/* Return the dtype string according to given tensor */

string OperatorBase::DTypeString(
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

/* Return the dtype string according to given type */

string OperatorBase::DTypeString(
    const string&               dtype,
    const Set<string>&          dtypes) const {
    std::stringstream ss;
    ss << "Unsupported DType: " << dtype << "\n";
    ss << "<" << type() << "Op>" << " supports the following dtypes: {\n";
    for (auto& dtype : dtypes) ss << "    * " << dtype << ",\n";
    ss << "}";
    return ss.str();
}

/* Modify operator according to the given def  */

void OperatorBase::UpdateFrom(const OperatorDef& def) {
    handle_ = def.name();
    inputs_.resize(def.input_size());
    outputs_.resize(def.output_size());
    for (int i = 0; i < inputs_.size(); i++)
        inputs_[i] = ws()->GetTensor(def.input(i));
    for (int i = 0; i < outputs_.size(); i++)
        outputs_[i] = ws()->CreateTensor(def.output(i));
}

/* Create an operator from the factory  */

OperatorBase* TryCreateOperator(
    const string&               key,
    const OperatorDef&          def,
    Workspace*                  ws) {
    switch (def.device_option().device_type()) {
        case PROTO_CPU:
            return CPUOperatorRegistry()->Create(key, def, ws);
        case PROTO_CUDA:
#ifdef WITH_CUDNN
            if (CUDNNOperatorRegistry()->Has(key) &&
                CUDAContext::obj()->cudnn_enabled_)
                return CUDNNOperatorRegistry()->Create(key, def, ws);
#endif
            return CUDAOperatorRegistry()->Create(key, def, ws);
        case PROTO_CNML:
            return CNMLOperatorRegistry()->Create(key, def, ws);
        default:
            LOG(FATAL) << "Unknown Device: "
                       << def.device_option().device_type();
            return nullptr;
    }
}

/* New an operator from the raw def */

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

/* Make the gradient for the given def */

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
                    (i > 0 ? (":" + str::to(i)) : "")
            );
        }
    }
    // Copy device option and arguments
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

/* Prepare the content of inputs */

template <class Context>
void Operator<Context>::PrepareResource() {
    string tensor_name;
    size_t ver_pos; int version;
    for (int i = 0; i < XSize(); i++) {
        if (X(i).version() >= 0) {
            tensor_name = def().input(i);
            ver_pos = tensor_name.find("/ver:");
            version = std::atoi(tensor_name.substr(ver_pos + 5).c_str());
            if (version == X(i).version()) continue;
            LOG(DEBUG) << "Excepted version of Tensor(" + X(i).name() + ") "
                       << "is " << version << ", got " << X(i).version()
                       << ". Recompute.";
            Tensor* flag = ws()->GetTensor("/opt/recomp_flag");
            flag->mutable_data<bool, CPUContext>()[0] = true;
            vector<OperatorBase*>& chain = subgraph()[tensor_name];
            for (auto* op : chain) op->Run(ctx()->stream_id());
            flag->mutable_data<bool, CPUContext>()[0] = false;
        }
    }
}

/* Release the ownership of inputs */

template <class Context>
void Operator<Context>::ReleaseResource() {
    string tensor_name;
    size_t ver_pos; int version;
    for (int i = 0; i < YSize(); i++) {
        if (Y(i)->version() >= 0) {
            tensor_name = def().output(i);
            ver_pos = tensor_name.find("/ver:");
            version = std::atoi(tensor_name.substr(ver_pos + 5).c_str());
            Y(i)->set_version(version);
        }
    }
}

/* Operator Registry */

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

/* Macros */

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
template<> vector<T> OpArgs<T>(const string& name) { \
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