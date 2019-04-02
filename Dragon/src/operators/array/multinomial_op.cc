#include "core/workspace.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"
#include "operators/array/multinomial_op.h"

namespace dragon {

template <class Context>
void MultinomialOp<Context>::SoftmaxRun() {
    auto softmax_def = MakeOperatorDef("Softmax", "",
        vector<string>({ Input(0).name() }),
        vector<string>({ mount_name("softmax/prob") }));
    Argument arg; arg.set_name("axis"); arg.set_i(axis);
    softmax_def.add_arg()->CopyFrom(arg);
    if (def().has_device_option())
        softmax_def.mutable_device_option()->CopyFrom(
            def().device_option());
    if (softmax_op) { softmax_op->UpdateFrom(softmax_def); }
    else { softmax_op.reset(NewOperator(softmax_def, ws())); }
    softmax_op->Run(ctx()->stream_id());
    prob = ws()->GetTensor(mount_name("softmax/prob"));
}

template <class Context> template <typename T>
void MultinomialOp<Context>::RunWithType() {
    auto* Xdata = normalize ?
        prob->template data<T, CPUContext>() :
            Input(0).template data<T, CPUContext>();

    vector<double> cumsum(Input(0).dim(axis));
    auto* Sdata = static_cast<double*>(cumsum.data());
    auto* Ydata = Output(0)->template mutable_data<int64_t, CPUContext>();

    double running_total, r;
    int idx = 0, num_classes = Input(0).dim(axis);

    auto* rng = ctx()->rand_generator();

    for (int i = 0; i < outer_dim; ++i) {
        running_total = 0.;
        for (int j = 0; j < num_classes; ++j) {
            running_total += (double)Xdata[j];
            Sdata[j] = running_total;
        }
        std::uniform_real_distribution<double> dist(
            0.f, running_total);
        for (int j = 0; j < (int)num_samples; ++j) {
            r = dist(*rng);
            auto found_iter = std::upper_bound(
                Sdata, Sdata + num_classes, r);
            Ydata[idx++] = std::min(
                (int)std::distance(Sdata,
                    found_iter), num_classes - 1);
        }
        Xdata += num_classes;
    }

    Output(0)->template data<int64_t, Context>();
}

template <class Context>
void MultinomialOp<Context>::RunOnDevice() {
    ctx()->set_stream_id(0);  // Enforce DefaultStream

    axis = Input(0).ndim() - 1;
    auto output_dims = Input(0).dims();
    output_dims[axis] = num_samples;
    outer_dim = Input(0).count(0, axis);
    Output(0)->Reshape(output_dims);

    // Normalize the logits if necessary
    if (normalize) SoftmaxRun();

    if (XIsType(Input(0), int8_t)) RunWithType<int8_t>();
    else if (XIsType(Input(0), uint8_t)) RunWithType<uint8_t>();
    else if (XIsType(Input(0), int)) RunWithType<int>();
    else if (XIsType(Input(0), int64_t)) RunWithType<int64_t>();
    else if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), double)) RunWithType<double>();
    else LOG(FATAL) << DTypeHelper(Input(0), {
        "bool", "int8", "uint8", "int32", "int64",
            "float32", "float64",
    });
}

DEPLOY_CPU(Multinomial);
#ifdef WITH_CUDA
DEPLOY_CUDA(Multinomial);
#endif
OPERATOR_SCHEMA(Multinomial).NumInputs(1).NumOutputs(1);

NO_GRADIENT(Multinomial);

}  // namespace dragon