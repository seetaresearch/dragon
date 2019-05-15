#include "core/workspace.h"
#include "utils/cast.h"
#include "utils/op_kernel.h"
#include "operators/array/multinomial_op.h"

namespace dragon {

template <class Context>
void MultinomialOp<Context>::SoftmaxRun() {
    auto softmax_def = MakeOperatorDef(
        "Softmax", "",
        vector<string>({ X(0).name() }),
        vector<string>({ unique_name("prob") })
    );
    Argument arg; arg.set_name("axis"); arg.set_i(axis_);
    softmax_def.add_arg()->CopyFrom(arg);
    if (def().has_device_option())
        softmax_def.mutable_device_option()
            ->CopyFrom(def().device_option());
    if (softmax_op_) { softmax_op_->UpdateFrom(softmax_def); }
    else { softmax_op_.reset(NewOperator(softmax_def, ws())); }
    softmax_op_->Run(ctx()->stream_id());
}

template <class Context> template <typename T>
void MultinomialOp<Context>::RunImpl() {
    auto* x = normalize_ ?
        ws()->GetTensor(unique_name("prob"))
            ->template data<T, CPUContext>()
       : X(0).template data<T, CPUContext>();

    vector<double> cumsum(X(0).dim(axis_));
    auto* cdf = static_cast<double*>(cumsum.data());
    auto* y = Y(0)->template mutable_data<int64_t, CPUContext>();

    double running_total, r;
    int yi = 0, num_classes = X(0).dim(axis_);
    double uniform_p = 1. / (double)num_classes;

    auto* rng = ctx()->rand_generator();
    std::uniform_real_distribution<float> eps_dist;

    for (int i = 0; i < outer_dim_; ++i) {
        running_total = 0.;
        if (eps_ > 0.f && eps_dist(*rng) < eps_) {
            for (int j = 0; j < num_classes; ++j) {
                running_total += uniform_p;
                cdf[j] = running_total;
            }
        } else {
            for (int j = 0; j < num_classes; ++j) {
                running_total += (double)x[j];
                cdf[j] = running_total;
            }
        }
        std::uniform_real_distribution<double>
            dist(0., running_total);
        for (int j = 0; j < (int)num_samples_; ++j) {
            r = dist(*rng);
            auto found_iter = std::upper_bound(
                cdf, cdf + num_classes, r);
            y[yi++] = std::min(
                (int)std::distance(cdf, found_iter),
                num_classes - 1
            );
        }
        x += num_classes;
    }

    Y(0)->template data<int64_t, Context>();
}

template <class Context>
void MultinomialOp<Context>::RunOnDevice() {
    ctx()->set_stream_id(0);  // Enforce DefaultStream

    axis_ = X(0).ndim() - 1;
    auto out_shape = X(0).dims();
    out_shape[axis_] = num_samples_;
    outer_dim_ = X(0).count(0, axis_);

    Y(0)->Reshape(out_shape);

    // Normalize the logits if necessary
    if (normalize_) SoftmaxRun();

    DispatchHelper<TensorTypes
        <int8_t, uint8_t, int,
         int64_t, float, double>
    >::Call(this, X(0));
}

DEPLOY_CPU(Multinomial);
#ifdef WITH_CUDA
DEPLOY_CUDA(Multinomial);
#endif

OPERATOR_SCHEMA(Multinomial)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

NO_GRADIENT(Multinomial);

}  // namespace dragon