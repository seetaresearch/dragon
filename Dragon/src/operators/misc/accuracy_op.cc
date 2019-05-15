#include <algorithm>

#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/misc/accuracy_op.h"

namespace dragon {

template <class Context>
template <typename Tx, typename Ty>
void AccuracyOp<Context>::RunImpl() {
    int64_t acc = 0, count = 0;
    int64_t cols = X(0).count() / outer_dim_;

    auto* x = X(0).template data<Tx, CPUContext>();
    auto* target = X(1).template data<Ty, CPUContext>();
    auto* ignore = !ignore_.count() ? nullptr :
                    ignore_.template data<int, CPUContext>();
    auto* y = Y(0)->template mutable_data<float, CPUContext>();

    for (int i = 0; i < outer_dim_; ++i) {
        for (int j = 0; j < inner_dim_; ++j) {
            const int label = target[i * inner_dim_ + j];
            for (int k = 0; k < ignore_.count(); k++)
                if (label == ignore[k]) continue;
            vector<pair<Tx, int>> vec;
            for (int k = 0; k < axis_dim_; k++)
                vec.push_back(
                    std::make_pair(
                        x[i * cols + k * inner_dim_ + j], k
                    )
                );
            std::partial_sort(
                vec.begin(),
                vec.begin() + top_k_,
                vec.end(),
                std::greater<pair<Tx, int>>()
            );
            for (int k = 0; k < top_k_; k++) {
                if (vec[k].second == label) { acc++; break; }
            }
            count++;
        }  // End inner_dim
    }  // End outer_dim

    y[0] = (float)acc / (float)count;
}

template <class Context>
void AccuracyOp<Context>::RunOnDevice() {
    axis_dim_  = X(0).dim(axis_);
    outer_dim_ = X(0).count(0, axis_);
    inner_dim_ = X(0).count(axis_ + 1);

    CHECK_EQ(outer_dim_ * inner_dim_, X(1).count())
        << "\nNum of preds must match the num of labels.";

    Y(0)->Reshape({});

    if (XIsType(X(0), float)) {
        if (XIsType(X(1), float)) {
            RunImpl<float, float>();
        } else if (XIsType(X(1), int64_t)) {
            RunImpl<float, int64_t>();
        } else {
            LOG(FATAL) << DTypeString(X(1),
                { "float32", "int64" }
            );
        }
    } else {
        LOG(FATAL) << DTypeString(
            X(0), { "float32" }
        );
    }
}

DEPLOY_CPU(Accuracy);
#ifdef WITH_CUDA
DEPLOY_CUDA(Accuracy);
#endif

OPERATOR_SCHEMA(Accuracy)
     /* X, T */
    .NumInputs(2)
     /* Y */
    .NumOutputs(1);

NO_GRADIENT(Accuracy);

}  // namespace dragon