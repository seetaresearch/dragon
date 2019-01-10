#include <algorithm>

#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/misc/accuracy_op.h"

namespace dragon {

template <class Context> template <typename Tx, typename Ty>
void AccuracyOp<Context>::RunWithType() {
    static CPUContext cctx;
    float* Y1data, *Y2data = nullptr;
    Y1data = Output(0)->template mutable_data<float, CPUContext>();
    if (OutputSize() > 1) {
        Y2data = Output(1)->template mutable_data<float, CPUContext>();
        math::Set<float, CPUContext>(num_classes, 0, Y2data, &cctx);
    }

    Map<int, int64_t> num_per_class;
    int64_t acc = 0, count = 0;

    const Tx* Xdata;
    if (XIsType(Input(0), float16)) {
        Tensor* X32T = ws()->CreateTensor(mount_name("accuracy/f32"));
        X32T->ReshapeLike(Input(0));
        auto* X16 = Input(0).template data<float16, CPUContext>();
        auto* X32 = X32T->template mutable_data<float, CPUContext>();
        kernel::TypeA2B<float16, float, CPUContext>(
            Input(0).count(), X16, X32, &cctx);
        Xdata = X32;
    } else Xdata = Input(0).template data<Tx, CPUContext>();

    auto* labels = Input(1).template data<Ty, CPUContext>();
    auto* ignores = ignore.count() > 0 ?
        ignore.data<int, CPUContext>() : nullptr;
    const int64_t dim = Input(0).count() / outer_dim;
    for (int i = 0; i < outer_dim; i++) {
        for (int j = 0; j < inner_dim; j++) {
            const int label = labels[i * inner_dim + j];
            for (int k = 0; k < ignore.count(); k++)
                if (label == ignores[k]) continue;
            if (OutputSize() > 1) num_per_class[label]++;
            vector<pair<Tx, int> > vec;
            for (int k = 0; k < num_classes; k++)
                vec.push_back(
                    std::make_pair(Xdata[i * dim + k * inner_dim + j], k));
            std::partial_sort(
                vec.begin(), vec.begin() + top_k, vec.end(),
                    std::greater<pair<Tx, int> >());
            for (int k = 0; k < top_k; k++) {
                if (vec[k].second == label) {
                    if (OutputSize() > 1) Y2data[label]++;
                    acc++;
                    break;
                }
            }
            count++;
        }  // End inner_dim
    }  // End outer_dim

    Y1data[0] = (float)acc / count;
    if (Y2data) {
        for (int i = 0; i < num_classes; i++)
            Y2data[i] = num_per_class[i] == 0 ?
                0 : Y2data[i] / num_per_class[i];
    }
}

template <class Context>
void AccuracyOp<Context>::RunOnDevice() {
    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    num_classes = Input(0).dim(axis);
    CHECK_EQ(outer_dim * inner_dim, Input(1).count())
        << "\nGiven (" << outer_dim << "," << inner_dim << ") predictions,"
        << "\nbut provided " << Input(1).count() << " labels.";
    Output(0)->Reshape({ 1 });
    if (OutputSize() > 1) Output(1)->Reshape({ num_classes });

    if (XIsType(Input(0), float) || XIsType(Input(0), float16)) {
        if (XIsType(Input(1), float)) RunWithType<float, float>();
        else if(XIsType(Input(1), int64_t)) RunWithType<float, int64_t>();
        else LOG(FATAL) << DTypeHelper(Input(1), { "float32", "int64" });
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(Accuracy);
#ifdef WITH_CUDA
DEPLOY_CUDA(Accuracy);
#endif
OPERATOR_SCHEMA(Accuracy).NumInputs(2).NumOutputs(1, 2);

NO_GRADIENT(Accuracy);

}  // namespace dragon