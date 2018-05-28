#include <algorithm>
#include "operators/misc/accuracy_op.h"
#include "utils/math_functions.h"

namespace dragon {

template <class Context> template <typename Tx, typename Ty>
void AccuracyOp<Context>::RunWithType() {
    if (OutputSize() > 1) {
        math::Set<float, CPUContext>(num_classes, 0,
            Output(1)->template mutable_data<float, CPUContext>());
    }
    Map<int, TIndex> num_per_class;

    TIndex acc = 0, count = 0;
    auto* Xdata = Input(0).template data<Tx, CPUContext>();
    auto* labels = Input(1).template data<Ty, CPUContext>();
    auto* ignores = ignore_labels.count() > 0 ?
                        ignore_labels.data<int, CPUContext>() : nullptr;
    const TIndex dim = Input(0).count() / outer_dim;
    for (int i = 0; i < outer_dim; i++) {
        for (int j = 0; j < inner_dim; j++) {
            const int label = labels[i * inner_dim + j];
            for (int k = 0; k < ignore_labels.count(); k++)
                if (label == ignores[k]) continue;
            if (OutputSize() > 1) num_per_class[label]++;
            vector<pair<Tx, int> > vec;
            for (int k = 0; k < num_classes; k++)
                vec.push_back(std::make_pair(Xdata[i * dim + k * inner_dim + j], k));
            std::partial_sort(vec.begin(), vec.begin() + top_k, vec.end(), std::greater<pair<Tx, int> >());
            for (int k = 0; k < top_k; k++) {
                if (vec[k].second == label) {
                    if (OutputSize() > 1)
                        Output(1)->template mutable_data<float, CPUContext>()[label]++;
                    acc++;
                    break;
                }
            }
            count++;
        }    //  end inner_dim
    }    // end outer_dim

    Output(0)->template mutable_data<float, CPUContext>()[0] = (float)acc / count;
    if (OutputSize() > 1) {
        auto* acc_per_class = Output(1)->template mutable_data<float, CPUContext>();
        for (int i = 0; i < num_classes; i++)
            acc_per_class[i] = num_per_class[i] == 0 ?
                0 : acc_per_class[i] / num_per_class[i];
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
    Output(0)->Reshape(vector<TIndex>(1, 1));
    if (OutputSize() > 1) Output(1)->Reshape(vector<TIndex>(1, num_classes)); 

    if (XIsType(Input(0), float)) {
        if (XIsType(Input(1), float)) RunWithType<float, float>();
        else if(XIsType(Input(1), int64_t)) RunWithType<float, int64_t>();
        else LOG(FATAL) << DTypeHelper(Input(1), { "float32", "int64" });
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(Accuracy);
#ifdef WITH_CUDA
DEPLOY_CUDA(Accuracy);
#endif
OPERATOR_SCHEMA(Accuracy).NumInputs(2).NumOutputs(1, 2);

NO_GRADIENT(Accuracy);

}    // namespace dragon