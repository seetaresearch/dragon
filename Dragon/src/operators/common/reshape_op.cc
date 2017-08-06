#include "operators/common/reshape_op.h"

namespace dragon {

template <class Context>
void ReshapeOp<Context>::RunOnDevice() {
    vector<TIndex> Xdims = input(0).dims();
    int infer_dim = -1;
    TIndex total_count = 1;
    for (int i = 0; i < shape.size(); i++) {
        //  handle unchanged dim
        if (shape[i] == 0) {
            CHECK_LT(i, (int)Xdims.size())
                << "\ndim(" << i << ") is out of the Xdims range of (0, "
                << Xdims.size() << ").";
            new_shape[i] = Xdims[i];
        }
        //  handle reseted dim
        else if (shape[i] > 0) {
            new_shape[i] = shape[i];
        }
        //  handle inferred dim
        else {
            CHECK_EQ(infer_dim, -1)
                << "\ndim(" << infer_dim << ") required infer before"
                << "\ncould not infer for dim(" << i << ") both.";
            new_shape[i] = -1;
            infer_dim = i;
        }
        if (new_shape[i] != -1) total_count *= new_shape[i];
    }

    //  solve inferred dim if necessary
    if (infer_dim != -1) {
        for (int i = 0; i < new_shape.size(); i++) {
            if (new_shape[i] == -1) {
                CHECK_EQ(input(0).count() % total_count, 0)
                    << "\nreshape could not change the total size.";
                new_shape[i] = input(0).count() / total_count;
                total_count *= new_shape[i];
                break;
            }
        }
    }
    CHECK_EQ(total_count, input(0).count())
        << "\nreshape could not change the total size.";
    output(0)->Reshape(new_shape);
    output(0)->Share(input(0));
}

DEPLOY_CPU(Reshape);
#ifdef WITH_CUDA
DEPLOY_CUDA(Reshape);
#endif
OPERATOR_SCHEMA(Reshape).NumInputs(1).NumOutputs(1);

template <class Context>
void ReshapeGradientOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(0));
    output(0)->Share(input(1));
}

DEPLOY_CPU(ReshapeGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ReshapeGradient);
#endif
OPERATOR_SCHEMA(ReshapeGradient).NumInputs(2).NumOutputs(1);

class GetReshapeGradient final : public GradientMakerBase {
public:
    GRADIENT_MAKER_CTOR(GetReshapeGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(Reshape, GetReshapeGradient);

}    // namespace dragon

