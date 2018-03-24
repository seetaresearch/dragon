#include "operators/ndarray/reshape_op.h"
#include "core/workspace.h"

namespace dragon {

string dim_string(const vector<TIndex>& shape) {
    std::stringstream ss;
    ss << "(";
    for (int i = 0; i < shape.size() - 1; i++) ss << shape[i] << ",";
    ss << shape[shape.size() - 1] << ")";
    return ss.str();
}

template <class Context>
void ReshapeOp<Context>::RunOnDevice() {
    if (shape_desc.size() > 0 || shape_value.size() > 0) {
        require_shape.resize(std::max(shape_desc.size(), shape_value.size()));
        for (int i = 0; i < require_shape.size(); i++)
            require_shape[i] = shape(i);
    } else if (shape_like_desc.size() > 0) {
        Tensor* shape_like_tensor = ws()->GetTensor(shape_like_desc);
        require_shape.resize(shape_like_tensor->ndim());
        for (int i = 0; i < require_shape.size(); i++)
            require_shape[i] = shape_like_tensor->dim(i);
    } else { LOG(FATAL) << "Missing the require shape."; }

    vector<TIndex> Xdims = input(0).dims();
    new_shape.resize(require_shape.size());
    int infer_dim = -1;
    TIndex total_count = 1;
    for (int i = 0; i < require_shape.size(); i++) {
        if (require_shape[i] == 0) {
            //  handle unchanged dim
            CHECK_LT(i, (int)Xdims.size())
                << "\nDim(" << i << ") is out of the Xdims range of (0, "
                << Xdims.size() << ").";
            new_shape[i] = Xdims[i];
        } else if (require_shape[i] > 0) {
            //  handle reseted dim
            new_shape[i] = require_shape[i];
        } else {
            //  handle inferred dim
            CHECK_EQ(infer_dim, -1)
                << "\nDim(" << infer_dim << ") required infer before"
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
                    << "\nCan not change the total size: "
                    << input(0).dim_string() << " -> " << dim_string(new_shape);
                new_shape[i] = input(0).count() / total_count;
                total_count *= new_shape[i];
                break;
            }
        }
    }
    CHECK_EQ(total_count, input(0).count())
        << "\nCan not change the total size."
        << input(0).dim_string() << " -> " << dim_string(new_shape);
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