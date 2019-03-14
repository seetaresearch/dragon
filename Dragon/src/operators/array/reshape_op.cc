#include "core/workspace.h"
#include "operators/array/dimension_op.h"

namespace dragon {

template <class Context>
void ReshapeOp<Context>::RunOnDevice() {
    if (dims_desc.size() > 0 || dims_value.size() > 0) {
        require_shape.resize(std::max(dims_desc.size(), dims_value.size()));
        for (int i = 0; i < require_shape.size(); i++)
            require_shape[i] = dims(i);
    } else if (shape_like_desc.size() > 0) {
        Tensor* shape_like_tensor = ws()->GetTensor(shape_like_desc);
        require_shape.resize(shape_like_tensor->ndim());
        for (int i = 0; i < require_shape.size(); i++)
            require_shape[i] = shape_like_tensor->dim(i);
    } else { LOG(FATAL) << "Missing the required shape."; }

    vector<int64_t> Xdims = Input(0).dims();
    new_shape.resize(require_shape.size());
    int infer_dim = -1;
    int64_t total_count = 1;
    for (int i = 0; i < require_shape.size(); i++) {
        if (require_shape[i] == 0) {
            // Handle unchanged dim
            CHECK_LT(i, (int)Xdims.size())
                << "\nDim(" << i << ") is out of the Xdims "
                << "range of (0, " << Xdims.size() << ").";
            new_shape[i] = Xdims[i];
        } else if (require_shape[i] > 0) {
            // Handle reseted dim
            new_shape[i] = require_shape[i];
        } else {
            // Handle inferred dim
            CHECK_EQ(infer_dim, -1)
                << "\nCould not infer Dim( " << infer_dim << "), "
                << "Dim(" << i << ") both.";
            new_shape[i] = -1;
            infer_dim = i;
        }
        if (new_shape[i] != -1) total_count *= new_shape[i];
    }

    // Solve inferred dim if necessary
    if (infer_dim != -1) {
        for (int i = 0; i < new_shape.size(); i++) {
            if (new_shape[i] == -1) {
                CHECK_EQ(Input(0).count() % total_count, 0)
                    << "\nCan not change the total size: "
                    << Input(0).DimString()
                    << " -> " << Tensor::DimString(new_shape);
                new_shape[i] = Input(0).count() / total_count;
                total_count *= new_shape[i];
                break;
            }
        }
    }
    CHECK_EQ(total_count, Input(0).count())
        << "\nCan not change the total size."
        << Input(0).DimString()
        << " -> " << Tensor::DimString(new_shape);
    Output(0)->Reshape(new_shape);
    Output(0)->SetMeta(Input(0).meta());
    Output(0)->Share(Input(0).memory());
}

DEPLOY_CPU(Reshape);
#ifdef WITH_CUDA
DEPLOY_CUDA(Reshape);
#endif
OPERATOR_SCHEMA(Reshape).NumInputs(1).NumOutputs(1);

DEPLOY_CPU(ReshapeGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ReshapeGradient);
#endif

OPERATOR_SCHEMA(ReshapeGradient)
    .NumInputs(2).NumOutputs(1)
    .Inplace({ { 1, 0 } });

REGISTER_GRADIENT(Reshape, SimpleGradientMaker);

}  // namespace dragon