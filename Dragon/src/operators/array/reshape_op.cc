#include "core/workspace.h"
#include "operators/array/dimension_op.h"

namespace dragon {

template <class Context>
void ReshapeOp<Context>::RunOnDevice() {
    if (dims_desc_.size() > 0 || dims_.size() > 0) {
        req_shape_.resize(std::max(
            dims_desc_.size(), dims_.size()));
        for (int i = 0; i < req_shape_.size(); i++)
            req_shape_[i] = dims(i);
    } else if (shape_desc_.size() > 0) {
        auto* sl = ws()->GetTensor(shape_desc_);
        req_shape_.resize(sl->ndim());
        for (int i = 0; i < req_shape_.size(); i++)
            req_shape_[i] = sl->dim(i);
    } else {
        LOG(FATAL) << "Missing the required shape.";
    }

    vec64_t in_shape = X(0).dims();
    new_shape_.resize(req_shape_.size());
    int infer_dim = -1;
    int64_t total_count = 1;
    for (int i = 0; i < req_shape_.size(); i++) {
        if (req_shape_[i] == 0) {
            // Handle unchanged dim
            CHECK_LT(i, (int)in_shape.size())
                << "\nDim(" << i << ") is out of the Xdims "
                << "range of (0, " << in_shape.size() << ").";
            new_shape_[i] = in_shape[i];
        } else if (req_shape_[i] > 0) {
            // Handle reseted dim
            new_shape_[i] = req_shape_[i];
        } else {
            // Handle inferred dim
            CHECK_EQ(infer_dim, -1)
                << "\nCould not infer Dim( " << infer_dim << "), "
                << "Dim(" << i << ") both.";
            new_shape_[i] = -1;
            infer_dim = i;
        }
        if (new_shape_[i] != -1) {
            total_count *= new_shape_[i];
        }
    }

    // Solve inferred dim if necessary
    if (infer_dim != -1) {
        for (int i = 0; i < new_shape_.size(); i++) {
            if (new_shape_[i] == -1) {
                CHECK_EQ(X(0).count() % total_count, 0)
                    << "\nCan not change the total size: "
                    << X(0).DimString() << " -> "
                    << Tensor::DimString(new_shape_);
                new_shape_[i] = X(0).count() / total_count;
                total_count *= new_shape_[i];
                break;
            }
        }
    }
    CHECK_EQ(total_count, X(0).count())
        << "\nCan not change the total size."
        << X(0).DimString() << " -> "
        << Tensor::DimString(new_shape_);
    Y(0)->Reshape(new_shape_);
    Y(0)->SetMeta(X(0).meta());
    Y(0)->Share(X(0).memory());
}

DEPLOY_CPU(Reshape);
#ifdef WITH_CUDA
DEPLOY_CUDA(Reshape);
#endif

DEPLOY_CPU(ReshapeGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(ReshapeGradient);
#endif

OPERATOR_SCHEMA(Reshape)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(ReshapeGradient)
     /* X, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1)
     /* dY => dX */
    .Inplace({ { 1, 0 } });

REGISTER_GRADIENT(Reshape, SimpleGradientMaker);

}  // namespace dragon