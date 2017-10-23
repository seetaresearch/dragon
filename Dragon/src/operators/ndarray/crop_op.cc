#include "operators/ndarray/crop_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void CropOp<Context>::RecursiveRunWithType(vector<TIndex> idxs,
                                           const vector<TIndex>& offsets,
                                           int cur_dim,
                                           Tensor* x,
                                           Tensor* y) { 
    int num_spatial_axes;
    if (TypeMeta::Id<Context>() == TypeMeta::Id<CPUContext>()) num_spatial_axes = 1;
    else num_spatial_axes = 2;
    if (cur_dim + num_spatial_axes < y->ndim()) {
        for (int i = 0; i < y->dim(cur_dim); i++) {
            idxs[cur_dim] = i;
            RecursiveRunWithType<T>(idxs, offsets, cur_dim + 1, x, y);
        }
    } else {
        kernel::Crop2D<T, Context>(idxs, offsets, cur_dim, x, y, &ctx());
    }
}

template <class Context> template <typename T>
void CropOp<Context>::RunWithType() {
    vector<TIndex> idxs(output(0)->ndim(), 0);
    RecursiveRunWithType<T>(idxs, offsets, 0, &input(0), output(0));
}

template <class Context>
void CropOp<Context>::ComputeOutputShape() {
    output_shape.clear(); offsets.clear();
    for (int i = 0; i < input(0).ndim(); i++) {
        TIndex crop_offset = 0;
        TIndex new_dim = input(0).dim(i);
        if (i >= axis) {
            if (shape.size() != 0) {
                new_dim = shape[i];
            } else {
                Tensor* like = ws()->GetTensor(shape_like);
                new_dim = like->dim(i);
            }
            if (offsets_param.size() == 1) {
                crop_offset = offsets_param[0];
            } else if (offsets_param.size() > 1) {
                crop_offset = offsets_param[i - axis];
            }
            CHECK_GE(input(0).dim(i) - crop_offset, new_dim)
                << "Cropping out of boundary: \n"
                << "axis: " << i << "\n"
                << "start offset: " << crop_offset << "\n"
                << "crop size: " << new_dim;
        }
        output_shape.push_back(new_dim);
        offsets.push_back(crop_offset);
    }
}

template <class Context>
void CropOp<Context>::RunOnDevice() {
    ComputeOutputShape();
    output(0)->Reshape(output_shape);
    
    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(Crop);
#ifdef WITH_CUDA
DEPLOY_CUDA(Crop);
#endif
OPERATOR_SCHEMA(Crop).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void CropGradientOp<Context>::RecursiveRunWithType(vector<TIndex> idxs,
                                                   const vector<TIndex>& offsets,
                                                   int cur_dim,
                                                   Tensor* dy,
                                                   Tensor* dx) { 
    int num_spatial_axes;
    if (TypeMeta::Id<Context>() == TypeMeta::Id<CPUContext>()) num_spatial_axes = 1;
    else num_spatial_axes = 2;
    if (cur_dim + num_spatial_axes < dy->ndim()) {
        for (int i = 0; i < dy->dim(cur_dim); i++) {
            idxs[cur_dim] = i;
            RecursiveRunWithType<T>(idxs, offsets, cur_dim + 1, dy, dx);
        }
    } else {
        kernel::Crop2DGrad<T, Context>(idxs, offsets, cur_dim, dy, dx, &ctx());
    }
}

template <class Context> template <typename T>
void CropGradientOp<Context>::RunWithType() {
    vector<TIndex> idxs(output(0)->ndim(), 0);
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    math::Set<T, Context>(output(0)->count(), 0, dXdata);
    RecursiveRunWithType<T>(idxs, offsets, 0, &input(-1), output(0));
}

template <class Context>
void CropGradientOp<Context>::ComputeOutputShape() {
    output_shape.clear(); offsets.clear();
    for (int i = 0; i < input(0).ndim(); i++) {
        TIndex crop_offset = 0;
        TIndex new_dim = input(0).dim(i);
        if (i >= axis) {
            if (shape.size() != 0) {
                new_dim = shape[i];
            } else {
                Tensor* like = ws()->GetTensor(shape_like);
                new_dim = like->dim(i);
            }
            if (offsets_param.size() == 1) {
                crop_offset = offsets_param[0];
            } else if (offsets_param.size() > 1) {
                crop_offset = offsets_param[i - axis];
            }
            CHECK_GE(input(0).dim(i) - crop_offset, new_dim)
                << "Cropping out of boundary: \n"
                << "axis: " << i << "\n"
                << "start offset: " << crop_offset << "\n"
                << "crop size: " << new_dim;
        }
        output_shape.push_back(new_dim);
        offsets.push_back(crop_offset);
    }
}

template <class Context>
void CropGradientOp<Context>::RunOnDevice() {
    ComputeOutputShape();
    output(0)->ReshapeLike(input(0));
    
    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(CropGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(CropGradient);
#endif
OPERATOR_SCHEMA(CropGradient).NumInputs(2).NumOutputs(1);

class GetCropGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetCropGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(Crop, GetCropGradient);

}    // namespace dragon