#include "operators/ndarray/crop_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void CropOp<Context>::RunWithType() {
    auto* Xdata = source->template data<T, Context>();
    auto* Ydata = dest->template mutable_data<T, Context>();
    kernel::Crop1D<T, Context>(dest->count(),
                                         dim,
                   ends[axis] - starts[axis],
                                   inner_dim,
                                starts[axis],
                                       Xdata,
                                       Ydata,
                                     &ctx());
}

template <class Context>
void CropOp<Context>::Setup() {
    //  make starts
    if (start_axis == -1) {
        //  static crop
        CHECK_EQ(starts.size(), input(0).ndim())
            << "\nThe cropping is performed on " << starts.size() << " dimensions, "
            << "while the num of dimensions of input is " << input(0).ndim() << ".";
    } else {
        //  dynamic crop
        starts.resize(input(0).ndim(), 0);
        for (int i = 0; i < starts.size(); i++) {
            if (i < start_axis || offsets.size() == 0) starts[i] = 0;
            else if (i - start_axis < (int)offsets.size()) starts[i] = offsets[i - start_axis];
            else starts[i] = offsets[offsets.size() - 1];
        }
    }

    // make ends
    if (ends.size() > 0) {
        //  static crop
        CHECK_EQ(ends.size(), input(0).ndim())
            << "\nThe cropping is performed on " << ends.size() << " dimensions, "
            << "but the num of dimensions of input is " << input(0).ndim() << "."; \
        //  fix end if necessary
        for (int i = 0; i < ends.size(); i++) 
            if (ends[i] == 0) ends[i] = input(0).dim(i);
    } else {
        CHECK(shape.size() * shape_like.size() == 0)
            << "\nCan not set shape and shape_like both.";
        CHECK(shape.size() + shape_like.size() != 0)
            << "\nMust set shape and shape_like either.";
        ends.resize(input(0).ndim(), 0);
        for (int i = 0; i < ends.size(); i++) {
            //  dynamic crop 1: keep unchanged
            if (start_axis != -1 && i < start_axis) {
                ends[i] = input(0).dim(i);
                continue;
            }
            //  dynamic crop 2: use shape
            if (shape.size() > 0) {
                CHECK_EQ(shape.size(), input(0).ndim())
                    << "\nThe cropping is performed on " << shape.size() << " dimensions, "
                    << "while the num of dimensions of input is " << input(0).ndim() << ".";
                ends[i] = starts[i] + shape[i];
            } else {
                //  dynamic crop 3: use shape_like
                Tensor* like = ws()->GetTensor(shape_like);
                CHECK_EQ(like->ndim(), input(0).ndim())
                    << "\nThe cropping is performed on " << like->ndim() << " dimensions, "
                    << "while the num of dimensions of input is " << input(0).ndim() << ".";
                ends[i] = starts[i] + like->dim(i);
            }
        }
    }

    //  check starts and ends
    for (int i = 0; i < starts.size(); i++) {
        CHECK(starts[i] >= 0 && starts[i] < (int)input(0).dim(i))
            << "\nThe cropping starts at the pos " << starts[i] << " of axis " << i << ", "
            << "while the dimension of this axis is " << input(0).dim(i) << ".";
        CHECK(ends[i] > 0 && ends[i] <= (int)input(0).dim(i))
            << "\nThe cropping ends at the pos " << ends[i] << " of axis " << i << ", "
            << "while the dimension of this axis is " << input(0).dim(i) << ".";
    }

    //  make tasks
    process_axes.clear();
    for (int i = 0; i < starts.size(); i++) {
        int cropping_size = ends[i] - starts[i];
        int reducing_size = (int)input(0).dim(i) - cropping_size;
        if (reducing_size > 0)
            process_axes.push_back({ reducing_size, i });
    }
    std::sort(process_axes.begin(), process_axes.end(), std::greater< pair<int, int> >());
}

template <class Context>
void CropOp<Context>::RunOnDevice() {
    Setup();

    //  do nothing
    if (process_axes.size() == 0) {
        output(0)->ReshapeLike(input(0));
        output(0)->Share(input(0));
        return;
    }

     //  select source & dest
    source = &input(0);
    if (process_axes.size() % 2 == 1) dest = output(0);
    else dest = ws()->GetBuffer();

    for (auto& task : process_axes) {
        axis = task.second;
        vector<TIndex> dims = source->dims();
        inner_dim = source->count(axis + 1);
        dim = source->dim(axis);
        dims[axis] = ends[axis] - starts[axis];
        dest->Reshape(dims);
        if (input(0).template IsType<float>()) RunWithType<float>();
        else LOG(FATAL) << "Unsupported input types.";
        //  allow buffer to protect X if the num of tasks >= 2
        std::swap(source, dest);
        if (process_axes.size() % 2 == 1) {
            if (dest == &input(0)) dest = ws()->GetBuffer();
        } else {
            if (dest == &input(0)) dest = output(0);
        }
    }
    ws()->ReleaseBuffer(dest);
}

DEPLOY_CPU(Crop);
#ifdef WITH_CUDA
DEPLOY_CUDA(Crop);
#endif
OPERATOR_SCHEMA(Crop).NumInputs(1).NumOutputs(1);

template <class Context>
void CropGradientOp<Context>::Setup() {
    //  make starts
    if (start_axis == -1) {
        //  static crop
        CHECK_EQ(starts.size(), input(0).ndim())
            << "\nThe cropping is performed on " << starts.size() << " dimensions, "
            << "while the num of dimensions of input is " << input(0).ndim() << ".";
    } else {
        //  dynamic crop
        starts.resize(input(0).ndim(), 0);
        for (int i = 0; i < starts.size(); i++) {
            if (i < start_axis || offsets.size() == 0) starts[i] = 0;
            else if (i - start_axis < (int)offsets.size()) starts[i] = offsets[i - start_axis];
            else starts[i] = offsets[offsets.size() - 1];
        }
    }

    // make ends
    if (ends.size() > 0) {
        //  static crop
        CHECK_EQ(ends.size(), input(0).ndim())
            << "\nThe cropping is performed on " << ends.size() << " dimensions, "
            << "but the num of dimensions of input is " << input(0).ndim() << "."; \
        //  fix end if necessary
        for (int i = 0; i < ends.size(); i++) 
            if (ends[i] == 0) ends[i] = input(0).dim(i);
    } else {
        CHECK(shape.size() * shape_like.size() == 0)
            << "\nCan not set shape and shape_like both.";
        CHECK(shape.size() + shape_like.size() != 0)
            << "\nMust set shape and shape_like either.";
        ends.resize(input(0).ndim(), 0);
        for (int i = 0; i < ends.size(); i++) {
            //  dynamic crop 1: keep unchanged
            if (start_axis != -1 && i < start_axis) {
                ends[i] = input(0).dim(i);
                continue;
            }
            //  dynamic crop 2: use shape
            if (shape.size() > 0) {
                CHECK_EQ(shape.size(), input(0).ndim())
                    << "\nThe cropping is performed on " << shape.size() << " dimensions, "
                    << "while the num of dimensions of input is " << input(0).ndim() << ".";
                ends[i] = starts[i] + shape[i];
            } else {
                //  dynamic crop 3: use shape_like
                Tensor* like = ws()->GetTensor(shape_like);
                CHECK_EQ(like->ndim(), input(0).ndim())
                    << "\nThe cropping is performed on " << like->ndim() << " dimensions, "
                    << "while the num of dimensions of input is " << input(0).ndim() << ".";
                ends[i] = starts[i] + like->dim(i);
            }
        }
    }

    //  check starts and ends
    for (int i = 0; i < starts.size(); i++) {
        CHECK(starts[i] >= 0 && starts[i] < (int)input(0).dim(i))
            << "\nThe cropping starts at the pos " << starts[i] << " of axis " << i << ", "
            << "while the dimension of this axis is " << input(0).dim(i) << ".";
        CHECK(ends[i] > 0 && ends[i] <= (int)input(0).dim(i))
            << "\nThe cropping ends at the pos " << ends[i] << " of axis " << i << ", "
            << "while the dimension of this axis is " << input(0).dim(i) << ".";
    }

    //  make tasks
    process_axes.clear();
    for (int i = 0; i < starts.size(); i++) {
        int cropping_size = ends[i] - starts[i];
        int reducing_size = (int)input(0).dim(i) - cropping_size;
        if (reducing_size > 0)
            process_axes.push_back({ reducing_size, i });
    }
    std::sort(process_axes.begin(), process_axes.end(), std::greater< pair<int, int> >());
    std::reverse(process_axes.begin(), process_axes.end());
}

template <class Context> template <typename T>
void CropGradientOp<Context>::RunWithType() {
    auto* dYdata = source->template data<T, Context>();
    auto* dXdata = dest->template mutable_data<T, Context>();
    kernel::Crop1DGrad<T, Context>(dest->count(),
                              input(0).dim(axis),
                                             dim,
                                       inner_dim,
                                    starts[axis],
                                      ends[axis],
                                          dYdata,
                                          dXdata,
                                         &ctx());
}

template <class Context>
void CropGradientOp<Context>::RunOnDevice() {
    Setup();

    //  do nothing 
    if (process_axes.size() == 0) {
        output(0)->ReshapeLike(input(-1));
        output(0)->Share(input(-1));
        return;
    }

    //  select source & buffer
    source = &input(-1);
    if (process_axes.size() % 2 == 1) dest = output(0);
    else dest = ws()->GetBuffer();

    for (auto& task : process_axes) {
        axis = task.second;
        vector<TIndex> dims = source->dims();
        inner_dim = source->count(axis + 1);
        dim = source->dim(axis);
        dims[axis] = input(0).dim(axis),
        dest->Reshape(dims);
        if (input(0).template IsType<float>()) RunWithType<float>();
        else LOG(FATAL) << "Unsupported input types.";
        //  allow buffer to protect X if the num of tasks >= 2
        std::swap(source, dest);
        if (process_axes.size() % 2 == 1) {
            if (dest == &input(-1)) dest = ws()->GetBuffer();
        } else {
            if (dest == &input(-1)) dest = output(0);
        }
    }
    ws()->ReleaseBuffer(dest);
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