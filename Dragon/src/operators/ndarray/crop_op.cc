#include "operators/ndarray/crop_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void CropOp<Context>::RunWithType() {
    const T* Xdata; T* Ydata;
    if (source == &navigator) {
        Xdata = ws()->template caches<T, Context>({ source->count() })[0];
    } else { Xdata = source->template data<T, Context>(); }
    if (dest == &navigator) {
        Ydata = ws()->template caches<T, Context>({ dest->count() })[0];
    } else { Ydata = dest->template mutable_data<T, Context>(); }

    kernel::Crop1D<T, Context>(dest->count(),
        dim, ed[axis] - st[axis], inner_dim,
            st[axis], Xdata, Ydata);
}

template <class Context>
void CropOp<Context>::Setup() {
    //  make starts
    st.assign(Input(0).ndim(), 0);
    if (start_axis == -1) {
        //  static crop
        int n_given = (int)GET_ARGUMENTS_SIZE(starts);
        for (int i = 0; i < st.size(); i++) {
            if (i < n_given) st[i] = starts(i);
            else st[i] = 0;
        }
    } else {
        //  dynamic crop
        for (int i = 0; i < st.size(); i++) {
            if (i < start_axis || offsets.size() == 0) {
                st[i] = 0;
            } else if (i - start_axis < (int)offsets.size()) {
                st[i] = offsets[i - start_axis];
            } else {
                st[i] = offsets[offsets.size() - 1];
            }
        }
    }

    // make ends
    ed.assign(Input(0).ndim(), 0);
    keep_dims.resize(Input(0).ndim(), 0);
    if (shape.size() + shape_like.size() != 0) {
        CHECK(shape.size() * shape_like.size() == 0)
            << "\nCan not set shape and shape_like both.";
        for (int i = 0; i < ed.size(); i++) {
            //  dynamic crop 1: keep unchanged
            if (start_axis != -1 && i < start_axis) {
                ed[i] = Input(0).dim(i);
                continue;
            }
            //  dynamic crop 2: use shape
            if (shape.size() > 0) {
                CHECK_EQ(shape.size(), Input(0).ndim())
                    << "\nThe cropping is performed on " << shape.size() << " dimensions, "
                    << "while the num of dimensions of input is " << Input(0).ndim() << ".";
                ed[i] = st[i] + shape[i];
            } else {
                //  dynamic crop 3: use shape_like
                Tensor* like = ws()->GetTensor(shape_like);
                CHECK_EQ(like->ndim(), Input(0).ndim())
                    << "\nThe cropping is performed on " << like->ndim() << " dimensions, "
                    << "while the num of dimensions of input is " << Input(0).ndim() << ".";
                ed[i] = st[i] + like->dim(i);
            }
        }
    } else {
        //  static crop
        int n_given = (int)GET_ARGUMENTS_SIZE(ends);
        for (int i = 0; i < ed.size(); i++) {
            keep_dims[i] = 1;
            if (i < n_given) ed[i] = ends(i);
            if (ed[i] == 0) ed[i] = Input(0).dim(i);
            if (ed[i] == -1) { ed[i] = st[i] + 1; keep_dims[i] = 0; }
        }
    }

    //  check starts and ends
    for (int i = 0; i < st.size(); i++) {
        CHECK(st[i] >= 0 && st[i] < (int)Input(0).dim(i))
            << "\nThe cropping starts at the pos " << st[i] << " of axis " << i << ", "
            << "while the dimension of this axis is " << Input(0).dim(i) << ".";
        CHECK(ed[i] > 0 && ed[i] <= (int)Input(0).dim(i))
            << "\nThe cropping ends at the pos " << ed[i] << " of axis " << i << ", "
            << "while the dimension of this axis is " << Input(0).dim(i) << ".";
    }

    //  store st & ed & kd
    Tensor* tst = ws()->CreateTensor("/mnt/" + anchor() + "/crop/starts");
    Tensor* ted = ws()->CreateTensor("/mnt/" + anchor() + "/crop/ends");
    Tensor* tkd = ws()->CreateTensor("/mnt/" + anchor() + "/crop/keep_dims");
    tst->Reshape(vector<TIndex>(1, st.size()));
    ted->Reshape(vector<TIndex>(1, ed.size()));
    tkd->Reshape(vector<TIndex>(1, keep_dims.size()));
    auto* Sdata = tst->template mutable_data<int, CPUContext>();
    auto* Edata = ted->template mutable_data<int, CPUContext>();
    auto* Kdata = tkd->template mutable_data<int, CPUContext>();
    for (int i = 0; i < st.size(); i++) Sdata[i] = st[i];
    for (int i = 0; i < ed.size(); i++) Edata[i] = ed[i];
    for (int i = 0; i < keep_dims.size(); i++) Kdata[i] = keep_dims[i];

    //  make tasks
    process_axes.clear();
    for (int i = 0; i < st.size(); i++) {
        int cropping_size = ed[i] - st[i];
        int reducing_size = (int)Input(0).dim(i) - cropping_size;
        if (reducing_size > 0)
            process_axes.push_back({ reducing_size, i });
    }
    std::sort(process_axes.begin(), process_axes.end(),
        std::greater< pair<int, int> >());
}

template <class Context>
void CropOp<Context>::RunOnDevice() {
    Setup();

    //  do nothing
    if (process_axes.size() == 0) {
        Output(0)->ReshapeLike(Input(0));
        Output(0)->template Copy<Context, Context>(Input(0));
        //  squeeze dimensions
        vector<TIndex> squeeze_shape;
        for (int i = 0; i < keep_dims.size(); i++)
            if (keep_dims[i]) squeeze_shape.push_back(Output(0)->dim(i));
        Output(0)->Reshape(squeeze_shape);
        return;
    }

     //  select source & dest
    source = &Input(0);
    if (process_axes.size() % 2 == 1) dest = Output(0);
    else dest = &navigator;

    for (auto& task : process_axes) {
        axis = task.second;
        vector<TIndex> dims = source->dims();
        inner_dim = source->count(axis + 1);
        dim = source->dim(axis);
        dims[axis] = ed[axis] - st[axis];
        dest->Reshape(dims);
        if (XIsType(Input(0), float)) RunWithType<float>();
        else if (XIsType(Input(0), int)) RunWithType<int>();
        else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "int32" });
        //  allow buffer to protect X if the num of tasks >= 2
        std::swap(source, dest);
        if (process_axes.size() % 2 == 1) {
            if (dest == &Input(0)) dest = &navigator;
        } else {
            if (dest == &Input(0)) dest = Output(0);
        }
    }

    //  squeeze dimensions
    vector<TIndex> squeeze_shape;
    for (int i = 0; i < keep_dims.size(); i++) 
        if (keep_dims[i]) squeeze_shape.push_back(Output(0)->dim(i));
    Output(0)->Reshape(squeeze_shape);
}

DEPLOY_CPU(Crop);
#ifdef WITH_CUDA
DEPLOY_CUDA(Crop);
#endif
OPERATOR_SCHEMA(Crop).NumInputs(1).NumOutputs(1);

template <class Context>
void CropGradientOp<Context>::Setup() {
    st.assign(Input(0).ndim(), 0);
    ed.assign(Input(0).ndim(), 0);
    keep_dims.resize(Input(0).ndim(), 0);
    Tensor* tst = ws()->GetTensor("/mnt/" + anchor() + "/crop/starts");
    Tensor* ted = ws()->GetTensor("/mnt/" + anchor() + "/crop/ends");
    Tensor* tkd = ws()->GetTensor("/mnt/" + anchor() + "/crop/keep_dims");;
    auto* Sdata = tst->template mutable_data<int, CPUContext>();
    auto* Edata = ted->template mutable_data<int, CPUContext>();
    auto* Kdata = tkd->template mutable_data<int, CPUContext>();
    for (int i = 0; i < tst->count(); i++) st[i] = Sdata[i];
    for (int i = 0; i < ted->count(); i++) ed[i] = Edata[i];
    for (int i = 0; i < tkd->count(); i++) keep_dims[i] = Kdata[i];
    //  make tasks
    process_axes.clear();
    for (int i = 0; i < st.size(); i++) {
        int cropping_size = ed[i] - st[i];
        int reducing_size = (int)Input(0).dim(i) - cropping_size;
        if (reducing_size > 0) process_axes.push_back({ reducing_size, i });
    }
    std::sort(process_axes.begin(), process_axes.end(),
        std::greater< pair<int, int> >());
    std::reverse(process_axes.begin(), process_axes.end());
}

template <class Context> template <typename T>
void CropGradientOp<Context>::RunWithType() {
    const T* dYdata; T* dXdata;
    if (source == &navigator) {
        dYdata = ws()->template caches<T, Context>({ source->count() })[0];
    } else { dYdata = source->template data<T, Context>(); }
    if (dest == &navigator) {
        dXdata = ws()->template caches<T, Context>({ dest->count() })[0];
    } else { dXdata = dest->template mutable_data<T, Context>(); }
    
    kernel::Crop1DGrad<T, Context>(dest->count(),
        Input(0).dim(axis), dim, inner_dim,
            st[axis], ed[axis], dYdata, dXdata);
}

template <class Context>
void CropGradientOp<Context>::RunOnDevice() {
    Setup();

    //  expand dimensions
    vector<TIndex> expand_shape(keep_dims.size(), 1);
    vector<TIndex> keep_axes;
    for (int i = 0; i < keep_dims.size(); i++)
        if (keep_dims[i]) keep_axes.push_back(i);
    CHECK_EQ(keep_axes.size(), Input(-1).ndim());
    for (int i = 0; i < keep_axes.size(); i++)
        expand_shape[keep_axes[i]] = Input(-1).dim(i);
    Input(-1).Reshape(expand_shape);

    //  do nothing 
    if (process_axes.size() == 0) {
        Output(0)->ReshapeLike(Input(-1));
        Output(0)->template Copy<Context, Context>(Input(-1));
        return;
    }

    //  select source & buffer
    source = &Input(-1);
    if (process_axes.size() % 2 == 1) dest = Output(0);
    else dest = &navigator;

    for (auto& task : process_axes) {
        axis = task.second;
        vector<TIndex> dims = source->dims();
        inner_dim = source->count(axis + 1);
        dim = source->dim(axis);
        dims[axis] = Input(0).dim(axis),
        dest->Reshape(dims);
        if (XIsType(Input(0), float)) RunWithType<float>();
        else if (XIsType(Input(0), int)) RunWithType<int>();
        else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "int32" });
        //  allow buffer to protect X if the num of tasks >= 2
        std::swap(source, dest);
        if (process_axes.size() % 2 == 1) {
            if (dest == &Input(-1)) dest = &navigator;
        } else {
            if (dest == &Input(-1)) dest = Output(0);
        }
    }
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