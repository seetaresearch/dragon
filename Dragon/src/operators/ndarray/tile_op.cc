#include "operators/ndarray/tile_op.h"
#include "core/workspace.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void TileOp<Context>::TileRunWithType() {
    vector<TIndex> dims = source->dims();
    outer_dim = source->count(0, axis);
    ex_inner_dim = source->count(axis);
    dims[axis] *= multiple;
    dest->Reshape(dims);

    auto* Xdata = source->template data<T, Context>();
    auto* Ydata = dest->template mutable_data<T, Context>();
    kernel::Tile<T, Context>(dest->count(),
                                 outer_dim,
                              ex_inner_dim,
                                  multiple,
                                     Xdata,
                                     Ydata,
                                   &ctx());
}

template <class Context>
void TileOp<Context>::RunOnDevice() {
    //  parse tasks from desc
    CHECK_EQ(multiples_desc.size(), input(0).ndim())
        << "\nThe num of dimensions of input is " << input(0).ndim()
        << ", but provided " << multiples_desc.size() << " multiples.";
    vector< pair<int, int> > process_axes;
    for (int i = 0; i < multiples_desc.size(); i++) {
        int mult = ws()->GetTensor(multiples_desc[i])->template data<int, CPUContext>()[0];
        if (mult > 1) process_axes.push_back({ mult, i });
    }
    std::sort(process_axes.begin(), process_axes.end());

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
        axis = task.second; multiple = task.first;
        if (input(0).template IsType<float>()) TileRunWithType<float>();
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

DEPLOY_CPU(Tile);
#ifdef WITH_CUDA
DEPLOY_CUDA(Tile);
#endif
OPERATOR_SCHEMA(Tile).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void TileGradientOp<Context>::TileRunWithType() {
    vector<TIndex> dims = source->dims();
    dims[axis] /= multiple;
    dest->Reshape(dims);
    outer_dim = dest->count(0, axis);
    ex_inner_dim = dest->count(axis);

    auto* dYdata = source->template data<T, Context>();
    auto* dXdata = dest->template mutable_data<T, Context>();
    kernel::TileGrad<T, Context>(dest->count(),
                                     outer_dim,
                                  ex_inner_dim,
                                      multiple,
                                        dYdata,
                                        dXdata,
                                       &ctx());
}

template <class Context>
void TileGradientOp<Context>::RunOnDevice() {
    //  parse tasks from desc
    CHECK_EQ(multiples_desc.size(), input(-1).ndim())
        << "\nThe num of dimensions of input is " << input(-1).ndim()
        << ", but provided " << multiples_desc.size() << " multiples.";
    vector< pair<int, int> > process_axes;
    for (int i = 0; i < multiples_desc.size(); i++) {
        int mult = ws()->GetTensor(multiples_desc[i])->template data<int, CPUContext>()[0];
        if (mult > 1) process_axes.push_back({ mult, i });
    }
    std::sort(process_axes.begin(), process_axes.end());
    std::reverse(process_axes.begin(), process_axes.end());

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
        axis = task.second; multiple = task.first;
        if (input(0).template IsType<float>()) TileRunWithType<float>();
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

DEPLOY_CPU(TileGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(TileGradient);
#endif
OPERATOR_SCHEMA(TileGradient).NumInputs(1).NumOutputs(1);

class GetTileGradient final : public GradientMakerBase { 
 public:
    GRADIENT_MAKER_CTOR(GetTileGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(Tile, GetTileGradient);

}    // namespace dragon