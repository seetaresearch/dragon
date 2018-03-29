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
    vector< pair<int, int> > process_axes;
    for (int i = 0; i < Input(0).ndim(); i++)
        if (multiples(i) > 1) process_axes.push_back({ multiples(i), i });
    std::sort(process_axes.begin(), process_axes.end());

    //  do nothing 
    if (process_axes.size() == 0) {
        Output(0)->ReshapeLike(Input(0));
        Output(0)->Share(Input(0));
        return;
    }

    //  select source & dest
    source = &Input(0);
    if (process_axes.size() % 2 == 1) dest = Output(0);
    else dest = ws()->GetBuffer();

    for (auto& task : process_axes) {
        axis = task.second; multiple = task.first;
        if (Input(0).template IsType<float>()) TileRunWithType<float>();
        else LOG(FATAL) << "Unsupported input types.";
        //  allow buffer to protect X if the num of tasks >= 2
        std::swap(source, dest);
        if (process_axes.size() % 2 == 1) {
            if (dest == &Input(0)) dest = ws()->GetBuffer();
        } else {
            if (dest == &Input(0)) dest = Output(0);
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
    vector< pair<int, int> > process_axes;
    for (int i = 0; i < Input(0).ndim(); i++)
        if (multiples(i) > 1) process_axes.push_back({ multiples(i), i });
    std::sort(process_axes.begin(), process_axes.end());
    std::reverse(process_axes.begin(), process_axes.end());

    //  do nothing 
    if (process_axes.size() == 0) {
        Output(0)->ReshapeLike(Input(-1));
        Output(0)->Share(Input(-1));
        return;
    }

    //  select source & buffer
    source = &Input(-1);
    if (process_axes.size() % 2 == 1) dest = Output(0);
    else dest = ws()->GetBuffer();

    for (auto& task : process_axes) {
        axis = task.second; multiple = task.first;
        if (Input(0).template IsType<float>()) TileRunWithType<float>();
        else LOG(FATAL) << "Unsupported input types.";
        //  allow buffer to protect X if the num of tasks >= 2
        std::swap(source, dest);
        if (process_axes.size() % 2 == 1) {
            if (dest == &Input(-1)) dest = ws()->GetBuffer();
        } else {
            if (dest == &Input(-1)) dest = Output(0);
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