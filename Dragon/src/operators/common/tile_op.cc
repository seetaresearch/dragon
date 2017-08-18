#include "operators/common/tile_op.h"
#include "core/workspace.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context> template <typename T>
void TileOp<Context>::TileRunWithType() {
    vector<TIndex> dims = source->dims();
    outer_dim = source->count(0, axis);
    dim = source->dim(axis);
    inner_dim = source->count(axis);
    dims[axis] *= multiple;
    dest->Reshape(dims);

    auto* Xdata = source->template data<T, Context>();
    auto* Ydata = dest->template mutable_data<T, Context>();
    kernel::Tile(dest->count(), outer_dim, inner_dim, dim,
                                                 multiple, 
                                                    Xdata, 
                                                    Ydata, 
                                                  &ctx());

    //  swap source & dest
    std::swap(source, dest);
}

template <class Context>
void TileOp<Context>::RunOnDevice() {
    CHECK_EQ(multiples.size(), input(0).ndim());

    //  do nothing 
    if (process_axes.size() == 0) {
        output(0)->ReshapeLike(input(-1));
        output(0)->Share(input(-1));
        return;
    }

    //  select source & dest
    source = &input(0);
    if (process_axes.size() % 2 == 1) dest = output(0);
    else dest = ws()->GetBuffer();

    for (auto& task : process_axes) {
        axis = task.first; multiple = task.second;
        if (input(0).template IsType<float>()) TileRunWithType<float>();
        else LOG(FATAL) << "unsupported input types.";

        //  allow buffer to protect X if num axes >= 2
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
    dim = dest->dim(axis);
    inner_dim = dest->count(axis);

    auto* dYdata = source->template data<T, Context>();
    auto* dXdata = dest->template mutable_data<T, Context>();
    kernel::TileGrad(dest->count(), outer_dim, inner_dim, dim,
                                                     multiple, 
                                                       dYdata, 
                                                       dXdata, 
                                                      &ctx());

    //  swap source & dest
    std::swap(source, dest);
}

template <class Context>
void TileGradientOp<Context>::RunOnDevice() {
    CHECK_EQ(multiples.size(), input(-1).ndim());

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
        axis = task.first; multiple = task.second;
        if (input(0).template IsType<float>()) TileRunWithType<float>();
        else LOG(FATAL) << "unsupported input types.";

        //  allow buffer to protect dY if num axes >= 2
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