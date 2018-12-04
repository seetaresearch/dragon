#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/vision/drop_block_op.h"

namespace dragon {

template <class Context> template <typename T>
void DropBlock2dOp<Context>::RunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    if (phase() == "TEST") {
        if (Output(0) != &Input(0)) {
            ctx()->template Copy<T, Context, Context>(
                Output(0)->count(), Ydata, Xdata);
        }
    } else if (phase() == "TRAIN") {
        auto* mask = ws()->CreateTensor(
            "/mnt/" + anchor() + "/drop_block/mask");
        auto* norm = ws()->CreateTensor(
            "/mnt/" + anchor() + "/drop_block/norm");
        mask->ReshapeLike(Input(0));
        norm->Reshape(vector<TIndex>({ (TIndex)1 }));

        auto WSdata = ws()->template caches<Context>({
            n * c * seed_h * seed_w * sizeof(uint32_t),
                mask->count() * sizeof(int),
                    mask->count() * sizeof(float)});

        auto* Mdata = mask->template mutable_data<uint8_t, Context>();
        auto* Ndata = norm->template mutable_data<float, CPUContext>();

        // Fill the mask with ones
        math::Set<int, Context>(mask->count(),
            1, (int*)WSdata[1], ctx());

        // Generate 2d mask from seed region
        kernel::DropBlock2d<Context>(n, c, h, w,
            seed_h, seed_w, block_size, gamma, data_format,
                (uint32_t*)WSdata[0], (int*)WSdata[1], ctx());

        // Convert to float mask for counting
        kernel::TypeA2B<int, float, Context>(mask->count(),
            (int*)WSdata[1], (float*)WSdata[2], ctx());

        // Convert to uint8 mask for applying
        kernel::TypeA2B<int, uint8_t, Context>(mask->count(),
            (int*)WSdata[1], Mdata, ctx());

        // Count && Apply
        float normalizer = math::ASum<float, Context>(
            mask->count(), (float*)WSdata[2]);
        normalizer = std::max(normalizer, 1.f);
        Ndata[0] = normalizer = mask->count() / normalizer;

        kernel::ApplyMask<T, uint8_t, Context>(mask->count(),
            normalizer, Xdata, Mdata, Ydata, ctx());

    } else LOG(FATAL) << "Incorrect Op phase: " << phase();
}

template <class Context>
void DropBlock2dOp<Context>::RunOnDevice() {
    ctx()->set_stream_id(0);  // Enforce SyncStream

    if (data_format == "NCHW") {
        n = Input(0).dim(0), c = Input(0).dim(1);
        h = Input(0).dim(2), w = Input(0).dim(3);
    } else if (data_format == "NHWC") {
        n = Input(0).dim(0), c = Input(0).dim(-1);
        h = Input(0).dim(1), w = Input(0).dim(2);
    }

    seed_h = h - block_size + 1;
    seed_w = w - block_size + 1;

    CHECK(seed_h > 0 && seed_w > 0) 
        << "\nExcepted block_size <= feat_size.";

    Output(0)->ReshapeLike(Input(0));

    if (decrement > 0 && apply_prob > keep_prob()) {
        apply_prob -= decrement;
    } else { apply_prob = keep_prob(); }

    gamma = (1.f - apply_prob) / (block_size * block_size);
    gamma *= (alpha * (h * w) / (seed_h * seed_w));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(DropBlock2d);
#ifdef WITH_CUDA
DEPLOY_CUDA(DropBlock2d);
#endif
OPERATOR_SCHEMA(DropBlock2d)
    .NumInputs(1).NumOutputs(1)
    .Inplace({ { 0, 0 } });

template <class Context> template <typename T>
void DropBlock2dGradientOp<Context>::RunWithType() {
    auto* mask = ws()->GetTensor(
        "/mnt/" + anchor() + "/drop_block/mask");
    auto* norm = ws()->GetTensor(
        "/mnt/" + anchor() + "/drop_block/norm");

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* Mdata   = mask->template data<uint8_t, Context>();
    auto* Ndata = norm->template mutable_data<float, CPUContext>();

    if (phase() == "TEST") { NOT_IMPLEMENTED; }
    else if (phase() == "TRAIN") {
        kernel::ApplyMask<T, uint8_t, Context>(mask->count(),
            Ndata[0], dYdata, Mdata, dXdata, ctx());
    } else LOG(FATAL) << "Incorrect Op phase: " << phase();
}

template <class Context>
void DropBlock2dGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
}

DEPLOY_CPU(DropBlock2dGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(DropBlock2dGradient);
#endif
OPERATOR_SCHEMA(DropBlock2dGradient)
    .NumInputs(2).NumOutputs(1)
    .Inplace({ { 1, 0 } });

class GetDropBlock2dGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetDropBlock2dGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {O(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(DropBlock2d, GetDropBlock2dGradient);

}    // namepsace dragon