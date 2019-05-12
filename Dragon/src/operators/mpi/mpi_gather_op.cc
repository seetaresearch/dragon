#include "utils/math_functions.h"
#include "operators/mpi/mpi_gather_op.h"

#ifdef WITH_MPI

namespace dragon {

template <class Context> template <typename T>
void MPIGatherOp<Context>::RunImpl() {
    if (comm_rank_ == comm_root_) {
        Y(comm_rank_)->CopyFrom(X(0), ctx());
        for (int i = 0; i < comm_size_; i++) {
            if (i == comm_root_) continue;
            auto* y = Y(i)->template mutable_data<T, Context>();
            Recv(y, Y(i)->count(), i);
        }
    } else {
        auto* x = X(0).template data<T, Context>();
        Send(x, X(0).count(), comm_root_);
    }
}

template <class Context>
void MPIGatherOp<Context>::RunOnDevice() {
    CHECK_EQ(comm_size_, YSize());
    // Reshape from root
    if (comm_rank_ == comm_root_) {
        Y(comm_rank_)->ReshapeLike(X(0));
    }

    // Reshape from others
    int ndim = X(0).ndim();
    vec32_t all_ndim(comm_size_, 0);
    if (comm_rank_ != comm_root_) {
        Send(&ndim, 1, comm_root_);
    } else {
        for (int i = 0; i < comm_size_; i++) {
            if (i == comm_root_) continue;
            Recv(all_ndim.data() + i, 1, i);
        }
    }
    if (comm_rank_ != comm_root_) {
        Send(X(0).dims().data(), ndim, comm_root_);
    } else {
        for (int i = 0; i < comm_size_; i++) {
            if (i == comm_root_) continue;
            vec64_t shape(all_ndim[i], 0);
            Recv(shape.data(), all_ndim[i], i);
            Y(i)->Reshape(shape);
        }
    }

    if (XIsType(X(0), int8_t)) {
        RunImpl<int8_t>();
    } else if (XIsType(X(0), int)) {
        RunImpl<int>();
    } else if (XIsType(X(0), int64_t)) {
        RunImpl<int64_t>();
    } else if (XIsType(X(0), float16)) {
        RunImpl<float16>();
    } else if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else {
        LOG(FATAL) << DTypeString(X(0), {
            "int8", "int32", "int64",
            "float16", "float32",
        });
    }
}

template <class Context> template <typename T>
void MPIGatherGradientOp<Context>::RunImpl() {
    if (comm_rank_ == comm_root_) {
        const auto& dY = X(comm_rank_ + 1);
        Y(0)->CopyFrom(dY, ctx());
        for (int i = 0; i < comm_size_; i++) {
            if (i == comm_root_) continue;
            auto* dy = dY.template data<T, Context>();
            Send(dy, dY.count(), i);
        }
    } else{
        auto* dx = Y(0)->template mutable_data<T, Context>();
        Recv(dx, X(0).count(), comm_root_);
    }
}

template <class Context>
void MPIGatherGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    if (XIsType(X(0), int8_t)) {
        RunImpl<int8_t>();
    } else if (XIsType(X(0), int)) {
        RunImpl<int>();
    } else if (XIsType(X(0), int64_t)) {
        RunImpl<int64_t>();
    } else if (XIsType(X(0), float16)) {
        RunImpl<float16>();
    } else if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else {
        LOG(FATAL) << DTypeString(X(0), {
            "int8", "int32", "int64",
            "float16", "float32",
        });
    }
}

DEPLOY_CPU(MPIGather);
#ifdef WITH_CUDA
DEPLOY_CUDA(MPIGather);
#endif

DEPLOY_CPU(MPIGatherGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(MPIGatherGradient);
#endif

OPERATOR_SCHEMA(MPIGather)
     /* X */
    .NumInputs(1)
     /* Y(0), ... */
    .NumOutputs(1, INT_MAX);

OPERATOR_SCHEMA(MPIGatherGradient)
     /* X, dY(0), ... */
    .NumInputs(2, INT_MAX)
     /* dX */
    .NumOutputs(1);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override {
        vector<string> inputs({ I(0) });
        for (auto e : def.output())
            inputs.push_back(e + "_grad");
        return SingleDef(
            def.type() + "Gradient", "",
            inputs, vector<string>({ GI(0) })
        );
    }
};

}  // namespace

REGISTER_GRADIENT(MPIGather, GradientMaker);

}  // namespace dragon

#endif  // WITH_MPI