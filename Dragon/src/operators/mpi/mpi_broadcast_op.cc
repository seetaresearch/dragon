#include "utils/math_functions.h"
#include "operators/mpi/mpi_broadcast_op.h"

#ifdef WITH_MPI

namespace dragon {

template <class Context> template <typename T>
void MPIBroadcastOp<Context>::RunImpl() {
    if (comm_rank_ == comm_root_) {
        auto* x = X(0).template mutable_data<T, Context>();
        BCast(x, X(0).count());
        Y(0)->CopyFrom(X(0), ctx());
    } else {
        auto* y = Y(0)->template mutable_data<T, Context>();
        BCast(y, Y(0)->count());
    }
}

template <class Context>
void MPIBroadcastOp<Context>::RunOnDevice() {
    CHECK(comm_ != MPI_COMM_NULL)
        << "\nMPIBroadcastOp, name: " << name()
        << ", does not belong to any group, can't run.";

    int ndim;
    vec64_t dims;
    if (comm_rank_ == comm_root_) {
        ndim = X(0).ndim();
        for (int i = 0; i < X(0).ndim(); i++)
            dims.emplace_back(X(0).dim(i));
    }
    BCast(&ndim, 1);
    if (dims.empty()) dims.resize((size_t)ndim, 0);
    BCast(dims.data(), ndim);
    Y(0)->Reshape(dims);

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
void MPIBroadcastGradientOp<Context>::RunImpl() {
    if (comm_rank_ == comm_root_) {
        auto* dy = X(-1).template mutable_data<T, Context>();
        auto* dx = Y(0)->template mutable_data<T, Context>();
        math::Copy(Y(0)->count(), dy, dx, ctx());
        for (int i = 0; i < comm_size_; i++) {
            if (i == comm_root_) continue;
            Recv(dy, Y(0)->count(), i);
            math::Add(Y(0)->count(), dy, dx, dx, ctx());
        }
    } else {
        auto* dy = X(-1).template data<T, Context>();
        Send(dy, X(-1).count(), comm_root_);
    }
}

template <class Context>
void MPIBroadcastGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(-1));

    if (XIsType(X(-1), int8_t)) {
        RunImpl<int8_t>();
    } else if (XIsType(X(-1), int)) {
        RunImpl<int>();
    } else if (XIsType(X(-1), int64_t)) {
        RunImpl<int64_t>();
    } else if (XIsType(X(-1), float16)) {
        RunImpl<float16>();
    } else if (XIsType(X(-1), float)) {
        RunImpl<float>();
    } else {
        LOG(FATAL) << DTypeString(X(-1), {
            "int8", "int32", "int64",
            "float16", "float32",
        });
    }
}

DEPLOY_CPU(MPIBroadcast);
#ifdef WITH_CUDA
DEPLOY_CUDA(MPIBroadcast);
#endif

DEPLOY_CPU(MPIBroadcastGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(MPIBroadcastGradient);
#endif

OPERATOR_SCHEMA(MPIBroadcast)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(MPIBroadcastGradient)
     /* dY */
    .NumInputs(1)
     /* dX */
    .NumOutputs(1);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ GO(0) }),
            vector<string>({ GI(0) })
        );
    }
};

}  // namespace

REGISTER_GRADIENT(MPIBroadcast, GradientMaker);

}  // namespace dragon

#endif  // WITH_MPI