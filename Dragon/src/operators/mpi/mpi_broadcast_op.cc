#include "utils/math_functions.h"
#include "operators/mpi/mpi_broadcast_op.h"

#ifdef WITH_MPI

#define DTYPE this->template mpi_dtype<T>()

namespace dragon {

template <class Context> template <typename T>
void MPIBroadcastOp<Context>::RunWithType() {
    if (comm_rank == comm_root) {
        auto* Xdata = Input(0).template mutable_data<T, Context>();
        MPI_Bcast(Xdata, Input(0).count(), DTYPE, comm_root, comm);
        Output(0)->template CopyFrom<Context>(Input(0), ctx());
    } else {
        auto* Ydata = Output(0)->template mutable_data<T, Context>();
        MPI_Bcast(Ydata, Output(0)->count(), DTYPE, comm_root, comm);
    }
}

template <class Context>
void MPIBroadcastOp<Context>::RunOnDevice() {
    CHECK(comm != MPI_COMM_NULL)
        << "\nMPIBroadcastOp, name: " << name()
        << ", does not belong to any group, can't run.";

    int ndim;
    vector<int64_t> dims;
    if (comm_rank == comm_root) {
        ndim = Input(0).ndim();
        for (int i = 0; i < Input(0).ndim(); i++)
            dims.emplace_back(Input(0).dim(i));
    }
    MPI_Bcast(&ndim, 1, MPI_INT, comm_root, comm);
    if (dims.empty()) dims.resize((size_t)ndim, 0);
    MPI_Bcast(dims.data(), ndim, MPI_LONG_LONG, comm_root, comm);
    Output(0)->Reshape(dims);

    if (XIsType(Input(0), int8_t)) RunWithType<int8_t>();
    else if (XIsType(Input(0), int)) RunWithType<int>();
    else if (XIsType(Input(0), int64_t)) RunWithType<int64_t>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { 
        "int8", "int32", "int64",
            "float16", "float32",
    });
}

DEPLOY_CPU(MPIBroadcast);
#ifdef WITH_CUDA
DEPLOY_CUDA(MPIBroadcast);
#endif
OPERATOR_SCHEMA(MPIBroadcast).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void MPIBroadcastGradientOp<Context>::RunWithType() {
    if (comm_rank == comm_root) {
        auto* dYdata = Input(-1).template mutable_data<T, Context>();
        auto* dXdata = Output(0)->template mutable_data<T, Context>();
        ctx()->template Copy<T, Context, Context>(
            Output(0)->count(), dXdata, dYdata);
        for (int i = 0; i < comm_size; i++) {
            if (i == comm_root) continue;
            MPI_Recv(dYdata, Output(0)->count(), DTYPE,
                i, 0, comm, MPI_STATUS_IGNORE);
            math::Add(Output(0)->count(),
                dYdata, dXdata, dXdata, ctx());
        }
    }
    else {
        auto* dYdata = Input(-1).template data<T, Context>();
        MPI_Send(dYdata, Input(-1).count(), DTYPE, comm_root, 0, comm);
    }
}

template <class Context>
void MPIBroadcastGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(-1));

    if (XIsType(Input(0), float16)) RunWithType<float16>();
    else if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float16", "float32" });
}

DEPLOY_CPU(MPIBroadcastGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(MPIBroadcastGradient);
#endif

OPERATOR_SCHEMA(MPIBroadcastGradient)
    .NumInputs(1).NumOutputs(1);

class GetMPIBroadcastGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetMPIBroadcastGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ GO(0) }),
            vector<string>({ GI(0) }));
    }
};

REGISTER_GRADIENT(MPIBroadcast, GetMPIBroadcastGradient);

}  // namespace dragon

#undef DTYPE

#endif  // WITH_MPI