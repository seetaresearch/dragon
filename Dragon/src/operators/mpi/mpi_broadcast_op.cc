#include "utils/math_functions.h"
#include "operators/mpi/mpi_broadcast_op.h"

#ifdef WITH_MPI

namespace dragon {

template <class Context> template <typename T>
void MPIBroadcastOp<Context>::RunWithType() {
    if (comm_rank == comm_root) {
#ifdef WITH_MPI_CUDA
        auto* Xdata = Input(0).template mutable_data<T, Context>();
#else
        auto* Xdata = Input(0).template mutable_data<T, CPUContext>();
#endif
        MPI_Bcast(Xdata, Input(0).count(), mpi_dtype(), comm_root, comm);
        Output(0)->template Copy<Context, Context>(Input(0));
    } else { 
#ifdef WITH_MPI_CUDA
        auto* Ydata = Output(0)->template mutable_data<T, Context>();
#else
        auto* Ydata = Output(0)->template mutable_data<T, CPUContext>();
#endif
        MPI_Bcast(Ydata, Output(0)->count(), mpi_dtype(), comm_root, comm);
    }
}

template <class Context>
void MPIBroadcastOp<Context>::RunOnDevice() {
    CHECK(comm != MPI_COMM_NULL)
        << "\nMPIBroadcastOp, name: " << name()
        << ", does not belong to any group, can't run.";

    size_t ndim[1];
    TIndex* dims = nullptr;
    if (comm_rank == comm_root) {
        ndim[0] = Input(0).ndim();
        dims = new TIndex[ndim[0]];
        for (int i = 0; i < Input(0).ndim(); i++)
            dims[i] = Input(0).dim(i);
    }
    MPI_Bcast(ndim, 1, MPI_UNSIGNED_LONG_LONG, comm_root, comm);
    if (dims == nullptr) dims = new TIndex[ndim[0]];
    MPI_Bcast(dims, (int)ndim[0], MPI_LONG_LONG, comm_root, comm);
    vector<TIndex> _dims;
    for (int i = 0; i < (int)ndim[0]; i++)  _dims.push_back(dims[i]);
    Output(0)->Reshape(_dims);

    if (dtype == "FLOAT32") RunWithType<float>();
    else LOG(FATAL) << "Unsupported input type: " << dtype;
}

DEPLOY_CPU(MPIBroadcast);
#ifdef WITH_CUDA
DEPLOY_CUDA(MPIBroadcast);
#endif
OPERATOR_SCHEMA(MPIBroadcast).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void MPIBroadcastGradientOp<Context>::RunWithType() {
    if (comm_rank == comm_root) {
#ifdef WITH_MPI_CUDA
        auto* dYdata = Input(-1).template mutable_data<T, Context>();
        auto* dXdata = Output(0)->template mutable_data<T, Context>();
        ctx().template Copy<T, Context, Context>(
            Output(0)->count(), dXdata, dYdata);
#else
        auto* dYdata = Input(-1).template mutable_data<T, CPUContext>();
        auto* dXdata = Output(0)->template mutable_data<T, CPUContext>();
        CPUContext::template Copy<T, CPUContext, CPUContext>(
            Output(0)->count(), dXdata, dYdata);
#endif
        for (int i = 0; i < comm_size; i++) {
            if (i == comm_root) continue;
            MPI_Recv(dYdata, Output(0)->count(), mpi_dtype(),
                i, 0, comm, MPI_STATUS_IGNORE);
#ifdef WITH_MPI_CUDA
            math::Add<T, Context>(Output(0)->count(),
                dYdata, dXdata, dXdata);
#else
            math::Add<T, CPUContext>(Output(0)->count(), 
                dYdata, dXdata, dXdata);
#endif
        }
    }
    else {
#ifdef WITH_MPI_CUDA
        auto* dYdata = Input(-1).template data<T, Context>();
#else
        auto* dYdata = Input(-1).template data<T, CPUContext>();
#endif
        MPI_Send(dYdata, Input(-1).count(), mpi_dtype(), comm_root, 0, comm);
    }
}

template <class Context>
void MPIBroadcastGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(-1));

    if (dtype == "FLOAT32") RunWithType<float>();
    else LOG(FATAL) << "Unsupported input type: " << dtype;
}

DEPLOY_CPU(MPIBroadcastGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(MPIBroadcastGradient);
#endif
OPERATOR_SCHEMA(MPIBroadcastGradient).NumInputs(1).NumOutputs(1);

class GetMPIBroadcastGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetMPIBroadcastGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(MPIBroadcast, GetMPIBroadcastGradient);

}    // namespace dragon

#endif // WITH_MPI