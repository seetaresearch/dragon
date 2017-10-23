#include "operators/mpi/mpi_broadcast_op.h"
#include "utils/math_functions.h"

#ifdef WITH_MPI

namespace dragon {

template <class Context> template <typename T>
void MPIBroadcastOp<Context>::RunWithType() {
    if (this->comm_rank == this->comm_root) {
#ifdef WITH_MPI_CUDA
        auto* Xdata = input(0).template mutable_data<T, Context>();
#else
        auto* Xdata = input(0).template mutable_data<T, CPUContext>();
#endif
        MPI_Bcast(Xdata, input(0).count(), MPI_FLOAT, this->comm_root, this->comm);
        output(0)->Share(input(0));
    } else { 
#ifdef WITH_MPI_CUDA
        auto* Ydata = output(0)->template mutable_data<T, Context>();
#else
        auto* Ydata = output(0)->template mutable_data<T, CPUContext>();
#endif
        MPI_Bcast(Ydata, output(0)->count(), MPI_FLOAT, this->comm_root, this->comm);
    }
}

template <class Context>
void MPIBroadcastOp<Context>::RunOnDevice() {
    CHECK(this->comm != MPI_COMM_NULL)
        << "\nMPIBroadcastOp, name: " << name()
        << ", does not belong to any group, can't run.";

    size_t ndim[1];
    TIndex* dims = nullptr;
    if (this->comm_rank == this->comm_root) {
        ndim[0] = input(0).ndim();
        dims = new TIndex[ndim[0]];
        for (int i = 0; i < input(0).ndim(); i++)
            dims[i] = input(0).dim(i);
    }
    MPI_Bcast(ndim, 1, MPI_UNSIGNED_LONG_LONG, this->comm_root, this->comm);
    if (dims == nullptr) dims = new TIndex[ndim[0]];
    MPI_Bcast(dims, 4, MPI_LONG_LONG, this->comm_root, this->comm);
    vector<TIndex> _dims;
    for (int i = 0; i < ndim[0]; i++)  _dims.push_back(dims[i]);
    output(0)->Reshape(_dims);

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(MPIBroadcast);
#ifdef WITH_CUDA
DEPLOY_CUDA(MPIBroadcast);
#endif
OPERATOR_SCHEMA(MPIBroadcast).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void MPIBroadcastGradientOp<Context>::RunWithType() {
    if (this->comm_rank == this->comm_root) {
#ifdef WITH_MPI_CUDA
        auto* dYdata = input(-1).template mutable_data<T, Context>();
        auto* dXdata = output(0)->template mutable_data<T, Context>();
        ctx().template Copy<T, Context, Context>(output(0)->count(), dXdata, dYdata);
#else
        auto* dYdata = input(-1).template mutable_data<T, CPUContext>();
        auto* dXdata = output(0)->template mutable_data<T, CPUContext>();
        CPUContext cpu_ctx;
        cpu_ctx.template Copy<T, CPUContext, CPUContext>(output(0)->count(), dXdata, dYdata);
#endif
        for (int i = 0; i < this->comm_size; i++) {
            if (i == this->comm_root) continue;
            MPI_Recv(dYdata, output(0)->count(), MPI_FLOAT, i, 0, this->comm, MPI_STATUS_IGNORE);
#ifdef WITH_MPI_CUDA
            math::Add<T, Context>(output(0)->count(), dYdata, dXdata, dXdata);
#else
            math::Add<T, CPUContext>(output(0)->count(), dYdata, dXdata, dXdata);
#endif
        }
    }
    else {
#ifdef WITH_MPI_CUDA
        auto* dYdata = input(-1).template data<T, Context>();
#else
        auto* dYdata = input(-1).template data<T, CPUContext>();
#endif
        MPI_Send(dYdata, input(-1).count(), MPI_FLOAT, this->comm_root, 0, this->comm);
    }
}

template <class Context>
void MPIBroadcastGradientOp<Context>::RunOnDevice() {
    output(0)->ReshapeLike(input(-1));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
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