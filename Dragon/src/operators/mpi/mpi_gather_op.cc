#include "utils/math_functions.h"
#include "operators/mpi/mpi_gather_op.h"

#ifdef WITH_MPI

namespace dragon {

template <class Context> template <typename T>
void MPIGatherOp<Context>::RunWithType() {
    if (comm_rank == comm_root) {
        Output(comm_rank)->template CopyFrom<Context>(Input(0), ctx());
        for (int i = 0; i < comm_size; i++) {
            if (i == comm_root) continue;
#ifdef WITH_MPI_CUDA
            auto* Ydata = Output(i)->template mutable_data<T, Context>();
#else
            auto* Ydata = Output(i)->template mutable_data<T, CPUContext>();
#endif
            MPI_Recv(Ydata, Output(i)->count(), mpi_dtype(),
                i, 0, this->comm, MPI_STATUS_IGNORE);
        }
    }
    else {
#ifdef WITH_MPI_CUDA
        auto* Xdata = Input(0).template data<T, Context>();
#else
        auto* Xdata = Input(0).template data<T, CPUContext>();
#endif
        MPI_Send(Xdata, Input(0).count(), mpi_dtype(),
            comm_root, 0, comm);
    }
}

template <class Context>
void MPIGatherOp<Context>::RunOnDevice() {
    CHECK_EQ(comm_size, OutputSize());
    //  reshape from root
    if (comm_rank == comm_root) Output(0)->ReshapeLike(Input(0));

    //  reshape from others
    size_t* all_ndim = new size_t[comm_size];
    size_t ndim[1];
    if (comm_rank != comm_root) {
        ndim[0] = Input(0).ndim();
        MPI_Send(ndim, 1, MPI_UNSIGNED_LONG_LONG, comm_root, 0, comm);
    } else {
        for (int i = 1; i < comm_size; i++)
            MPI_Recv(all_ndim + i, 1, MPI_UNSIGNED_LONG_LONG,
               i, 0, comm, MPI_STATUS_IGNORE);
    }
    if (comm_rank != comm_root) {
        MPI_Send(Input(0).dims().data(), (int)ndim[0],
            MPI_LONG_LONG, comm_root, 0, comm);
    } else {
        for (int i = 1; i < comm_size; i++) {
            TIndex* dims = new TIndex[all_ndim[i]];
            MPI_Recv(dims, (int)all_ndim[i], MPI_LONG_LONG, 
                i, 0, comm, MPI_STATUS_IGNORE);
            vector<TIndex> dims_;
            for (int j = 0; j < (int)all_ndim[i]; j++)
                dims_.push_back(dims[j]);
            Output(i)->Reshape(dims_);
        }
    }

    if (dtype == "FLOAT32") RunWithType<float>();
    else LOG(FATAL) << "Unsupported input type: " << dtype;
}

DEPLOY_CPU(MPIGather);
#ifdef WITH_CUDA
DEPLOY_CUDA(MPIGather);
#endif
OPERATOR_SCHEMA(MPIGather).NumInputs(1).NumOutputs(1, INT_MAX);

template <class Context> template <typename T>
void MPIGatherGradientOp<Context>::RunWithType() {
    if (comm_rank == comm_root) {
        Output(0)->template CopyFrom<Context>(
            Input(this->comm_rank + 1), ctx());
        for (int i = 0; i < comm_size; i++) {
            if (i == comm_root) continue;
#ifdef WITH_MPI_CUDA
            auto* dYdata = Input(comm_rank + 1).template data<T, Context>();
#else
            auto* dYdata = Input(comm_rank + 1).template data<T, CPUContext>();
#endif
            MPI_Send(dYdata, Input(comm_rank + 1).count(), mpi_dtype(), i, 0, comm);
        }
    }
    else{
#ifdef WITH_MPI_CUDA
        auto* dXdata = Output(0)->template mutable_data<T, Context>();
#else
        auto* dXdata = Output(0)->template mutable_data<T, CPUContext>();
#endif
        MPI_Recv(dXdata, Output(0)->count(), mpi_dtype(),
            comm_root, 0, comm, MPI_STATUS_IGNORE);
    }
}

template <class Context>
void MPIGatherGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (this->dtype == "FLOAT32") RunWithType<float>();
    else LOG(FATAL) << "Unsupported input type: " << this->dtype;
}

DEPLOY_CPU(MPIGatherGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(MPIGatherGradient);
#endif
OPERATOR_SCHEMA(MPIGatherGradient).NumInputs(2, INT_MAX).NumOutputs(1);

class GetMPIGatherGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetMPIGatherGradient);
    vector<OperatorDef> MakeDefs() override {
        vector<string> inputs(1, I(0));
        for (auto out : def.output()) inputs.push_back(out + "_grad");
        return SingleDef(def.type() + "Gradient", "", 
            inputs, vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(MPIGather, GetMPIGatherGradient);

}    // namespace dragon

#endif // WITH_MPI