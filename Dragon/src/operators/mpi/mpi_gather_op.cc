#include "utils/math_functions.h"
#include "operators/mpi/mpi_gather_op.h"

#ifdef WITH_MPI

#define DTYPE this->template mpi_dtype<T>()

namespace dragon {

template <class Context> template <typename T>
void MPIGatherOp<Context>::RunWithType() {
    if (comm_rank == comm_root) {
        Output(comm_rank)->template CopyFrom<Context>(Input(0), ctx());
        for (int i = 0; i < comm_size; i++) {
            if (i == comm_root) continue;
            auto* Ydata = Output(i)->template mutable_data<T, Context>();
            MPI_Recv(Ydata, Output(i)->count(), DTYPE,
                i, 0, this->comm, MPI_STATUS_IGNORE);
        }
    }
    else {
        auto* Xdata = Input(0).template data<T, Context>();
        MPI_Send(Xdata, Input(0).count(), DTYPE, comm_root, 0, comm);
    }
}

template <class Context>
void MPIGatherOp<Context>::RunOnDevice() {
    CHECK_EQ(comm_size, OutputSize());
    //  reshape from root
    if (comm_rank == comm_root) Output(0)->ReshapeLike(Input(0));

    //  reshape from others
    int ndim;
    vector<int> all_ndim(comm_size, 0);
    if (comm_rank != comm_root) {
        ndim = Input(0).ndim();
        MPI_Send(&ndim, 1, MPI_INT, comm_root, 0, comm);
    } else {
        for (int i = 1; i < comm_size; i++)
            MPI_Recv(all_ndim.data() + i, 1, MPI_INT,
               i, 0, comm, MPI_STATUS_IGNORE);
    }
    if (comm_rank != comm_root) {
        MPI_Send(Input(0).dims().data(), ndim,
            MPI_LONG_LONG, comm_root, 0, comm);
    } else {
        for (int i = 1; i < comm_size; i++) {
            vector<int64_t> dims(all_ndim[i], 0);
            MPI_Recv(dims.data(), all_ndim[i], MPI_LONG_LONG,
                i, 0, comm, MPI_STATUS_IGNORE);
            Output(i)->Reshape(dims);
        }
    }

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
            auto* dYdata = Input(comm_rank + 1).template data<T, Context>();
            MPI_Send(dYdata, Input(comm_rank + 1).count(), DTYPE, i, 0, comm);
        }
    }
    else{
        auto* dXdata = Output(0)->template mutable_data<T, Context>();
        MPI_Recv(dXdata, Output(0)->count(), DTYPE,
            comm_root, 0, comm, MPI_STATUS_IGNORE);
    }
}

template <class Context>
void MPIGatherGradientOp<Context>::RunOnDevice() {
    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), int8_t)) RunWithType<int8_t>();
    else if (XIsType(Input(0), int)) RunWithType<int>();
    else if (XIsType(Input(0), int64_t)) RunWithType<int64_t>();
    else if (XIsType(Input(0), float16)) RunWithType<float16>();
    else if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0),
        { "int8", "int32", "int64", "float16", "float32" });
}

DEPLOY_CPU(MPIGatherGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(MPIGatherGradient);
#endif

OPERATOR_SCHEMA(MPIGatherGradient)
    .NumInputs(2, INT_MAX).NumOutputs(1);

class GetMPIGatherGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetMPIGatherGradient);
    vector<OperatorDef> MakeDefs() override {
        vector<string> inputs({ I(0) });
        for (auto out : def.output()) inputs.push_back(out + "_grad");
        return SingleDef(def.type() + "Gradient", "",
            inputs, vector<string>({ GI(0) }));
    }
};

REGISTER_GRADIENT(MPIGather, GetMPIGatherGradient);

}  // namespace dragon

#undef DTYPE

#endif  // WITH_MPI