#include "core/workspace.h"
#include "utils/math_functions.h"
#include "operators/update/collective_update_op.h"

namespace dragon {

#ifdef WITH_MPI

template <class Context>
void CollectiveUpdateOp<Context>::InitMPI() {
    comm = (MPI_Comm)OperatorBase::Arg<int64_t>("comm", 0);
    group = (MPI_Group)OperatorBase::Arg<int64_t>("group", 0);
    int world_root = OperatorBase::Arg<int64_t>("root", 0);
    CHECK(comm != MPI_COMM_NULL) << "\nMPI is not initialized.";
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &comm_rank);
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_translate_ranks(
        world_group, 1, &world_root,
            group, &comm_root);
    CHECK(comm_root != MPI_UNDEFINED)
        << "\nMPI root is not included in layer group.";
}

template <class Context>
void CollectiveUpdateOp<Context>::InitNCCL() {
#ifdef WITH_NCCL
    ncclUniqueId id;
    if (comm_rank == comm_root) NCCL_CHECK(ncclGetUniqueId(&id));
    MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, comm_root, comm);
    ctx()->SwitchToDevice();
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, comm_size, id, comm_rank));
#else
    LOG(FATAL) << "NCCL was not compiled.";
#endif
}

template <class Context> template <typename T>
void CollectiveUpdateOp<Context>::MPIAllReduce(
    Tensor*                 tensor,
    MPI_Datatype            dtype) {
    int64_t count = tensor->count();
    MPI_Request recv_req;
    int64_t segment_size = count / comm_size;
    int64_t residual = count % comm_size;
    vector<int64_t> segment_sizes(comm_size, segment_size);
    for (int i = 0; i < residual; i++) segment_sizes[i]++;
    vector<int64_t> segment_ends(comm_size);
    segment_ends[0] = segment_sizes[0];
    for (int i = 1; i < segment_ends.size(); i++)
        segment_ends[i] = segment_sizes[i] + segment_ends[i - 1];
    auto* WSdata = ws()->template caches<T, Context>({ segment_sizes[0] })[0];
    auto* dXdata = tensor->template mutable_data<T, Context>();
    int recv_from = (comm_rank - 1 + comm_size) % comm_size;
    int send_to = (comm_rank + 1) % comm_size;

    // Scatter-Reduce
    for (int i = 0; i < comm_size - 1; i++) {
        int recv_chunk = (comm_rank - i - 1 + comm_size) % comm_size;
        int send_chunk = (comm_rank - i + comm_size) % comm_size;
        auto* segment_send = &(dXdata[
            segment_ends[send_chunk] - segment_sizes[send_chunk]]);
        MPI_Irecv(WSdata, segment_sizes[recv_chunk],
            dtype, recv_from, 0, comm, &recv_req);
        MPI_Send(segment_send, segment_sizes[send_chunk],
            dtype, send_to, 0, comm);
        auto* segment_update = &(dXdata[
            segment_ends[recv_chunk] - segment_sizes[recv_chunk]]);
        MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
        math::Axpy(segment_sizes[recv_chunk],
            1.f, WSdata, segment_update, ctx());
        ctx()->FinishDeviceCompution();
    }

    // Allgather
    for (int i = 0; i < comm_size - 1; i++) {
        int send_chunk = (comm_rank - i + 1 + comm_size) % comm_size;
        int recv_chunk = (comm_rank - i + comm_size) % comm_size;
        auto* segment_send = &(dXdata[
            segment_ends[send_chunk] - segment_sizes[send_chunk]]);
        auto* segment_recv = &(dXdata[
            segment_ends[recv_chunk] - segment_sizes[recv_chunk]]);
        MPI_Sendrecv(segment_send, segment_sizes[send_chunk],
            dtype, send_to, 0, segment_recv, segment_sizes[recv_chunk],
            dtype, recv_from, 0, comm, MPI_STATUS_IGNORE);
    }

    // Normalization
    if (comm_size > 1) {
        math::Scale(count, 1.f / comm_size, dXdata, dXdata, ctx());
    }
}

template <class Context> template <typename T>
void CollectiveUpdateOp<Context>::MPIBcast(
    Tensor*                 tensor,
    MPI_Datatype            dtype) {
    auto* dXdata = tensor->template mutable_data<T, Context>();
    MPI_Bcast(dXdata, tensor->count(), dtype, comm_root, comm);
}

#ifdef WITH_NCCL

template <class Context> template <typename T>
void CollectiveUpdateOp<Context>::NCCLAllReduce(
    Tensor*                 tensor,
    ncclDataType_t          dtype,
    cudaStream_t&           stream) {
    int64_t count = tensor->count();
    auto* dXdata = tensor->template mutable_data<T, Context>();
    NCCL_CHECK(ncclAllReduce((const void*)dXdata, (void*)dXdata,
        count, dtype, ncclSum, nccl_comm, stream));
}

template <class Context> template <typename T>
void CollectiveUpdateOp<Context>::NCCLBcast(
    Tensor*                 tensor,
    ncclDataType_t          dtype,
    cudaStream_t&           stream) {
    int64_t count = tensor->count();
    auto* dXdata = tensor->template mutable_data<T, Context>();
    NCCL_CHECK(ncclBcast((void*)dXdata,
        count, dtype, comm_root, nccl_comm, stream));
}

#endif

template <class Context>
void CollectiveUpdateOp<Context>::RunOnDevice() {
    if (mode == "MPI_ALLREDUCE") {
        for (int i = 0; i < InputSize(); i++) {
            if (XIsType(Input(i), float))
                MPIAllReduce<float>(&Input(i), MPI_FLOAT);
            else if (XIsType(Input(i), float16))
                MPIAllReduce<float16>(&Input(i), MPI_UNSIGNED_SHORT);
            else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
        }
    } else if (mode == "MPI_BCAST") {
        for (int i = 0; i < InputSize(); i++) {
            if (XIsType(Input(i), float))
                MPIBcast<float>(&Input(i), MPI_FLOAT);
            else if (XIsType(Input(i), float16))
                MPIBcast<float16>(&Input(i), MPI_UNSIGNED_SHORT);
            else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
        }
    }
#ifdef WITH_NCCL
    else if (mode == "NCCL_ALLREDUCE") {
        auto stream = ((CUDAContext*)ctx())->cuda_stream();
        for (int i = 0; i < InputSize(); i++) {
            if (XIsType(Input(i), float))
                NCCLAllReduce<float>(&Input(i), ncclFloat, stream);
            else if (XIsType(Input(i), float16))
                NCCLAllReduce<float16>(&Input(i), ncclHalf, stream);
            else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
        }
        for (int i = 0; i < InputSize(); i++) {
            int64_t count = Input(i).count();
            if (XIsType(Input(i), float)) {
                auto* dXdata = Input(i).template mutable_data<float, Context>();
                math::Scale(count, 1.f / comm_size, dXdata, dXdata, ctx());
            }
            else if (XIsType(Input(i), float16)) {
                auto* dXdata = Input(i).template mutable_data<float16, Context>();
                math::Scale(count, 1.f / comm_size, dXdata, dXdata, ctx());
            }
        }
    } else if (mode == "NCCL_BCAST") {
        auto stream = ((CUDAContext*)ctx())->cuda_stream();
        for (int i = 0; i < InputSize(); i++) {
            if (XIsType(Input(i), float))
                NCCLBcast<float>(&Input(i), ncclFloat, stream);
            else if (XIsType(Input(i), float16))
                NCCLBcast<float16>(&Input(i), ncclHalf, stream);
            else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
        }
    }
#endif
    else LOG(FATAL) << "Unsupported collective mode: " << mode;
}

DEPLOY_CPU(CollectiveUpdate);
#ifdef WITH_CUDA
DEPLOY_CUDA(CollectiveUpdate);
#endif
OPERATOR_SCHEMA(CollectiveUpdate).IgnoreVerify();

#endif  // WITH_MPI

}  // namespace dragon