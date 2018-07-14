#include "core/workspace.h"
#include "utils/math_functions.h"
#include "operators/update/collective_update_op.h"

namespace dragon {

#ifdef WITH_MPI

template <class Context>
void CollectiveUpdateOp<Context>::InitMPI() {
    comm = (MPI_Comm)OperatorBase::Arg<int64_t>("comm", 0);
    group = (MPI_Group)OperatorBase::Arg<int64_t>("group", 0);
    int world_root = OperatorBase::Arg<int>("root", 0);
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
#ifdef WITH_MPI_NCCL
    ncclUniqueId id;
    if (comm_rank == comm_root) NCCL_CHECK(ncclGetUniqueId(&id));
    MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, comm_root, comm);
    NCCL_CHECK(ncclCommInitRank(&nccl_comm, comm_size, id, comm_rank));
    closure = CUDAClosure<Context>(&ctx());
#else
    LOG(FATAL) << "NCCL was not compiled.";
#endif
}

template <class Context>
void CollectiveUpdateOp<Context>::MPIAllReduceWithFloat() {
    for (int j = 0; j < InputSize(); j++) {
        TIndex count = Input(j).count();
        MPI_Request recv_req;
        TIndex segment_size = count / comm_size;
        TIndex residual = count % comm_size;
        vector<TIndex> segment_sizes(comm_size, segment_size);
        for (int i = 0; i < residual; i++) segment_sizes[i]++;
        vector<TIndex> segment_ends(comm_size);
        segment_ends[0] = segment_sizes[0];
        for (int i = 1; i < segment_ends.size(); i++) 
            segment_ends[i] = segment_sizes[i] + segment_ends[i - 1];
#ifdef WITH_MPI_CUDA
        auto* WSdata = ws()->template caches<float, Context>({ segment_sizes[0] })[0];
        auto* dXdata = Input(j).template mutable_data<float, Context>();
#else
        auto* WSdata = ws()->template caches<float, CPUContext>({ segment_sizes[0] })[0];
        auto* dXdata = Input(j).template mutable_data<float, CPUContext>();
#endif // WITH_MPI_CUDA
        int recv_from = (comm_rank - 1 + comm_size) % comm_size;
        int send_to = (comm_rank + 1) % comm_size;

        //  scatter-reduce
        for (int i = 0; i < comm_size - 1; i++) {
            int recv_chunk = (comm_rank - i - 1 + comm_size) % comm_size;
            int send_chunk = (comm_rank - i + comm_size) % comm_size;
            auto* segment_send = &(dXdata[
                segment_ends[send_chunk] - segment_sizes[send_chunk]
            ]);
            MPI_Irecv(WSdata, segment_sizes[recv_chunk],
                MPI_FLOAT, recv_from, 0, comm, &recv_req);
            MPI_Send(segment_send, segment_sizes[send_chunk],
                MPI_FLOAT, send_to, 0, comm);
            auto* segment_update = &(dXdata[
                segment_ends[recv_chunk] - segment_sizes[recv_chunk]
            ]);
            MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
#ifdef WITH_MPI_CUDA
            math::Axpy<float, Context>(segment_sizes[recv_chunk],
                1.0, WSdata, segment_update, &ctx());
            ctx().FinishDeviceCompution();
#else 
            math::Axpy<float, CPUContext>(segment_sizes[recv_chunk],
                1.0, WSdata, segment_update, &ctx());
#endif // WITH_MPI_CUDA
        }

        //  allgather
        for (int i = 0; i < comm_size - 1; i++) {
            int send_chunk = (comm_rank - i + 1 + comm_size) % comm_size;
            int recv_chunk = (comm_rank - i + comm_size) % comm_size;
            auto* segment_send = &(dXdata[
                segment_ends[send_chunk] - segment_sizes[send_chunk]
            ]);
            auto* segment_recv = &(dXdata[
                segment_ends[recv_chunk] - segment_sizes[recv_chunk]
            ]);
            MPI_Sendrecv(segment_send, segment_sizes[send_chunk],
                MPI_FLOAT, send_to, 0,
                    segment_recv, segment_sizes[recv_chunk],
                        MPI_FLOAT, recv_from, 0,
                            comm, MPI_STATUS_IGNORE);
        }

        //  normalization
        if (comm_size > 1) {
#ifdef WITH_MPI_CUDA
            math::Scal<float, Context>(count,
                1.f / comm_size, dXdata, &ctx());
#else
            math::Scal<float, CPUContext>(count,
                1.f / comm_size, dXdata, &ctx());
#endif  // WITH_MPI_CUDA
        }
    }
}

template <class Context>
void CollectiveUpdateOp<Context>::NCCLAllReduceWithFloat() {
#ifdef WITH_MPI_NCCL
    auto stream = closure.cuda_stream(0);
    for (int i = 0; i < InputSize(); i++) {
        TIndex count = Input(i).count();
        auto* dXdata = Input(i).template mutable_data<float, Context>();
        NCCL_CHECK(ncclAllReduce((const void*)dXdata, (void*)dXdata,
            count, ncclFloat, ncclSum, nccl_comm, stream));
    }
    closure.Sync();
    for (int i = 0; i < InputSize(); i++) {
        TIndex count = Input(i).count();
        auto* dXdata = Input(i).template mutable_data<float, Context>();
        math::Scal<float, Context>(count, 1.f / comm_size, dXdata, &ctx());
    }
#endif
}

template <class Context>
void CollectiveUpdateOp<Context>::MPIBcastWithFloat() {
    for (int i = 0; i < InputSize(); i++) {
        TIndex count = Input(i).count();
#ifdef WITH_MPI_CUDA
        auto* dXdata = Input(i).template mutable_data<float, Context>();
#else
        auto* dXdata = Input(i).template mutable_data<float, CPUContext>();
#endif
        MPI_Bcast(dXdata, count, MPI_FLOAT, comm_root, comm);
    }
}

template <class Context>
void CollectiveUpdateOp<Context>::NCCLBcastWithFloat() {
#ifdef WITH_MPI_NCCL
    auto stream = closure.cuda_stream(0);
    for (int i = 0; i < InputSize(); i++) {
        TIndex count = Input(i).count();
        auto* dXdata = Input(i).template mutable_data<float, Context>();
        NCCL_CHECK(ncclBcast((void*)dXdata, count,
            ncclFloat, comm_root, nccl_comm, stream));
    }
    closure.Sync();
#endif
}

template <class Context>
void CollectiveUpdateOp<Context>::RunOnDevice() {
    if(XIsType(Input(0), float)) {
        if (mode == "MPI_ALLREDUCE") {
            MPIAllReduceWithFloat();
        } else if (mode == "NCCL_ALLREDUCE") {
            NCCLAllReduceWithFloat();
        } else if (mode == "MPI_BCAST") {
            MPIBcastWithFloat();
        } else if (mode == "NCCL_BCAST") {
            NCCLBcastWithFloat();
        } else LOG(FATAL) << "Unsupported collective mode: " << mode;
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(CollectiveUpdate);
#ifdef WITH_CUDA
DEPLOY_CUDA(CollectiveUpdate);
#endif
OPERATOR_SCHEMA(CollectiveUpdate).IgnoreVerify();

#endif    // WITH_MPI

}    // namespace dragon