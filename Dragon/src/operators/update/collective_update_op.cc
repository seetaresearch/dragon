#include "core/workspace.h"
#include "utils/math_functions.h"
#include "operators/update/collective_update_op.h"

namespace dragon {

#ifdef WITH_MPI

template <class Context>
void CollectiveUpdateOp<Context>::InitNCCL() {
#ifdef WITH_NCCL
    ncclUniqueId id;
    if (comm_rank_ == comm_root_)
        NCCL_CHECK(ncclGetUniqueId(&id));
    BCast((uint8_t*)&id, sizeof(id));
    ctx()->SwitchToDevice();
    NCCL_CHECK(ncclCommInitRank(
        &nccl_comm,
        comm_size_,
        id,
        comm_rank_
    ));
#else
    LOG(FATAL) << "NCCL was not compiled.";
#endif
}

template <class Context> template <typename T>
void CollectiveUpdateOp<Context>::MPIAllReduce(
    Tensor*                 tensor) {
    MPI_Request recv_req;

    int64_t count = tensor->count();
    int64_t seg_size = count / comm_size_;
    int64_t residual = count % comm_size_;
    vec64_t sizes(comm_size_, seg_size);
    for (int i = 0; i < residual; i++) sizes[i]++;

    vec64_t ends(comm_size_);
    ends[0] = sizes[0];
    for (int i = 1; i < ends.size(); i++)
        ends[i] = sizes[i] + ends[i - 1];

    auto to = (comm_rank_ + 1) % comm_size_;
    auto from = (comm_rank_ - 1 + comm_size_) % comm_size_;

    auto* scratch = ws()
        ->template data<T, Context>
            ({ sizes[0] })[0];

    auto* x = tensor->template mutable_data<T, Context>();

    // Scatter-Reduce
    for (int i = 0; i < comm_size_ - 1; i++) {
        auto base_id = comm_rank_ - i + comm_size_;
        auto recv_i = (base_id - 1) % comm_size_;
        auto send_i = base_id % comm_size_;
        auto* send = &(x[ends[send_i] - sizes[send_i]]);
        auto* update = &(x[ends[recv_i] - sizes[recv_i]]);
        IRecv(scratch, sizes[recv_i], from, &recv_req);
        Send(send, sizes[send_i], to);
        MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
        math::Axpy(
            sizes[recv_i],
            1.f, scratch,
            update, ctx()
        ); ctx()->FinishDeviceCompution();
    }

    // Allgather
    for (int i = 0; i < comm_size_ - 1; i++) {
        auto base_id = comm_rank_ - i + comm_size_;
        auto send_i = (base_id + 1) % comm_size_;
        auto recv_i = base_id % comm_size_;
        auto* send = &(x[ends[send_i] - sizes[send_i]]);
        auto* recv = &(x[ends[recv_i] - sizes[recv_i]]);
        SendRecv(
            send, sizes[send_i], to,
            recv, sizes[recv_i], from
        );
    }

    // Normalization
    if (comm_size_ > 1) {
        math::Scale(
            count,
            1.f / comm_size_,
            x, x, ctx()
        );
    }
}

template <class Context> template <typename T>
void CollectiveUpdateOp<Context>::MPIBCast(
    Tensor*                 tensor) {
    auto* x = tensor->template
        mutable_data<T, Context>();
    BCast(x, tensor->count());
}

#ifdef WITH_NCCL

template <class Context> template <typename T>
void CollectiveUpdateOp<Context>::NCCLAllReduce(
    Tensor*                 tensor,
    ncclDataType_t          dtype) {
    auto* x = tensor->template
        mutable_data<T, Context>();
    NCCL_CHECK(ncclAllReduce(
        (const void*)x, (void*)x,
        tensor->count(), dtype,
        ncclSum, nccl_comm,
        ((CUDAContext*)ctx())->cuda_stream()
    ));
    // Normalization
    if (comm_size_ > 1) {
        math::Scale(
            tensor->count(),
            1.f / comm_size_,
            x, x, ctx()
        );
    }
}

template <class Context> template <typename T>
void CollectiveUpdateOp<Context>::NCCLBcast(
    Tensor*                 tensor,
    ncclDataType_t          dtype) {
    auto* x = tensor->template
        mutable_data<T, Context>();
    NCCL_CHECK(ncclBcast(
        (void*)x,
        tensor->count(), dtype,
        comm_root_, nccl_comm,
        ((CUDAContext*)ctx())->cuda_stream()
    ));
}

#endif

template <class Context>
void CollectiveUpdateOp<Context>::RunOnDevice() {
    if (mode_ == "MPI_ALLREDUCE") {
        for (int i = 0; i < XSize(); i++) {
            if (XIsType(X(i), float)) {
                MPIAllReduce<float>(&X(i));
            } else if (XIsType(X(i), float16)) {
                MPIAllReduce<float16>(&X(i));
            } else {
                LOG(FATAL) << DTypeString(X(i),
                    { "float32", "float16" }
                );
            }
        }
    } else if (mode_ == "MPI_BCAST") {
        for (int i = 0; i < XSize(); i++) {
            if (XIsType(X(i), float)) {
                MPIBCast<float>(&X(i));
            } else if (XIsType(X(i), float16)) {
                MPIBCast<float16>(&X(i));
            } else {
                LOG(FATAL) << DTypeString(X(i),
                    { "float32", "float16" }
                );
            }
        }
    }
#ifdef WITH_NCCL
    else if (mode_ == "NCCL_ALLREDUCE") {
        for (int i = 0; i < XSize(); i++) {
            if (XIsType(X(i), float)) {
                NCCLAllReduce<float>(&X(i), ncclFloat);
            } else if (XIsType(X(i), float16)) {
                NCCLAllReduce<float16>(&X(i), ncclHalf);
            } else {
                LOG(FATAL) << DTypeString(X(i),
                    { "float32", "float16" }
                );
            }
        }
    } else if (mode_ == "NCCL_BCAST") {
        for (int i = 0; i < XSize(); i++) {
            if (XIsType(X(i), float)) {
                NCCLBcast<float>(&X(i), ncclFloat);
            } else if (XIsType(X(i), float16)) {
                NCCLBcast<float16>(&X(i), ncclHalf);
            } else {
                LOG(FATAL) << DTypeString(X(0),
                    { "float32", "float16" }
                );
            }
        }
    }
#endif
    else {
        LOG(FATAL) << "Unknown Mode: " << mode_;
    }
}

DEPLOY_CPU(CollectiveUpdate);
#ifdef WITH_CUDA
DEPLOY_CUDA(CollectiveUpdate);
#endif

OPERATOR_SCHEMA(CollectiveUpdate).IgnoreVerify();

#endif  // WITH_MPI

}  // namespace dragon