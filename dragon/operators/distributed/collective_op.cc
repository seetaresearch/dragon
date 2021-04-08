#ifdef USE_MPI

#include "dragon/operators/distributed/collective_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CollectiveOp<Context>::AllReduceMPI() {
  int64_t count = src_tensor_->count();
  int64_t seg_size = count / comm_size_;
  int64_t residual = count % comm_size_;

  vec64_t sizes(comm_size_, seg_size);
  for (int i = 0; i < residual; ++i) {
    sizes[i]++;
  }

  vec64_t ends(comm_size_, sizes[0]);
  for (int i = 1; i < ends.size(); ++i) {
    ends[i] = sizes[i] + ends[i - 1];
  }

  auto to = (comm_rank_ + 1) % comm_size_;
  auto from = (comm_rank_ - 1 + comm_size_) % comm_size_;

  auto* data = src_tensor_->template mutable_data<T, Context>();
  auto* scratch = ctx()->workspace()->template data<T, Context>({sizes[0]})[0];

  // Scatter-Reduce
  MPI_Request recv_req;
  for (int i = 0; i < comm_size_ - 1; i++) {
    auto base_idx = comm_rank_ - i + comm_size_;
    auto recv_idx = (base_idx - 1) % comm_size_;
    auto send_idx = base_idx % comm_size_;
    auto* send_buf = &(data[ends[send_idx] - sizes[send_idx]]);
    auto* recv_buf = &(data[ends[recv_idx] - sizes[recv_idx]]);
    IRecv(scratch, sizes[recv_idx], from, &recv_req);
    Send(send_buf, sizes[send_idx], to);
    MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
    math::Axpy(sizes[recv_idx], 1.f, scratch, recv_buf, ctx());
    // Wait stream to finish the local reduce before next sending
    ctx()->FinishDeviceComputation();
  }

  // Allgather
  for (int i = 0; i < comm_size_ - 1; i++) {
    auto base_idx = comm_rank_ - i + comm_size_;
    auto send_idx = (base_idx + 1) % comm_size_;
    auto recv_idx = base_idx % comm_size_;
    auto* send_buf = &(data[ends[send_idx] - sizes[send_idx]]);
    auto* recv_buf = &(data[ends[recv_idx] - sizes[recv_idx]]);
    SendRecv(send_buf, sizes[send_idx], to, recv_buf, sizes[recv_idx], from);
  }
}

template <class Context>
template <typename T>
void CollectiveOp<Context>::AllReduceNCCL() {
#ifdef USE_NCCL
  auto* data = src_tensor_->template mutable_data<T, Context>();
  NCCL_CHECK(ncclAllReduce(
      (const void*)data,
      (void*)data,
      src_tensor_->count(),
      this->template nccl_dtype<T>(),
      ncclSum,
      this->nccl_comm(),
      ((CUDAContext*)ctx())->cuda_stream()));
#endif // USE_NCCL
}

template <class Context>
template <typename T>
void CollectiveOp<Context>::BroadcastMPI() {
  auto* data = src_tensor_->template mutable_data<T, Context>();
  Broadcast(data, src_tensor_->count());
}

template <class Context>
template <typename T>
void CollectiveOp<Context>::BroadcastNCCL() {
#ifdef USE_NCCL
  NCCL_CHECK(ncclBcast(
      (void*)src_tensor_->template mutable_data<T, Context>(),
      src_tensor_->count(),
      this->template nccl_dtype<T>(),
      comm_root_,
      this->nccl_comm(),
      ((CUDAContext*)ctx())->cuda_stream()));
#endif // USE_NCCL
}

template <class Context>
template <typename T>
void CollectiveOp<Context>::DoRunWithType() {
  if (src_tensor_ != nullptr) {
    // Dispatch collective communication
    if (communication_ == "ALLREDUCE") {
      if (enable_nccl_) {
        AllReduceNCCL<T>();
      } else {
        AllReduceMPI<T>();
      }
    } else if (communication_ == "BROADCAST") {
      if (enable_nccl_) {
        BroadcastNCCL<T>();
      } else {
        BroadcastMPI<T>();
      }
    } else {
      LOG(FATAL) << "Unknown communication: " << communication_;
    }
  } else {
    // Dispatch other computation
    if (communication_ == "ALLREDUCE" && operation_ == "MEAN") {
      auto* data = dest_tensor_->template mutable_data<T, Context>();
      math::Scale(dest_tensor_->count(), 1.f / comm_size_, data, data, ctx());
    }
  }
}

template <class Context>
void CollectiveOp<Context>::RunOnDevice() {
  if (comm_size_ <= 1) return;
  // Wait stream to finish the enqueued kernels.
  // Otherwise, data corruption will happen through UVA
  // during executing collectives asynchronously.
  ctx()->FinishDeviceComputation();
  for (int i = 0; i < InputSize(); i++) {
    src_tensor_ = &Input(i);
    DispatchHelper<dtypes::Numerical>::Call(this, *src_tensor_);
  }
  src_tensor_ = nullptr;
  for (int i = 0; i < InputSize(); i++) {
    dest_tensor_ = &Input(i);
    DispatchHelper<dtypes::Numerical>::Call(this, *dest_tensor_);
  }
}

DEPLOY_CPU_OPERATOR(Collective);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Collective);
#endif

OPERATOR_SCHEMA(Collective).AllowInplace([](int, int) -> bool { return true; });

} // namespace dragon

#endif // USE_MPI
