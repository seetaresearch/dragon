#ifdef USE_MPI

#include "dragon/operators/distributed/collective_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CollectiveOp<Context>::CopyTensors(bool done) {
  int64_t count = 0, num = 0;
  for (int i = 0; i + src_index_ < InputSize(); ++i) {
    num += 1;
    count += Input(src_index_ + i).count();
    if (count * sizeof(T) > buffer_size_) break;
  }
  if (num == 1) {
    if (done) src_index_ += 1;
    return;
  }
  auto* Y = Output("Y");
  if (!done) {
    src_tensor_ = Y;
    auto* data = Y->Reshape({count})->template mutable_data<T, Context>();
    for (int i = 0; i < num; ++i) {
      auto& X = Input(src_index_ + i);
      math::Copy(X.count(), X.template data<T, Context>(), data, ctx());
      data += X.count();
    }
  } else {
    auto* data = Y->template data<T, Context>();
    for (int i = 0; i < num; ++i) {
      auto& X = Input(src_index_ + i);
      math::Copy(X.count(), data, X.template mutable_data<T, Context>(), ctx());
      data += X.count();
    }
    src_index_ += num;
  }
}

template <class Context>
template <typename T>
void CollectiveOp<Context>::AllReduceMPI() {
  const auto N = src_tensor_->count();
  const auto split_size = N / comm_size_;
  const auto residual_size = N % comm_size_;

  vec64_t sizes(comm_size_, split_size);
  for (int i = 0; i < residual_size; ++i) {
    sizes[i]++;
  }

  vec64_t ends(comm_size_, sizes[0]);
  for (int i = 1; i < ends.size(); ++i) {
    ends[i] = sizes[i] + ends[i - 1];
  }

  auto to = (comm_rank_ + 1) % comm_size_;
  auto from = (comm_rank_ - 1 + comm_size_) % comm_size_;

  auto* data = src_tensor_->template mutable_data<T, Context>();
  auto* scratch = ctx()->workspace()->template data<T, Context>(sizes[0]);

  // Scatter-Reduce.
  MPI_Request recv_req;
  for (int i = 0; i < comm_size_ - 1; ++i) {
    auto idx = comm_rank_ - i + comm_size_;
    auto send_idx = idx % comm_size_;
    auto recv_idx = (idx - 1) % comm_size_;
    auto* send_buf = &(data[ends[send_idx] - sizes[send_idx]]);
    auto* recv_buf = &(data[ends[recv_idx] - sizes[recv_idx]]);
    IRecv(scratch, sizes[recv_idx], from, &recv_req);
    Send(send_buf, sizes[send_idx], to);
    MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
    math::Axpy(sizes[recv_idx], 1.f, scratch, recv_buf, ctx());
    // Wait stream to finish the local reduce before next sending.
    ctx()->FinishDeviceComputation();
  }

  // Allgather.
  for (int i = 0; i < comm_size_ - 1; ++i) {
    auto idx = comm_rank_ - i + comm_size_;
    auto recv_idx = idx % comm_size_;
    auto send_idx = (idx + 1) % comm_size_;
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
      this->template nccl_data_type<T>(),
      ncclSum,
      this->nccl_comm(),
      ((CUDAContext*)ctx())->cuda_stream()));
#endif
}

template <class Context>
template <typename T>
void CollectiveOp<Context>::AllReduceCNCL() {
#ifdef USE_MLU
  auto* data = src_tensor_->template mutable_data<T, Context>();
  CNCL_CHECK(cnclAllReduce(
      (const void*)data,
      (void*)data,
      src_tensor_->count(),
      this->template cncl_data_type<T>(),
      cnclSum,
      this->cncl_comm(),
      ((MLUContext*)ctx())->mlu_stream()));
#endif
}

template <class Context>
template <typename T>
void CollectiveOp<Context>::AllGatherMPI() {
  auto dest_dims = src_tensor_->dims();
  dest_dims[0] *= comm_size_;
  AllGather(
      src_tensor_->template data<T, Context>(),
      dest_tensor_->Reshape(dest_dims)->template mutable_data<T, Context>(),
      src_tensor_->count());
}

template <class Context>
template <typename T>
void CollectiveOp<Context>::AllGatherNCCL() {
#ifdef USE_NCCL
  auto dest_dims = src_tensor_->dims();
  dest_dims[0] *= comm_size_;
  NCCL_CHECK(ncclAllGather(
      src_tensor_->template data<T, Context>(),
      dest_tensor_->Reshape(dest_dims)->template mutable_data<T, Context>(),
      src_tensor_->count(),
      this->template nccl_data_type<T>(),
      this->nccl_comm(),
      ((CUDAContext*)ctx())->cuda_stream()));
#endif
}

template <class Context>
template <typename T>
void CollectiveOp<Context>::AllGatherCNCL() {
#ifdef USE_MLU
  auto dest_dims = src_tensor_->dims();
  dest_dims[0] *= comm_size_;
  CNCL_CHECK(cnclAllGather(
      src_tensor_->template data<T, Context>(),
      dest_tensor_->Reshape(dest_dims)->template mutable_data<T, Context>(),
      src_tensor_->count(),
      this->template cncl_data_type<T>(),
      this->cncl_comm(),
      ((MLUContext*)ctx())->mlu_stream()));
#endif
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
      this->template nccl_data_type<T>(),
      comm_root_,
      this->nccl_comm(),
      ((CUDAContext*)ctx())->cuda_stream()));
#endif
}

template <class Context>
template <typename T>
void CollectiveOp<Context>::BroadcastCNCL() {
#ifdef USE_MLU
  CNCL_CHECK(cnclBcast(
      (void*)src_tensor_->template mutable_data<T, Context>(),
      src_tensor_->count(),
      this->template cncl_data_type<T>(),
      comm_root_,
      this->cncl_comm(),
      ((MLUContext*)ctx())->mlu_stream()));
#endif
}

template <class Context>
template <typename T>
void CollectiveOp<Context>::DoRunWithType() {
  if (src_tensor_ == nullptr) { // Transformation.
    if (operation_ == "ALLREDUCE" && reduction_ == "MEAN") {
      auto* data = dest_tensor_->template mutable_data<T, Context>();
      math::Scale(dest_tensor_->count(), 1.f / comm_size_, data, data, ctx());
    }
    return;
  }
  CopyTensors<T>(/* done = */ false); // Concat tensors.
  if (operation_ == "ALLREDUCE") {
    if (enable_nccl_) {
      AllReduceNCCL<T>();
    } else if (enable_cncl_) {
      AllReduceCNCL<T>();
    } else {
      AllReduceMPI<T>();
    }
  } else if (operation_ == "ALLGATHER") {
    if (enable_nccl_) {
      AllGatherNCCL<T>();
    } else if (enable_cncl_) {
      AllGatherCNCL<T>();
    } else {
      AllGatherMPI<T>();
    }
  } else if (operation_ == "BROADCAST") {
    if (enable_nccl_) {
      BroadcastNCCL<T>();
    } else if (enable_cncl_) {
      BroadcastCNCL<T>();
    } else {
      BroadcastMPI<T>();
    }
  } else {
    LOG(FATAL) << "Unsupported operation: " << operation_;
  }
  CopyTensors<T>(/* done = */ true); // Split tensors.
}

template <class Context>
void CollectiveOp<Context>::RunOnDevice() {
  if (comm_size_ <= 1) return;
  // Wait stream to finish the enqueued kernels.
  ctx()->FinishDeviceComputation();
  // Enqueue collective kernels.
  for (src_index_ = 0; src_index_ < InputSize();) {
    src_tensor_ = &Input(src_index_), dest_tensor_ = Output(src_index_);
    DispatchHelper<dtypes::Numerical>::Call(this, *src_tensor_);
  }
  // Enqueue transform kernels.
  src_tensor_ = nullptr;
  for (src_index_ = 0; src_index_ < InputSize(); ++src_index_) {
    dest_tensor_ = &Input(src_index_);
    DispatchHelper<dtypes::Numerical>::Call(this, *dest_tensor_);
  }
}

DEPLOY_CPU_OPERATOR(Collective);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Collective);
#endif
#ifdef USE_MLU
DEPLOY_MLU_OPERATOR(Collective);
#endif

OPERATOR_SCHEMA(Collective).AllowInplace([](int, int) -> bool { return true; });

} // namespace dragon

#endif // USE_MPI
