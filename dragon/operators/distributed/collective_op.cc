#ifdef USE_MPI

#include "dragon/operators/distributed/collective_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CollectiveOp<Context>::AllReduceMPI(Tensor* tensor) {
  MPI_Request recv_req;

  int64_t count = tensor->count();
  int64_t seg_size = count / comm_size_;
  int64_t residual = count % comm_size_;
  vec64_t sizes(comm_size_, seg_size);
  for (int i = 0; i < residual; i++)
    sizes[i]++;

  vec64_t ends(comm_size_);
  ends[0] = sizes[0];
  for (int i = 1; i < ends.size(); i++)
    ends[i] = sizes[i] + ends[i - 1];

  auto to = (comm_rank_ + 1) % comm_size_;
  auto from = (comm_rank_ - 1 + comm_size_) % comm_size_;

  auto* x = tensor->template mutable_data<T, Context>();
  auto* scratch = ws()->template data<T, Context>({sizes[0]})[0];

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
    math::Axpy(sizes[recv_i], 1.f, scratch, update, ctx());
    ctx()->FinishDeviceComputation();
  }

  // Allgather
  for (int i = 0; i < comm_size_ - 1; i++) {
    auto base_id = comm_rank_ - i + comm_size_;
    auto send_i = (base_id + 1) % comm_size_;
    auto recv_i = base_id % comm_size_;
    auto* send = &(x[ends[send_i] - sizes[send_i]]);
    auto* recv = &(x[ends[recv_i] - sizes[recv_i]]);
    SendRecv(send, sizes[send_i], to, recv, sizes[recv_i], from);
  }

  // Normalization
  if (comm_size_ > 1 && operation_ == "MEAN") {
    math::Scale(count, 1.f / comm_size_, x, x, ctx());
  }
}

template <class Context>
template <typename T>
void CollectiveOp<Context>::AllReduceNCCL(Tensor* tensor) {
#ifdef USE_NCCL
  auto* x = tensor->template mutable_data<T, Context>();
  NCCL_CHECK(ncclAllReduce(
      (const void*)x,
      (void*)x,
      tensor->count(),
      this->template nccl_dtype<T>(),
      ncclSum,
      this->nccl_comm(),
      ((CUDAContext*)ctx())->cuda_stream()));
  if (comm_size_ > 1 && operation_ == "MEAN") {
    math::Scale(tensor->count(), 1.f / comm_size_, x, x, ctx());
  }
#endif // USE_NCCL
}

template <class Context>
template <typename T>
void CollectiveOp<Context>::AllReduceDispatcher(Tensor* tensor) {
  if (enable_nccl_) {
    AllReduceNCCL<T>(tensor);
  } else {
    AllReduceMPI<T>(tensor);
  }
}

template <class Context>
template <typename T>
void CollectiveOp<Context>::BroadcastMPI(Tensor* tensor) {
  auto* x = tensor->template mutable_data<T, Context>();
  Broadcast(x, tensor->count());
}

template <class Context>
template <typename T>
void CollectiveOp<Context>::BroadcastNCCL(Tensor* tensor) {
#ifdef USE_NCCL
  auto* x = tensor->template mutable_data<T, Context>();
  NCCL_CHECK(ncclBcast(
      (void*)x,
      tensor->count(),
      this->template nccl_dtype<T>(),
      comm_root_,
      this->nccl_comm(),
      ((CUDAContext*)ctx())->cuda_stream()));
#endif // USE_NCCL
}

template <class Context>
template <typename T>
void CollectiveOp<Context>::BroadcastDispatcher(Tensor* tensor) {
  if (enable_nccl_) {
    BroadcastNCCL<T>(tensor);
  } else {
    BroadcastMPI<T>(tensor);
  }
}

template <class Context>
void CollectiveOp<Context>::RunOnDevice() {
  if (communication_ == "ALLREDUCE") {
    for (int i = 0; i < InputSize(); i++) {
      auto& X = Input(i);
      if (XIsType(X, int8_t)) {
        AllReduceDispatcher<int8_t>(&Input(i));
      } else if (XIsType(X, uint8_t)) {
        AllReduceDispatcher<uint8_t>(&Input(i));
      } else if (XIsType(X, int)) {
        AllReduceDispatcher<int>(&Input(i));
      } else if (XIsType(X, int64_t)) {
        AllReduceDispatcher<int64_t>(&Input(i));
      } else if (XIsType(X, float16)) {
        AllReduceDispatcher<float16>(&Input(i));
      } else if (XIsType(X, float)) {
        AllReduceDispatcher<float>(&Input(i));
      } else if (XIsType(X, double)) {
        AllReduceDispatcher<double>(&Input(i));
      } else {
        LOG(FATAL) << MessageForUnsupported(
            types::to_string(X.meta()),
            {"int8",
             "uint8",
             "int32",
             "int64",
             "float16",
             "float32",
             "float64"});
      }
    }
  } else if (communication_ == "BROADCAST") {
    for (int i = 0; i < InputSize(); i++) {
      auto& X = Input(i);
      if (XIsType(X, bool)) {
        BroadcastDispatcher<bool>(&Input(i));
      } else if (XIsType(X, int8_t)) {
        BroadcastDispatcher<int8_t>(&Input(i));
      } else if (XIsType(X, uint8_t)) {
        BroadcastDispatcher<uint8_t>(&Input(i));
      } else if (XIsType(X, int)) {
        BroadcastDispatcher<int>(&Input(i));
      } else if (XIsType(X, int64_t)) {
        BroadcastDispatcher<int64_t>(&Input(i));
      } else if (XIsType(X, float16)) {
        BroadcastDispatcher<float16>(&Input(i));
      } else if (XIsType(X, float)) {
        BroadcastDispatcher<float>(&Input(i));
      } else if (XIsType(X, double)) {
        BroadcastDispatcher<double>(&Input(i));
      } else {
        LOG(FATAL) << MessageForUnsupported(
            types::to_string(X.meta()),
            {"bool",
             "int8",
             "uint8",
             "int32",
             "int64",
             "float16",
             "float32",
             "float64"});
      }
    }
  } else {
    LOG(FATAL) << "Unknown communication: " << communication_;
  }
}

DEPLOY_CPU(Collective);
#ifdef USE_CUDA
DEPLOY_CUDA(Collective);
#endif

OPERATOR_SCHEMA(Collective).AllowInplace([](int, int) -> bool { return true; });

} // namespace dragon

#endif // USE_MPI
