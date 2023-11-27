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
void CollectiveOp<Context>::DoRunWithType() {
  if (src_tensor_ == nullptr) { // Transformation.
    if (operation_ == "ALLREDUCE" && reduction_ == "MEAN") {
      auto* data = dest_tensor_->template mutable_data<T, Context>();
      math::Scale(
          dest_tensor_->count(),
          1.f / float(coll_impl_.comm_size()),
          data,
          data,
          ctx());
    }
    return;
  }
  CopyTensors<T>(/* done = */ false); // Concat tensors.
  if (operation_ == "ALLREDUCE") {
    coll_impl_.AllReduce(
        src_tensor_->template data<T, Context>(),
        src_tensor_->template mutable_data<T, Context>(),
        src_tensor_->count(),
        ctx());
  } else if (operation_ == "ALLGATHER") {
    auto dest_dims = src_tensor_->dims();
    dest_dims.insert(dest_dims.begin(), coll_impl_.comm_size());
    coll_impl_.AllGather(
        src_tensor_->template data<T, Context>(),
        dest_tensor_->Reshape(dest_dims)->template mutable_data<T, Context>(),
        src_tensor_->count(),
        ctx());
  } else if (operation_ == "REDUCESCATTER") {
    auto dest_dims = src_tensor_->dims();
    dest_dims[0] /= coll_impl_.comm_size();
    coll_impl_.ReduceScatter(
        src_tensor_->template data<T, Context>(),
        dest_tensor_->Reshape(dest_dims)->template mutable_data<T, Context>(),
        src_tensor_->count() / coll_impl_.comm_size(),
        ctx());
  } else if (operation_ == "BROADCAST") {
    coll_impl_.Bcast(
        src_tensor_->template mutable_data<T, Context>(),
        src_tensor_->count(),
        ctx());
  } else {
    LOG(FATAL) << "Unsupported operation: " << operation_;
  }
  CopyTensors<T>(/* done = */ true); // Split tensors.
}

template <class Context>
void CollectiveOp<Context>::RunOnDevice() {
  if (coll_impl_.comm_size() <= 1) return;
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
