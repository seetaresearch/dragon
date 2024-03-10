#include "dragon/operators/distributed/collective_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CollectiveOp<Context>::CopyBuffer(bool cat) { // clang-format off
  if (bucket_size_ == 0) { src_index_ += cat ? 0 : 1; return; }
  int64_t bucket_count = 0, input_count = 0;
  for (; input_count + src_index_ < InputSize(); ++input_count) {
    bucket_count += Input(src_index_ + input_count).count();
    if (bucket_count * sizeof(T) > bucket_size_) break;
  } 
  if (input_count == 1) { if (!cat) src_index_ += 1; return; }
  auto* B = ctx()->workspace()->CreateTensor("BufferShared");
  if (cat) { // clang-format on
    src_tensor_ = dest_tensor_ = B;
    auto* buf = B->Reshape({bucket_count})->template mutable_data<T, Context>();
    for (int i = 0; i < input_count; ++i) {
      auto& X = Input(src_index_ + i);
      math::Copy(X.count(), X.template data<T, Context>(), buf, ctx());
      buf += X.count();
    }
    return;
  }
  auto* buf = B->template data<T, Context>();
  for (int i = 0; i < input_count; ++i) {
    auto* Y = Output(src_index_ + i);
    math::Copy(Y->count(), buf, Y->template mutable_data<T, Context>(), ctx());
    buf += Y->count();
  }
  src_index_ += input_count;
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
  CopyBuffer<T>(/* cat = */ true); // Concat tensors.
  if (operation_ == "ALLREDUCE") {
    coll_impl_.AllReduce(
        src_tensor_->template data<T, Context>(),
        dest_tensor_->template mutable_data<T, Context>(),
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
    coll_impl_.Broadcast(
        src_tensor_->template data<T, Context>(),
        dest_tensor_->template mutable_data<T, Context>(),
        src_tensor_->count(),
        ctx());
  } else {
    LOG(FATAL) << "Unsupported operation: " << operation_;
  }
  CopyBuffer<T>(/* done = */ false); // Split tensors.
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
