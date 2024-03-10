#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/operators/training/zero_update_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
T ZeroUpdateOpBase<Context>::GetHyper(const string& key) {
  auto* X = workspace()->GetTensor(name() + "/" + key);
  return X->template data<T, CPUContext>()[0];
}

template <class Context>
Tensor* ZeroUpdateOpBase<Context>::GetState(const string& key) {
  return workspace()->CreateTensor(name() + "/" + bucket_name_ + "/" + key);
}

template <class Context>
template <typename T>
Tensor* ZeroUpdateOpBase<Context>::InitMaster() {
  auto* Y = GetState("master");
  if (Y->count() > 0) return Y; // clang-format off
  int64_t recv_count = 0, comm_size = coll_impl_.comm_size();
  for (int i = 0; i < InputSize(); ++i) recv_count += Output(i)->count();
  recv_count = (recv_count + comm_size - 1) / comm_size; // clang-format on
  int64_t comm_rank = coll_impl_.comm_rank();
  int64_t send_st = comm_rank * recv_count, flat_st = 0, slice_st = 0;
  int64_t send_ed = send_st + recv_count, flat_ed = 0, slice_num = 0;
  auto* y = Y->Reshape({recv_count})->template mutable_data<float, Context>();
  for (int i = 0; i < InputSize(); ++i, flat_st = flat_ed) {
    auto* X = Output(i);
    flat_ed += X->count();
    slice_num = std::min(flat_ed, send_ed) - std::max(flat_st, send_st);
    slice_num = std::min(slice_num, X->count());
    if (slice_num <= 0) continue;
    slice_st = std::min(send_ed - flat_st, X->count()) - slice_num;
    math::Cast(slice_num, X->template data<T, Context>() + slice_st, y, ctx());
    y += slice_num;
  }
  return Y;
}

template <class Context>
template <typename T>
void ZeroUpdateOpBase<Context>::SetDenseHyper(int64_t N, T* x, T* y) {
  int64_t send_st = coll_impl_.comm_rank() * N, flat_st = 0;
  int64_t send_ed = send_st + N, flat_ed = 0, M = 0;
  for (int i = 0; i < InputSize(); ++i, flat_st = flat_ed) {
    auto* X = Output(i);
    flat_ed += X->count();
    M = std::min(flat_ed, send_ed) - std::max(flat_st, send_st);
    M = std::min(M, X->count());
    if (M <= 0) continue;
    math::Set(M, x[i], y, ctx());
    y += M;
  }
}

template <class Context>
template <typename T>
void ZeroUpdateOpBase<Context>::SetLRS(int64_t N, T* y) {
  int num;
  this->lr_scales(0, &num);
  CHECK_EQ(num, InputSize()) << "\nBad number of <lr_scales>.";
  vector<T> lrs(num); // clang-format off
  for (int i = 0; i < num; ++i) lrs[i] = this->lr_scales(i);
  SetDenseHyper(N, lrs.data(), y); // clang-format on
}

template <class Context>
template <typename T>
void ZeroUpdateOpBase<Context>::CopyBuffer(T* y, bool cat) {
  for (int i = 0; i < InputSize(); ++i) {
    auto* X = cat ? &Input(i) : Output(i);
    const auto N = X->count();
    if (cat) math::Copy(N, X->template data<T, Context>(), y, ctx());
    if (!cat) math::Copy(N, y, X->template mutable_data<T, Context>(), ctx());
    y += N;
  }
}

template <class Context>
template <typename T>
void ZeroUpdateOpBase<Context>::TransformGrad(Tensor* dX) {
  if (grad_scale_ != 1.f) {
    auto* dx = dX->template mutable_data<T, Context>();
    math::Scale(dX->count(), grad_scale_, dx, dx, ctx());
  }
  if (clip_norm_ > 0.f) {
    auto* dx = dX->template mutable_data<T, Context>();
    float norm = std::sqrt(math::Dot(dX->count(), dx, dx, ctx()));
    if (norm > clip_norm_) {
      math::Scale(dX->count(), clip_norm_ / norm, dx, dx, ctx());
    }
  } else if (clip_value_ > 0.f) {
    auto* dx = dX->template mutable_data<T, Context>();
    kernels::Clip(dX->count(), -clip_value_, clip_value_, dx, dx, ctx());
  }
}

template <class Context>
template <typename T>
void ZeroUpdateOpBase<Context>::DoRunWithType() {
  GetArguments();
  auto* X = InitMaster<T>();
  auto* Y = ctx()->workspace()->CreateTensor("BufferShared");
  auto* dX = ctx()->workspace()->CreateTensor("BufferKernel");
  auto* IsInfAll = OutputSize() > InputSize() ? Output(-1) : nullptr;
  auto recv_count = X->count();
  auto send_count = recv_count * coll_impl_.comm_size();
  auto buff_count = send_count + recv_count;
  auto* recvbuf = Y->Reshape({buff_count})->template mutable_data<T, Context>();
  auto* sendbuf = recvbuf + recv_count;
  auto* x = X->template mutable_data<float, Context>();
  auto* dx = dX->ReshapeLike(*X)->template mutable_data<float, Context>();
  // ReduceScatter.
  if (IsInfAll) math::Set(send_count, convert::To<T>(0.f), sendbuf, ctx());
  CopyBuffer(sendbuf, true);
  coll_impl_.ReduceScatter(sendbuf, recvbuf, recv_count, ctx());
  // Check overflow.
  float isinf_all = 0.f;
  if (IsInfAll) {
    auto* IsInf = ctx()->workspace()->CreateTensor("BufferIsInf");
    auto* isinf = IsInf->Reshape({1})->template mutable_data<float, Context>();
    math::Set(1, 0.f, isinf, ctx());
    kernels::CheckFinite(recv_count, recvbuf, isinf, ctx());
    isinf_all = IsInf->template data<float, CPUContext>()[0];
    coll_impl_.AllReduce(&isinf_all, &isinf_all, 1);
    isinf_all = std::min(isinf_all, 1.f);
    IsInfAll->template mutable_data<float, CPUContext>()[0] += isinf_all;
  }
  // UpdateMaster.
  if (isinf_all > 0.f) return;
  math::Cast(recv_count, recvbuf, dx, ctx());
  TransformGrad<float>(dX);
  if (use_lr_scales_ > 0) SetLRS(dX->count(), (float*)sendbuf);
  ApplyUpdate(dX, X, Y);
  // AllGather.
  coll_impl_.AllGather(recvbuf, sendbuf, recv_count, ctx());
  CopyBuffer(sendbuf, false);
}

DEFINE_OP_REPEATED_ARG(float, ZeroUpdateOpBase, lr_scales);

#ifdef USE_CUDA
template class ZeroUpdateOpBase<CUDAContext>;
#endif
#ifdef USE_MLU
template class ZeroUpdateOpBase<MLUContext>;
#endif

} // namespace dragon
