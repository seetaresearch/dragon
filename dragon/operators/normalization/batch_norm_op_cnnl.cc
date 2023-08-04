#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/normalization/batch_norm_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLBatchNormOp<Context>::DoRunWithType() {
  using ParamT = typename CNNLType<T>::BNParamType;
  INITIALIZE_TENSOR_VIA_SPEC(Input(1), vec64_t({C_}), ParamT);
  INITIALIZE_TENSOR_VIA_SPEC(Input(2), vec64_t({C_}), ParamT);
  INITIALIZE_TENSOR_VIA_SPEC(Input(3), vec64_t({C_}), ParamT);
  INITIALIZE_TENSOR_VIA_SPEC(Input(4), vec64_t({C_}), ParamT);

  // Set descriptors.
  CNNLSetTensorDesc<ParamT>(bn_desc_, vec64_t({C_}));
  if (Input(0).ndim() == 2) {
    bn_mode_ = CNNL_BATCHNORM_SPATIAL;
    CNNLSetTensorDesc<T>(input_desc_, vec64_t({N_, 1, 1, C_}), data_format());
  } else {
    bn_mode_ = CNNL_BATCHNORM_SPATIAL;
    CNNLSetTensorDesc<T>(input_desc_, Input(0).dims(), data_format());
  }

  if (sync_stats_ > 0) {
    int64_t N = N_;
    coll_impl_.AllReduce(&N, &N, 1);
    const auto comm_size = coll_impl_.comm_size();
    const ParamT count_avg = ParamT(N * S_) / ParamT(comm_size);
    size_t scratch_size = 0;
    auto* X_mu = Output("X_mu")->Reshape({C_});
    auto* X_rsig = Output("X_rsig")->Reshape({C_});
    auto* buffer = ctx()->workspace()->template data<ParamT, Context>(
        (2 + comm_size * 4) * C_ + comm_size, "BufferKernel");
    auto* host_buffer = ctx()->workspace()->template data<ParamT, CPUContext>(
        (2 + comm_size * 2) * C_, "BufferCPUKernel");
    math::Set(comm_size, count_avg, buffer + C_ * (2 + comm_size * 4), ctx());
    CNNLSetTensorDesc<ParamT>(count_desc_, {comm_size});
    CNNLSetTensorDesc<ParamT>(syncbn_desc_, {comm_size, C_}, "NC");
    trans_impl_.Setup<ParamT>({comm_size, 2, C_}, {1, 0, 2}, ctx());
    CNNL_CHECK(cnnlGetSyncBatchNormStatsWorkspaceSize(
        ctx()->cnnl_handle(), input_desc_, &scratch_size));
    CNNL_CHECK(cnnlSyncBatchNormStats_v2(
        ctx()->cnnl_handle(),
        input_desc_,
        Input(0).template data<T, Context>(),
        ctx()->workspace()->template data<Context>(scratch_size),
        scratch_size,
        epsilon_,
        bn_desc_,
        buffer,
        bn_desc_,
        buffer + C_));
    CNRT_CHECK(cnrtMemcpyAsync(
        host_buffer,
        buffer,
        sizeof(ParamT) * C_ * 2,
        ctx()->mlu_stream(),
        cnrtMemcpyDevToHost));
    ctx()->FinishDeviceComputation();
    // HostAllGather is faster.
    coll_impl_.AllGather(host_buffer, host_buffer + C_ * 2, C_ * 2);
    CNRT_CHECK(cnrtMemcpyAsync(
        buffer + C_ * 2,
        host_buffer + C_ * 2,
        sizeof(ParamT) * comm_size * C_ * 2,
        ctx()->mlu_stream(),
        cnrtMemcpyHostToDev));
    trans_impl_.Compute<ParamT>(
        buffer + C_ * (2 + comm_size * 0),
        buffer + C_ * (2 + comm_size * 2),
        ctx()->workspace()->template data<Context>(trans_impl_.scratch_size()),
        ctx());
    CNNL_CHECK(cnnlSyncBatchNormGatherStatsWithCounts(
        ctx()->cnnl_handle(),
        syncbn_desc_,
        buffer + C_ * (2 + comm_size * 2),
        syncbn_desc_,
        buffer + C_ * (2 + comm_size * 3),
        bn_desc_,
        Input(3).template mutable_data<ParamT, Context>(), // rm
        bn_desc_,
        Input(4).template mutable_data<ParamT, Context>(), // rv
        1.f - momentum(),
        epsilon_,
        count_desc_,
        buffer + C_ * (2 + comm_size * 4),
        bn_desc_,
        X_mu->template mutable_data<ParamT, Context>(), // sm
        bn_desc_,
        X_rsig->template mutable_data<ParamT, Context>())); // sv
    CNNL_CHECK(cnnlSyncBatchNormElemt(
        ctx()->cnnl_handle(),
        input_desc_,
        Input(0).template data<T, Context>(),
        bn_desc_,
        X_mu->template data<ParamT, Context>(),
        bn_desc_,
        X_rsig->template data<ParamT, Context>(),
        bn_desc_,
        Input(1).template data<ParamT, Context>(),
        bn_desc_,
        Input(2).template data<ParamT, Context>(),
        input_desc_,
        Output(0)->template mutable_data<T, Context>()));
    return;
  }

  // Run training or inference.
  if (training_ > 0) {
    auto* X_mu = Output("X_mu")->Reshape({C_});
    auto* X_rsig = Output("X_rsig")->Reshape({C_});
    size_t scratch_size = 0;
    CNNL_CHECK(cnnlGetBatchNormForwardWorkspaceSize(
        ctx()->cnnl_handle(), input_desc_, &scratch_size));
    CNNL_CHECK(cnnlBatchNormForwardTrainingV2(
        ctx()->cnnl_handle(),
        act_desc_,
        bn_mode_,
        CNNL_BATCHNORM_OPS_BN,
        nullptr, // alpha
        nullptr, // beta
        input_desc_,
        Input(0).template data<T, Context>(), // x
        nullptr,
        nullptr,
        bn_desc_,
        Input(1).template data<ParamT, Context>(), // gamma
        Input(2).template data<ParamT, Context>(), // beta
        Input(3).template mutable_data<ParamT, Context>(), // rm
        Input(4).template mutable_data<ParamT, Context>(), // rv
        epsilon_,
        1.f - momentum(),
        input_desc_,
        Output(0)->template mutable_data<T, Context>(), // y
        X_mu->template mutable_data<ParamT, Context>(), // sm
        X_rsig->template mutable_data<ParamT, Context>(), // sv
        ctx()->workspace()->template data<Context>(scratch_size),
        scratch_size,
        nullptr,
        0));
  } else {
    CNNL_CHECK(cnnlBatchNormForwardInferenceV2(
        ctx()->cnnl_handle(),
        act_desc_,
        bn_mode_,
        CNNL_BATCHNORM_OPS_BN,
        nullptr, // alpha
        nullptr, // beta
        input_desc_,
        Input(0).template data<T, Context>(), // x
        bn_desc_,
        Input(1).template data<ParamT, Context>(), // gamma
        Input(2).template data<ParamT, Context>(), // beta
        nullptr,
        nullptr, // y
        Input(3).template data<ParamT, Context>(), // rm
        Input(4).template data<ParamT, Context>(), // rv
        epsilon_,
        input_desc_,
        Output(0)->template mutable_data<T, Context>())); // y
  }
}

template <class Context>
template <typename T>
void CNNLBatchNormGradientOp<Context>::RunTraining() {
  using ParamT = typename CNNLType<T>::BNParamType;
  auto *dX = Output(0), *dW = Output(1), *dB = Output(2);
  auto &X_mu = Input("X_mu"), &X_rsig = Input("X_rsig");

  // Set descriptors.
  CNNLSetTensorDesc<ParamT>(bn_desc_, vec64_t({C_}));
  if (Input(0).ndim() == 2) {
    bn_mode_ = CNNL_BATCHNORM_SPATIAL;
    CNNLSetTensorDesc<T>(input_desc_, vec64_t({N_, 1, 1, C_}), data_format());
  } else {
    bn_mode_ = CNNL_BATCHNORM_SPATIAL;
    CNNLSetTensorDesc<T>(input_desc_, Input(0).dims(), data_format());
  }

  if (sync_stats_ > 0) {
    int64_t N = N_;
    coll_impl_.AllReduce(&N, &N, 1);
    size_t scratch_size = 0;
    auto* buffer = ctx()->workspace()->template data<ParamT, Context>(
        2 * C_ + 1, "BufferKernel");
    CNNLSetTensorDesc<int>(count_desc_, {1});
    math::Set(1, int(N * S_), (int*)buffer + C_ * 2, ctx());
    CNNL_CHECK(cnnlGetSyncBatchnormBackwardReduceWorkspaceSize(
        ctx()->cnnl_handle(), input_desc_, &scratch_size));
    CNNL_CHECK(cnnlSyncBatchnormBackwardReduce_v2(
        ctx()->cnnl_handle(),
        input_desc_,
        Input(4).template data<T, Context>(), // dy
        input_desc_,
        Input(0).template data<T, Context>(), // x
        bn_desc_,
        X_mu.template data<ParamT, Context>(),
        bn_desc_,
        X_rsig.template data<ParamT, Context>(),
        ctx()->workspace()->template data<Context>(scratch_size),
        scratch_size,
        bn_desc_,
        dW->Reshape({C_})->template mutable_data<ParamT, Context>(),
        bn_desc_,
        dB->Reshape({C_})->template mutable_data<ParamT, Context>(),
        bn_desc_,
        buffer,
        bn_desc_,
        buffer + C_,
        true,
        true,
        true));
    ctx()->FinishDeviceComputation();
    coll_impl_.AllReduce(buffer, buffer, C_ * 2, ctx());
    CNNL_CHECK(cnnlSyncBatchNormBackwardElemtV2(
        ctx()->cnnl_handle(),
        input_desc_,
        Input(4).template data<T, Context>(), // dy
        input_desc_,
        Input(0).template data<T, Context>(), // x
        bn_desc_,
        X_mu.template data<ParamT, Context>(),
        bn_desc_,
        X_rsig.template data<ParamT, Context>(),
        bn_desc_,
        Input(1).template data<ParamT, Context>(), // gamma
        bn_desc_,
        buffer,
        bn_desc_,
        buffer + C_,
        count_desc_,
        buffer + C_ * 2,
        input_desc_,
        Output(0)->template mutable_data<T, Context>()));
    return;
  }

  size_t scratch_size = 0;
  CNNL_CHECK(cnnlGetBatchNormBackwardWorkspaceSize(
      ctx()->cnnl_handle(), input_desc_, &scratch_size));
  CNNL_CHECK(cnnlBatchNormBackward_v2(
      ctx()->cnnl_handle(),
      act_desc_,
      bn_mode_,
      CNNL_BATCHNORM_OPS_BN,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      input_desc_,
      Input(0).template data<T, Context>(), // x
      nullptr,
      nullptr,
      input_desc_,
      Input(4).template data<T, Context>(), // dy
      bn_desc_,
      Input(1).template data<ParamT, Context>(), // gamma
      nullptr,
      X_mu.template data<ParamT, Context>(), // mu
      X_rsig.template data<ParamT, Context>(), // rsig
      epsilon_,
      nullptr,
      nullptr,
      input_desc_,
      Output(0)->template mutable_data<T, Context>(), // dx
      dW->Reshape({C_})->template mutable_data<ParamT, Context>(), // dw
      dB->Reshape({C_})->template mutable_data<ParamT, Context>(), // db
      ctx()->workspace()->template data<Context>(scratch_size),
      scratch_size,
      nullptr,
      0));
}

DEPLOY_CNNL_OPERATOR(BatchNorm);
DEPLOY_CNNL_OPERATOR(BatchNormGradient);
REGISTER_CNNL_OPERATOR(SyncBatchNorm, CNNLBatchNormOp<MLUContext>);
REGISTER_CNNL_OPERATOR(
    SyncBatchNormGradient,
    CNNLBatchNormGradientOp<MLUContext>);
DEFINE_OP_SINGLE_ARG(float, CNNLBatchNormOp, momentum);

} // namespace dragon

#endif // USE_MLU
