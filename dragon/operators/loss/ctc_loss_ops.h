/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_LOSS_CTC_LOSS_OP_H_
#define DRAGON_OPERATORS_LOSS_CTC_LOSS_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class CTCLossOp final : public Operator<Context> {
 public:
  CTCLossOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    LOG(FATAL) << "CTCLoss requires CuDNN support.";
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {}
};

template <class Context>
class CTCLossGradientOp final : public Operator<Context> {
 public:
  CTCLossGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();
};

#ifdef USE_CUDNN

template <class Context>
class CuDNNCTCLossOp final : public Operator<Context> {
 public:
  CuDNNCTCLossOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        padding_mask_(OP_SINGLE_ARG(int64_t, "padding_mask", -1)) {
    CuDNNCreateTensorDesc(&prob_desc_);
    CuDNNCreateTensorDesc(&grad_desc_);
    ctc_algo_ = CUDNN_CTC_LOSS_ALGO_DETERMINISTIC;
    CUDNN_CHECK(cudnnCreateCTCLossDescriptor(&ctc_desc_));
  }
  USE_OPERATOR_FUNCTIONS;

  ~CuDNNCTCLossOp() {
    CuDNNDestroyTensorDesc(&prob_desc_);
    CuDNNDestroyTensorDesc(&grad_desc_);
    CUDNN_CHECK(cudnnDestroyCTCLossDescriptor(ctc_desc_));
  }

  void RunOnDevice() override;

  void Reshape();

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t padding_mask_;
  size_t workspace_size_;
  cudnnCTCLossAlgo_t ctc_algo_;
  cudnnCTCLossDescriptor_t ctc_desc_;
  cudnnTensorDescriptor_t prob_desc_, grad_desc_;
  vec32_t packed_labels_, label_lengths_, input_lengths_;
};

#endif // USE_CUDNN

} // namespace dragon

#endif // DRAGON_OPERATORS_LOSS_CTC_LOSS_OP_H_
