/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_LOSS_CTC_LOSS_OP_H_
#define DRAGON_OPERATORS_LOSS_CTC_LOSS_OP_H_

#include "core/operator.h"

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
    template <typename T> void RunWithType();
};

#ifdef WITH_CUDNN

#if CUDNN_VERSION_MIN(7, 0, 0)

template <class Context>
class CuDNNCTCLossOp final : public Operator<Context> {
 public:
    CuDNNCTCLossOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          blank_first(OperatorBase::Arg<bool>("blank_first", true)),
          padding_mask(OperatorBase::Arg<int64_t>("padding_mask", -1)) {
         CUDNN_CHECK(cudnnCreateCTCLossDescriptor(&ctc_desc));
         CUDNN_CHECK(cudnnCreateTensorDescriptor(&prob_desc));
         CUDNN_CHECK(cudnnCreateTensorDescriptor(&grad_desc));
         ctc_algo = CUDNN_CTC_LOSS_ALGO_DETERMINISTIC;
     }
     USE_OPERATOR_FUNCTIONS;

     ~CuDNNCTCLossOp() {
         CUDNN_CHECK(cudnnDestroyCTCLossDescriptor(ctc_desc));
         CUDNN_CHECK(cudnnDestroyTensorDescriptor(prob_desc));
         CUDNN_CHECK(cudnnDestroyTensorDescriptor(grad_desc));
     }

    void RunOnDevice() override;
    template <typename T> void RunWithType();
    
    void WrapIO();

 protected:
    bool blank_first;
    int64_t padding_mask;

    cudnnCTCLossAlgo_t ctc_algo;
    cudnnCTCLossDescriptor_t ctc_desc;
    cudnnTensorDescriptor_t prob_desc, grad_desc;
    size_t workspace_size;

    vector<int> packed_labels, label_lengths, input_lengths;
};

#endif

#endif  // WITH_CUDNN

}  // namespace dragon

#endif  // DRAGON_OPERATORS_LOSS_CTC_LOSS_OP_H_