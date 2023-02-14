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

#ifndef DRAGON_OPERATORS_VISION_ROI_ALIGN_OP_H_
#define DRAGON_OPERATORS_VISION_ROI_ALIGN_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class RoiAlignOp final : public Operator<Context> {
 public:
  RoiAlignOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        out_h_(OP_SINGLE_ARG(int64_t, "pooled_h", 0)),
        out_w_(OP_SINGLE_ARG(int64_t, "pooled_w", 0)),
        spatial_scale_(OP_SINGLE_ARG(float, "spatial_scale", 1.f)),
        sampling_ratio_(OP_SINGLE_ARG(int64_t, "sampling_ratio", 0)),
        aligned_(OP_SINGLE_ARG(int64_t, "aligned", 0)) {
    CHECK_GT(out_h_, 0) << "\npooled_h must > 0";
    CHECK_GT(out_w_, 0) << "\npooled_w must > 0";
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  float spatial_scale_;
  int64_t sampling_ratio_, aligned_;
  int64_t out_h_, out_w_;
};

template <class Context>
class RoiAlignGradientOp final : public Operator<Context> {
 public:
  RoiAlignGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        out_h_(OP_SINGLE_ARG(int64_t, "pooled_h", 0)),
        out_w_(OP_SINGLE_ARG(int64_t, "pooled_w", 0)),
        spatial_scale_(OP_SINGLE_ARG(float, "spatial_scale", 1.f)),
        sampling_ratio_(OP_SINGLE_ARG(int64_t, "sampling_ratio", 0)),
        aligned_(OP_SINGLE_ARG(int64_t, "aligned", 0)) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(1));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  float spatial_scale_;
  int64_t sampling_ratio_, aligned_;
  int64_t out_h_, out_w_;
};

#ifdef USE_MLU
template <class Context>
class CNNLRoiAlignOp : public Operator<Context> {
 public:
  CNNLRoiAlignOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        out_h_(OP_SINGLE_ARG(int64_t, "pooled_h", 0)),
        out_w_(OP_SINGLE_ARG(int64_t, "pooled_w", 0)),
        spatial_scale_(OP_SINGLE_ARG(float, "spatial_scale", 1.f)),
        sampling_ratio_(OP_SINGLE_ARG(int64_t, "sampling_ratio", 0)),
        aligned_(OP_SINGLE_ARG(int64_t, "aligned", 0)) {
    CHECK_GT(out_h_, 0) << "\npooled_h must > 0";
    CHECK_GT(out_w_, 0) << "\npooled_w must > 0";
    CNNLCreateTensorDesc(&input_desc_);
    CNNLCreateTensorDesc(&boxes_desc_);
    CNNLCreateTensorDesc(&output_desc_);
    CNNL_CHECK(cnnlCreateRoiAlignDescriptor(&pool_desc_));
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLRoiAlignOp() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNLDestroyTensorDesc(boxes_desc_);
    CNNLDestroyTensorDesc(output_desc_);
    CNNL_CHECK(cnnlDestroyRoiAlignDescriptor(pool_desc_));
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  float spatial_scale_;
  int64_t sampling_ratio_, aligned_;
  int64_t out_h_, out_w_;
  cnnlRoiAlignDescriptor_t pool_desc_;
  cnnlTensorDescriptor_t input_desc_, boxes_desc_, output_desc_;
};

template <class Context>
class CNNLRoiAlignGradientOp final : public CNNLRoiAlignOp<Context> {
 public:
  CNNLRoiAlignGradientOp(const OperatorDef& def, Workspace* ws)
      : CNNLRoiAlignOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(1));
  }

  template <typename T>
  void DoRunWithType();
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_ROI_ALIGN_OP_H_
