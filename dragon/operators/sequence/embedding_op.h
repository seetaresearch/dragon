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

#ifndef DRAGON_OPERATORS_SEQUENCE_EMBEDDING_OP_H_
#define DRAGON_OPERATORS_SEQUENCE_EMBEDDING_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

#ifdef USE_MLU
template <class Context>
class CNNLEmbeddingOp : public Operator<Context> {
 public:
  CNNLEmbeddingOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        padding_index_(OP_SINGLE_ARG(int64_t, "padding_index", -1)) {
    CNNLCreateTensorDesc(&input_desc_);
    CNNLCreateTensorDesc(&index_desc_);
    CNNLCreateTensorDesc(&output_desc_);
  }
  USE_OPERATOR_FUNCTIONS;

  ~CNNLEmbeddingOp() {
    CNNLDestroyTensorDesc(input_desc_);
    CNNLDestroyTensorDesc(index_desc_);
    CNNLDestroyTensorDesc(output_desc_);
  }

  void RunOnDevice() override {
    DispatchHelper<dtypes::Numerical>::Call(this, Input(0));
  }

  template <typename T>
  void DoRunWithType();

 protected:
  int64_t padding_index_;
  cnnlTensorDescriptor_t input_desc_, index_desc_, output_desc_;
};

template <class Context>
class CNNLEmbeddingGradientOp final : public CNNLEmbeddingOp<Context> {
 public:
  CNNLEmbeddingGradientOp(const OperatorDef& def, Workspace* ws)
      : CNNLEmbeddingOp<Context>(def, ws) {}
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override {
    DispatchHelper<dtypes::Floating>::Call(this, Input(1));
  }

  template <typename T>
  void DoRunWithType();
};
#endif // USE_MLU

} // namespace dragon

#endif // DRAGON_OPERATORS_SEQUENCE_EMBEDDING_OP_H_
