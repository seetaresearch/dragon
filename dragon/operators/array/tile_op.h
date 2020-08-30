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

#ifndef DRAGON_OPERATORS_ARRAY_TILE_OP_H_
#define DRAGON_OPERATORS_ARRAY_TILE_OP_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class TileOp final : public Operator<Context> {
 public:
  TileOp(const OperatorDef& def, Workspace* ws) : Operator<Context>(def, ws) {
    INIT_OP_REPEATED_ARG_WITH_DESC(int64_t, repeats);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  DECLARE_OP_REPEATED_ARG_WITH_DESC(int64_t, repeats);
};

template <class Context>
class TileGradientOp final : public Operator<Context> {
 public:
  TileGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {
    INIT_OP_REPEATED_ARG_WITH_DESC(int64_t, repeats);
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void DoRunWithType();

 protected:
  Tensor *dest_, *src_, nav_;
  int64_t axis_, repeat_;
  DECLARE_OP_REPEATED_ARG_WITH_DESC(int64_t, repeats);
};

DEFINE_OP_REPEATED_ARG_WITH_DESC(int64_t, TileOp, repeats);
DEFINE_OP_REPEATED_ARG_WITH_DESC(int64_t, TileGradientOp, repeats);

} // namespace dragon

#endif // DRAGON_OPERATORS_ARRAY_TILE_OP_H_
