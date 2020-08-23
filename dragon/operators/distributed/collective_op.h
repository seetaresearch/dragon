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

#ifndef DRAGON_OPERATORS_DISTRIBUTED_COLLECTIVE_OP_H_
#define DRAGON_OPERATORS_DISTRIBUTED_COLLECTIVE_OP_H_

#ifdef USE_MPI

#include "dragon/operators/distributed/collective_op_base.h"

namespace dragon {

template <class Context>
class CollectiveOp final : public CollectiveOpBase<Context> {
 public:
  CollectiveOp(const OperatorDef& def, Workspace* ws)
      : CollectiveOpBase<Context>(def, ws),
        communication_(OpArg<string>("communication", "")),
        operation_(OpArg<string>("operation", "MEAN")) {}
  USE_OPERATOR_FUNCTIONS;
  USE_COLLECTIVE_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void AllReduceMPI();

  template <typename T>
  void AllReduceNCCL();

  template <typename T>
  void BroadcastMPI();

  template <typename T>
  void BroadcastNCCL();

  template <typename T>
  void DoRunWithType();

 protected:
  string communication_, operation_;
  Tensor *src_tensor_, *dest_tensor_;
};

} // namespace dragon

#endif // USE_MPI

#endif // DRAGON_OPERATORS_DISTRIBUTED_COLLECTIVE_OP_H_
