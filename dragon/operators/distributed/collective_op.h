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

#include "dragon/core/operator.h"
#include "dragon/operators/distributed/collective_op_impl.h"

namespace dragon {

template <class Context>
class CollectiveOp final : public Operator<Context> {
 public:
  CollectiveOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        operation_(OP_SINGLE_ARG(string, "operation", "")),
        reduction_(OP_SINGLE_ARG(string, "reduction", "SUM")),
        bucket_size_(OP_SINGLE_ARG(int64_t, "bucket_size", 0)) {
    coll_impl_.SetBackend(OP_SINGLE_ARG(string, "backend", "MPI"));
    coll_impl_.SetReduction(reduction_);
    coll_impl_.SetComm(
        OP_SINGLE_ARG(int64_t, "comm", 0),
        OP_SINGLE_ARG(int64_t, "group", 0),
        OP_REPEATED_ARG(int64_t, "ranks"));
    if (operation_ == "ALLGATHER") bucket_size_ = 0; // Disabled.
    if (operation_ == "REDUCESCATTER") bucket_size_ = 0; // Disabled.
  }
  USE_OPERATOR_FUNCTIONS;

  void RunOnDevice() override;

  template <typename T>
  void CopyBuffer(bool cat = true);

  template <typename T>
  void DoRunWithType();

 protected:
  int src_index_;
  int64_t bucket_size_;
  string operation_, reduction_;
  Tensor *src_tensor_, *dest_tensor_;
  CollectiveOpImpl coll_impl_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_DISTRIBUTED_COLLECTIVE_OP_H_
