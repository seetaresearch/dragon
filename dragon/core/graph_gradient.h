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

#ifndef DRAGON_CORE_GRAPH_GRADIENT_H_
#define DRAGON_CORE_GRAPH_GRADIENT_H_

#include "dragon/core/common.h"

namespace dragon {

class DRAGON_API GraphGradientMaker {
 public:
  /*! \brief Generate graph from the executed ops */
  void Make(
      const vector<OperatorDef*>& ops,
      const vector<string>& targets,
      const vector<string>& input_grads,
      GraphDef& graph);

  /*! \brief Eliminate the unused and make sharing of outputs */
  GraphDef Optimize(const GraphDef& graph);

  /*! \brief Add an empty gradient */
  void add_empty_grad(const string& name) {
    empty_grads_.insert(name);
  }

  /*! \brief Add a retained gradient */
  void add_retained_grad(const string& name) {
    retained_grads_.insert(name);
  }

  /*! \brief Set the prefix of backward op name */
  void set_op_prefix(const string& prefix) {
    op_prefix_ = prefix;
  }

 private:
  /*! \brief Check the missing grads */
  bool CheckGrad(
      const OperatorDef& op,
      const Set<string>& targets,
      vector<pair<string, int>>& gen_grads);

  /*! \brief Return a dummy operator name */
  string GetOperatorName() {
    if (op_prefix_.empty()) return "GradientOp";
    return op_prefix_ + str::to(op_idx_++);
  }

  /*! \brief The mapping from input to grad */
  Map<string, string> inputs_to_grads_;

  /*! \brief The gradients should be retained */
  Set<string> retained_grads_;

  /*! \brief The gradients should be set to empty */
  Set<string> empty_grads_;

  /*! \brief The prefix of op name */
  string op_prefix_;

  /*! \brief The counter of op name */
  int64_t op_idx_ = 0;
};

} // namespace dragon

#endif // DRAGON_CORE_GRAPH_GRADIENT_H_
