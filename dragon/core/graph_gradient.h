/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *    <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_CORE_GRAPH_GRADIENT_H_
#define DRAGON_CORE_GRAPH_GRADIENT_H_

#include "dragon/core/common.h"

namespace dragon {

class DRAGON_API GraphGradientMaker {
 public:
  /*! \brief Generate a backward graph from the forward ops */
  void Make(
      const vector<OperatorDef*>& forward_ops,
      const vector<string>& targets,
      const vector<string>& input_grads,
      GraphDef& backward_def);

  /*! \brief Rewrite a graph to share the intermediate grads */
  GraphDef Share(const GraphDef& input_def);

  /*! \brief Add a hooked gradient */
  void add_hooked_grad(const string& name) {
    hooked_grads_.insert(name);
  }

  /*! \brief Add an ignored gradient */
  void add_ignored_grad(const string& name) {
    ignored_grads_.insert(name);
  }

  /*! \brief Set the prefix of backward op name */
  void set_op_prefix(const string& prefix) {
    op_prefix_ = prefix;
  }

 private:
  /*! \brief Check the missing grads of backward procedure */
  bool CheckGrad(
      const OperatorDef& forward_op,
      const Set<string>& targets,
      vector<pair<string, int>>& gen_grads);

  /*! \brief Return a dummy operator name */
  string GetOperatorName() {
    if (op_prefix_.empty()) return "Generic";
    return op_prefix_ + str::to(op_index_++);
  }

  /*! \brief Store the mapping of intermediate grads */
  Map<string, string> inputs_to_grads_;

  /*! \brief Store the non-gradient outputs */
  Set<string> blacklist_set_;

  /*! \brief Store the non-shared gradients */
  Set<string> hooked_grads_;

  /*! \brief Store the gradients that are not required */
  Set<string> ignored_grads_;

  /*! \brief Store the prefix of dummy operator name */
  string op_prefix_;

  /*! \brief Store the counter of dummy operator name */
  int64_t op_index_ = 0;
};

} // namespace dragon

#endif // DRAGON_CORE_GRAPH_GRADIENT_H_
