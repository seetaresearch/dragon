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

#ifndef DRAGON_CORE_GRAPH_OPTIMIZER_H_
#define DRAGON_CORE_GRAPH_OPTIMIZER_H_

#include "dragon/core/common.h"

namespace dragon {

class Workspace;

class GraphOptimizer {
 public:
  /*! \brief The simple node structure */
  struct Node {
    vector<string> parents;
    vector<string> childs;
    int op_idx = -1;
    OperatorDef op_def;
  };

  /*! \brief Default constructor */
  GraphOptimizer(Workspace* ws) : ws_(ws) {}

  /*! \brief Build the DAG resources for given def */
  void BuildDAG(const GraphDef& input_def);

  /*! \brief Prune the redundant nodes (-O1) */
  GraphDef PruneNodes(const GraphDef& input_def);

  /*! \brief Add the inplace for outputs (-O2) */
  void AddInplace(
      const GraphDef& input_def,
      Map<string, Set<string>>& output_aliases);

  /*! \brief Plan the recomputing for inputs (-O3) */
  GraphDef MirrorStage(
      const GraphDef& input_def,
      Map<string, vec32_t>& op_indices);

  /*! \brief Allocate the buffer for outputs (-O3) */
  GraphDef SimulateGC(const GraphDef& input_def);

 protected:
  /*! \brief Pass from gradients to remove unused nodes */
  void ForwardPrunePass(
      const string& u,
      const string& leaf,
      const std::deque<string>& path);

  /*! \brief Pass from targets to remove unused nodes */
  void BackwardPrunePass(const string& v);

  /* \brief Store the workspace of parent graph */
  Workspace* ws_;

  /* \brief Store the DAG */
  Map<string, Node> dag_;

  /* \brief Store the traversal flags */
  Map<string, bool> visited_, colored_;

  /* \brief Store the count of references */
  Map<string, int> reference_count_;
};

} // namespace dragon

#endif // DRAGON_CORE_GRAPH_OPTIMIZER_H_
