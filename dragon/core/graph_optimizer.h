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

  /*! \brief Constructor */
  GraphOptimizer() {}

  /*! \brief Build the DAG */
  void BuildDAG(const GraphDef& graph);

  /*! \brief Plan the in-place for outputs */
  void PlanInplace(const GraphDef& graph, Map<string, Set<string>>& sources);

  /*! \brief Eliminate the intermediate outputs */
  GraphDef EliminateIntermediates(const GraphDef& graph);

 protected:
  /* \brief The graph nodes */
  Map<string, Node> nodes_;

  /* \brief The traversal flags */
  Map<string, bool> visited_, used_;

  /* \brief The inputs counter */
  Map<string, int> inputs_count_;

 private:
  DISABLE_COPY_AND_ASSIGN(GraphOptimizer);
};

} // namespace dragon

#endif // DRAGON_CORE_GRAPH_OPTIMIZER_H_
