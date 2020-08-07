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

  /*! \brief Default constructor */
  GraphOptimizer(Workspace* ws) : ws_(ws) {}

  /*! \brief Build the DAG */
  void BuildDAG(const GraphDef& graph);

  /*! \brief Eliminate the unused outputs and operators */
  GraphDef EliminateUnused(const GraphDef& graph);

  /*! \brief Plan the inplace for inputs */
  void PlanInplace(
      const GraphDef& graph,
      Map<string, Set<string>>& output_aliases);

  /*! \brief Plan the checkpoint for inputs */
  GraphDef PlanCheckpoint(
      const GraphDef& graph,
      Map<string, vec32_t>& subgraph_indices);

  /*! \brief Allocate the shared buffer for outputs */
  GraphDef SimulateGC(const GraphDef& graph);

 protected:
  /*! \brief Remote the unused nodes from a sink to all sources */
  void EliminateUnusedNode(const string& sink);

  /*! \brief Remote the unused nodes from a source to a sink */
  void EliminateUnusedNode(const string& source, const string& sink);

  /* \brief The graph workspace */
  Workspace* ws_;

  /* \brief The graph nodes */
  Map<string, Node> nodes_;

  /* \brief The traversal flags */
  Map<string, bool> visited_, used_;

  /* \brief The reference count */
  Map<string, int> reference_count_;

 private:
  DISABLE_COPY_AND_ASSIGN(GraphOptimizer);
};

} // namespace dragon

#endif // DRAGON_CORE_GRAPH_OPTIMIZER_H_
