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

#ifndef DRAGON_CORE_GRAPH_H_
#define DRAGON_CORE_GRAPH_H_

#include "dragon/core/common.h"
#include "dragon/core/operator.h"

namespace dragon {

class Workspace;

/*!
 * \brief The base graph class.
 */
class DRAGON_API GraphBase {
 public:
  /*! \brief Constructor */
  GraphBase(const GraphDef& def, Workspace* ws);

  /*! \brief Destructor */
  virtual ~GraphBase() {}

  /*! \brief Create a new graph */
  static GraphBase* New(const GraphDef& def, Workspace* ws);

  /*! \brief Create graph in the workspace */
  virtual bool Create(const GraphDef& def) = 0;

  /*! \brief Run graph on the given stream */
  virtual bool Run(
      int stream = 0,
      const string& include = "",
      const string& exclude = "") = 0;

  /*! \brief Return the graph name */
  const string& name() const {
    return name_;
  }

  /*! \brief Return the executing phase */
  const string& phase() const {
    return phase_;
  }

  /*! \brief Return the specified argument */
  const Argument& arg(const string& name) {
    return *(args_[name]);
  }

  /*! \brief Return all the arguments */
  const Map<string, const Argument*>& args() {
    return args_;
  }

  /*! \brief Return the graph def */
  const GraphDef& def() const {
    return def_;
  }

  /*! \brief Return the optimized graph def */
  const GraphDef& optimized_def() const {
    return optimized_def_;
  }

  /*! \brief Return the parent workspace */
  Workspace* workspace() const {
    return workspace_;
  }

 protected:
  /*! \brief The graph def */
  GraphDef def_;

  /*! \brief The optimized graph def */
  GraphDef optimized_def_;

  /*! \brief The parent workspace */
  Workspace* workspace_;

  /*! \brief The graph name */
  string name_;

  /*! \brief The executing phase */
  string phase_;

  /*! \brief The arguments */
  Map<string, const Argument*> args_;

  DISABLE_COPY_AND_ASSIGN(GraphBase);
};

/*!
 * \brief Graph to execute operators sequentially.
 */
class Graph : public GraphBase {
 public:
  /*! \brief Constructor */
  Graph(const GraphDef& def, Workspace* ws);

  /*! \brief Create graph in the workspace */
  bool Create(const GraphDef& def) override;

  /*! \brief Run graph on the given stream */
  bool Run(
      int stream = 0,
      const string& include = "",
      const string& exclude = "") override;

 protected:
  /*! \brief The operators */
  vector<unique_ptr<OperatorBase>> operators_;

  /*! \brief The output sourcing tensors */
  Map<string, Set<string>> outputs_from_;
};

/* Macros */

DECLARE_REGISTRY(GraphRegistry, GraphBase, const GraphDef&, Workspace*);

#define REGISTER_GRAPH(name, ...) \
  REGISTER_CLASS(GraphRegistry, name, __VA_ARGS__)

} // namespace dragon

#endif // DRAGON_CORE_GRAPH_H_
