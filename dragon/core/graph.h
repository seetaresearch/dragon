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

class DRAGON_API GraphBase {
 public:
  /*! \brief Default constructor */
  GraphBase(const GraphDef&, Workspace*);

  /*! \brief Default Destructor */
  virtual ~GraphBase() {}

  /*! \brief Create a graph from the optimized def */
  virtual bool Create(const GraphDef&, Workspace*) = 0;

  /*! \brief Run the graph once synchronously */
  virtual bool Run(const string&, const string&, int = 0) = 0;

  /*! \brief Return the graph name */
  const string& name() const {
    return name_;
  }

  /*! \brief Return the defined running phase */
  const string& phase() const {
    return phase_;
  }

  /*! \brief Return the specified argument */
  const Argument& arg(const string& name) {
    return *(args_[name]);
  }

  /*! \brief Return the argument map */
  const Map<string, const Argument*>& args() {
    return args_;
  }

  /*! \brief Return the stored raw def */
  const GraphDef& def() const {
    return def_;
  }

  /*! \brief Return the stored opt def */
  const GraphDef& opt_def() const {
    return opt_def_;
  }

  /*! \brief Return the parent workspace */
  Workspace* ws() const {
    return ws_;
  }

 protected:
  /*! \brief Store the name and running phase */
  string name_, phase_;

  /*! \brief Store the defined arguments */
  Map<string, const Argument*> args_;

  /*! \brief Store the parent workspace */
  Workspace* ws_;

  /*! \brief Store the graph definition */
  GraphDef def_, opt_def_;
};

class Graph : public GraphBase {
 public:
  /*! \brief Default constructor */
  Graph(const GraphDef& def, Workspace* ws);

  /*! \brief Default Destructor */
  virtual ~Graph() {
    for (auto* cached_op : cached_ops_) {
      delete cached_op;
    }
  }

  /*! \brief Create a graph from the optimized def */
  bool Create(const GraphDef&, Workspace*) override;

  /*! \brief Run the graph once synchronously */
  bool Run(const string&, const string&, int = 0) override;

 protected:
  /*! \brief The cached operators */
  vector<OperatorBase*> cached_ops_;

  /*! \brief Store the candidate output aliases */
  Map<string, Set<string>> output_aliases_;
};

/*! \brief Create a graph from the raw def */
GraphBase* NewGraph(const GraphDef&, Workspace*);

/* Macros */

DECLARE_REGISTRY(GraphRegistry, GraphBase, const GraphDef&, Workspace*);

#define REGISTER_GRAPH(name, ...) \
  REGISTER_CLASS(GraphRegistry, name, __VA_ARGS__)

} // namespace dragon

#endif // DRAGON_CORE_GRAPH_H_
