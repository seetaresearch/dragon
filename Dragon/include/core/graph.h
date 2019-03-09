/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_CORE_GRAPH_H_
#define DRAGON_CORE_GRAPH_H_

#include "core/common.h"
#include "core/operator.h"

namespace dragon {

class GraphBase {
 public:
    /*! \brief Default constructor */
    GraphBase(
        const GraphDef&         meta_graph,
        Workspace*              ws);

    /*! \brief Default deconstructor */
    virtual ~GraphBase() {}

    GraphDef BuildUpdateOps(const GraphDef& input_def);

    /*! \brief Create a graph from the optimized def */
    virtual bool Create(
        const GraphDef&         optimized_graph,
        Workspace*              ws) = 0;

    /*! \brief Run the graph once synchronously */
    virtual bool Run(
        const string&           include,
        const string&           exclude,
        int                     stream_id = 0) = 0;

    /*! \brief Return the name of this graph */
    string name() const { return name_; }

 protected:
    /*! \brief Store the name and running phase */
    string name_, phase_;

    /*! \brief Store the defined arguments */
    Map<string, Argument> args_;

    /*! \brief Store the parent workspace */
    Workspace* ws_;
};

class Graph : public GraphBase {
 public:
    /*! \brief Default constructor */
    Graph(const GraphDef& meta_graph, Workspace* ws);

    /*! \brief Default deconstructor */
    virtual ~Graph() { for (auto* op : ops_) delete op; }

    /*! \brief Create a graph from the optimized def */
    bool Create(
        const GraphDef&         optimized_graph,
        Workspace*              ws) override;

    /*! \brief Run the graph once synchronously */
    bool Run(
        const string&           include,
        const string&           exclude,
        int                     stream_id = 0) override;

    /*! \brief Return the parent workspace */
    Workspace* ws() const { return ws_; }

 protected:
    /*! \brief Store the internal operators */
    vector<OperatorBase*> ops_;
};

/*! \brief Create a graph from the raw def */
GraphBase* NewGraph(
    const GraphDef&             meta_graph,
    Workspace*                  ws);

DECLARE_REGISTRY(
    GraphRegistry,
    GraphBase,
    const GraphDef&,
    Workspace*);

#define REGISTER_GRAPH(name, ...) \
    REGISTER_CLASS(GraphRegistry, name, __VA_ARGS__)

}  // namespace dragon

#endif  // DRAGON_CORE_GRAPH_H_