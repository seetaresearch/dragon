// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_CORE_GRAPH_H_
#define DRAGON_CORE_GRAPH_H_

#include "core/common.h"
#include "core/operator.h"

namespace dragon {

class GraphBase {
 public:
    struct Node {
        vector<string> parents;
        vector<string> childs;
        int op_idx = -1;
        OperatorDef op_def;
    };

    GraphBase(
        const GraphDef&         meta_graph,
        Workspace*              ws);
    virtual ~GraphBase() {}

    virtual bool Create(
        const GraphDef&         optimized_graph,
        Workspace*              ws) = 0;

    virtual bool Run(
        const string&           include,
        const string&           exclude,
        const int               stream_id = 1) = 0;

    inline string name() const { return name_; }

 protected:
    string name_, phase_;
    Map<string, Argument> args_;
    Workspace* ws_;
};

class Graph final : public GraphBase {
 public:
    Graph(const GraphDef& meta_graph, Workspace* ws);
    ~Graph() { for (auto* op : ops_) delete op; }

    bool Create(
        const GraphDef&         optimized_graph,
        Workspace*              ws) override;

    bool Run(
        const string&           include,
        const string&           exclude,
        const int               stream_id = 1) override;

    GraphDef Prune(const GraphDef& meta_graph);
    GraphDef MakeUpdate(const GraphDef& meta_graph);
    GraphDef Share(const GraphDef& optimized_graph);
    void ShareGrads(GraphDef& optimized_graph);

    void RecomputingAware(
        const GraphDef&         optimized_graph,
        Workspace*              ws);

    inline Workspace* ws() const { return ws_; }

 private:
    void ForwardShareDyeing(string u, string ancestor);
    void ForwardPruneDyeing(
        string                  u,
        string                  leaf,
        vector<string>          path);
    void BackwardPruneDyeing(string v);

    vector<OperatorBase*> ops_;
    Map<string, Node> dag_;
    Map<string, bool> visited_, colored_;
    Map<string, string> renamed_;
    Set<string> targets_;
};

GraphBase* NewGraph(
    const GraphDef&             meta_graph,
    Workspace*                  ws);

DECLARE_REGISTRY(
    GraphRegistry,
    GraphBase,
    const GraphDef&,
    Workspace*);

}    // namespace dragon

#endif    // DRAGON_CORE_GRAPH_H_