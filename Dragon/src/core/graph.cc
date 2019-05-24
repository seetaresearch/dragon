#include "core/graph.h"
#include "core/workspace.h"
#include "core/graph_gradient.h"
#include "core/graph_optimizer.h"

namespace dragon {

/* Default constructor of <GraphBase> */

GraphBase::GraphBase(const GraphDef& def, Workspace* ws)
    : def_(def), ws_(ws), name_(def.name()), phase_("TEST") {
    // Scan the defined arguments
    for (auto& arg : def_.arg()) {
        CHECK_GT(arg.name().size(), 0);
        CHECK_EQ(args_.count(arg.name()), 0);
        args_[arg.name()] = &arg;
        if (arg.name() == "phase") phase_ = arg.s();
    }

    // Collect outputs
    Set<string> outputs;
    for (const auto& op : def.op()) {
        for (const auto& in : op.input())
            CHECK(outputs.count(in) || ws_->HasTensor(in))
                << "\nInput: " << in << " for op: "
                << op.name() << " is unknown.";
        for (const auto& out : op.output()) outputs.insert(out);
    }

    // Check targets
    Set<string> targets;
    for (const auto& target : def.output()) {
        CHECK(outputs.count(target) || ws_->HasTensor(target))
            << "\nTarget: " << target
            << " does not exist in the graph.";
        targets.insert(target);
    }

    // Check gradients
    for (const auto& gradient : def.gradient()) {
        const auto& cost = gradient.cost();
        const auto& wrt = gradient.wrt();
        CHECK(outputs.count(cost) || ws_->HasTensor(cost))
            << "\nTarget: " << cost
            << "does not exist in the graph.";
        CHECK(outputs.count(wrt) || ws_->HasTensor(wrt))
            << "\nTarget: " << wrt
            << "does not exist in the graph.";
        CHECK_GT(targets.count(cost), 0)
            << "\nTo solve d(" << cost << ")/d(" << wrt << "),\n"
            << cost << " should be set as a target.";
    }
}

/* Create a graph from the optimized def */

bool Graph::Create(const GraphDef& def, Workspace* ws) {
    this->opt_def_ = def;  // Store for debugging
    bool has_device_option = def.has_device_option();
    for (int i = 0; i < def.op_size(); i++) {
        auto op_def(def.op(i));
        LOG(DEBUG) << "Create Operator " << op_def.name()
                   << ": " << op_def.type();
        // Inherit device option if necessary
        if (!op_def.has_device_option() && has_device_option)
            op_def.mutable_device_option()
                ->CopyFrom(def.device_option());
        // For the static graph, do recomputing-aware
        Argument arg; arg.set_name("allow_recomp");
        arg.set_i(1); op_def.add_arg()->CopyFrom(arg);
        // For the last operator, enforce the synchronization
        if (i == def.op_size() - 1) {
            arg.set_name("do_sync");
            arg.set_i(1); op_def.add_arg()->CopyFrom(arg);
        }
        ops_.push_back(NewOperator(op_def, ws));
    }
    return true;
}

/* Default constructor of <Graph> */

Graph::Graph(const GraphDef& def, Workspace* ws)
    : GraphBase(def, ws) {
    // Apply the optimizations
    GraphDef opt_def = def;
    GraphOptimizer graph_optim(ws);
    GraphGradientMaker gradient_maker;
    Map<string, vec32_t> subgraph_indices;
    int opt = 3;  // defaults: O3
    if (args().count("optimization_level"))
        opt = arg("optimization_level").i();
    if (opt >= 1) opt_def = graph_optim.PruneNodes(def);
    if (opt >= 2) opt_def = graph_optim.AddInplace(opt_def);
    if (opt >= 3) {
        if (phase() == "TRAIN") {
            opt_def = graph_optim.MirrorStage(
                opt_def, subgraph_indices);
            opt_def = gradient_maker.Share(opt_def);
        } else {
            opt_def = graph_optim.SimulateGC(opt_def);
        }
    }

    // Create
    Create(opt_def, ws);

    // Recomputation and SubGraph
    if (subgraph_indices.size() > 0) {
        Map<string, vector<OperatorBase*>> subgraph;
        for (const auto& it : subgraph_indices) {
            subgraph[it.first] = vector<OperatorBase*>();
            for (const auto& idx : subgraph_indices[it.first])
                subgraph[it.first].push_back(ops_[idx]);
        }
        for (const auto& op : ops_) op->set_subgraph(subgraph);
    }
}

/* Run the graph once synchronously */

bool Graph::Run(
    const string&               include,
    const string&               exclude,
    int                         stream_id) {
    LOG(DEBUG) << "Run Graph: " << name();
    for (auto op : ops_) {
        if (!include.empty() &&
            !str::find(op->type(), include)
            ) continue;
        if (!exclude.empty() &&
            str::find(op->type(), exclude)
            ) continue;
        op->SwitchToPhase(phase());
        LOG(DEBUG) << "$ Before Operator: " << op->name();
        op->Run(stream_id);
        LOG(DEBUG) << "$ After Operator: " << op->name();
    }
    return true;
}

/* New a graph from the raw def */

GraphBase* NewGraph(
    const GraphDef&             meta_graph,
    Workspace*                  ws) {
    if (!meta_graph.has_graph_type() ||
        meta_graph.graph_type().empty()) {
        return new Graph(meta_graph, ws);
    }
    return GraphRegistry()->Create(
        meta_graph.graph_type(), meta_graph, ws);
}

/* Graph Registry */

DEFINE_REGISTRY(
    GraphRegistry,
    GraphBase,
    const GraphDef&,
    Workspace*
);

}  // namespace dragon