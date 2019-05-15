#include "core/graph.h"
#include "core/workspace.h"
#include "core/graph_gradient.h"
#include "core/graph_optimizer.h"

namespace dragon {

/* Default constructor of <GraphBase> */

GraphBase::GraphBase(const GraphDef& def, Workspace* ws)
    : name_(def.name()), ws_(ws) {
    for (auto arg : def.arg()) {
        CHECK_GT(arg.name().size(), 0);
        CHECK_EQ(args_.count(arg.name()), 0);
        args_[arg.name()] = arg;
    }

    Set<string> known_tensors;

    // Topo-check for a graph
    for (const auto& op : def.op()) {
        // Check inputs
        for (const auto& in : op.input())
            CHECK(known_tensors.count(in) || ws_->HasTensor(in))
                << "\nInput: " << in << " for op: "
                << op.name() << " is unknown.";
        // Add outputs
        for (const auto& out : op.output()) known_tensors.insert(out);
    }

    // Check for all solving targets
    Set<string> objective_targets;
    for (const auto& target : def.output()) {
        CHECK(known_tensors.count(target) ||
              ws_->HasTensor(target))
            << "\nTarget: " << target
            << " does not exist in computional graph.";
        objective_targets.insert(target);
    }

    // Check for all gradients
    for (const auto& gradient : def.gradient()) {
        const auto& cost = gradient.cost();
        const auto& wrt = gradient.wrt();
        CHECK(known_tensors.count(cost) || ws_->HasTensor(cost))
            << "\nTarget: " << cost
            << "_grad does not exist in computional graph.";
        CHECK(known_tensors.count(wrt) || ws_->HasTensor(wrt))
            << "\nTarget: " << wrt
            << "_grad does not exist in computional graph.";
        CHECK_GT(objective_targets.count(cost), 0)
            << "\nTo solve d(" << cost << ")/d(" << wrt << "), "
            << "must set " << cost
            << "\nas a objective tensor to solve before derivating it.";
    }
}

/* Create a graph from the optimized def */

bool Graph::Create(const GraphDef& def, Workspace* ws) {
    bool has_device_option = def.has_device_option();
    for (int i = 0; i < def.op_size(); i++) {
        OperatorDef op_def(def.op(i));
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
        OperatorBase* op = NewOperator(op_def, ws);
        ops_.push_back(op);
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
    Map< string, vec32_t > subgraph_indices;
    int opt = 3;  // defaults: O3
    if (this->args_.count("optimization_level"))
        opt = this->args_["optimization_level"].i();
    if (opt >= 1) opt_def = graph_optim.PruneNodes(def);
    if (opt >= 2) opt_def = graph_optim.AddInplace(opt_def);
    if (opt >= 3) {
        if (this->args_["phase"].s() == "TRAIN") {
            opt_def = graph_optim.MirrorStage(
                opt_def, subgraph_indices);
            opt_def = gradient_maker.Share(opt_def);
        } else {
            opt_def = graph_optim.SimulateGC(opt_def);
        }
    }

    // Try to store the final graph as a tensor for visualization
    bool could_be_serialized = true;
    for (auto& op : opt_def.op())
        if (op.type() == "GivenTensorFill")
            could_be_serialized = false;
    if (could_be_serialized) {
        ws_->CreateTensor(
            "/graph_def/optimized/" + opt_def.name())
            ->Reshape({ 1 })
            ->mutable_data<string, CPUContext>()[0]
            = opt_def.DebugString();
    }

    // Create
    Create(opt_def, ws);

    // Recomputing-aware
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
        if (!include.empty())
            if (op->type().find(include) == string::npos) continue;
        if (!exclude.empty())
            if (op->type().find(exclude) != string::npos) continue;
        op->SwitchToPhase(this->args_["phase"].s());
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