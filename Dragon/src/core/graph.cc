#include "core/graph.h"
#include "core/workspace.h"
#include "core/graph_gradient.h"
#include "core/graph_optimizer.h"

namespace dragon {

/*! Default constructor of <GraphBase> */

GraphBase::GraphBase(const GraphDef& meta_graph, Workspace* ws)
    : name_(meta_graph.name()), ws_(ws) {
    for (auto arg : meta_graph.arg()) {
        CHECK_GT(arg.name().size(), 0);
        CHECK_EQ(args_.count(arg.name()), 0);
        args_[arg.name()] = arg;
    }

    Set<string> known_tensors;

    // Topo-check for a graph
    for (const auto& op : meta_graph.op()) {
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
    for (const auto& target : meta_graph.output()) {
        CHECK(known_tensors.count(target) ||
              ws_->HasTensor(target))
            << "\nTarget: " << target
            << " does not exist in computional graph.";
        objective_targets.insert(target);
    }

    // Check for all gradients
    for (const auto& gradient : meta_graph.gradient()) {
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

/*! Build the update operators from the def */

GraphDef GraphBase::BuildUpdateOps(const GraphDef& input_def) {
    OperatorDef collective_op;
    collective_op.set_type("CollectiveUpdate");

    // Generate Update Ops
    vector<OperatorDef> update_ops;
    for (const auto& updater : input_def.updater()) {
        vector<string> missing_tensors;
        for (const auto& tensor : updater.tensor()) {
            if (!ws_->HasTensor(tensor)) {
                LOG(INFO) << "Missing Tensor: " << tensor;
                missing_tensors.push_back(tensor);
            }
        }
        if (missing_tensors.size() == 0) {
            vector<Argument> args;
            for (const auto& arg : updater.arg()) args.push_back(arg);
            OperatorDef op_def = MakeOperatorDef(updater.type(),
                                                 updater.name(),
                          vector<string>({ updater.tensor(1) }),  // dX
                          vector<string>({ updater.tensor(0) })); // X
            collective_op.add_input(updater.tensor(1));
            collective_op.add_output(updater.tensor(1));
            op_def.mutable_arg()->CopyFrom(updater.arg());
            update_ops.push_back(op_def);
        } else {
            LOG(INFO) << "Missing tensors. Skip the update to Tensor("
                      << updater.tensor(0) << ")";
        }
    }

    // Generate Collective Ops if necessary
    vector<OperatorDef> collective_ops;
    if (args_.count("parallel_mode")) {
        if (args_["parallel_mode"].s() == "MPI" ||
            args_["parallel_mode"].s() == "NCCL") {
            OperatorDef op_def;
            op_def.CopyFrom(collective_op);
            Argument collective_mode;
            collective_mode.set_name("mode");
            collective_mode.set_s(
                args_["parallel_mode"].s() + "_ALLREDUCE");
            op_def.add_arg()->CopyFrom(collective_mode);
            if (args_.count("comm") &&
                args_.count("group") &&
                args_.count("root")) {
                op_def.add_arg()->CopyFrom(args_["comm"]);
                op_def.add_arg()->CopyFrom(args_["group"]);
                op_def.add_arg()->CopyFrom(args_["root"]);
            } else {
                LOG(FATAL) << "MPI was not initialized.";
            }
            collective_ops.push_back(op_def);
        }
    }

    // Generate graph
    GraphDef update_graph(input_def);
    update_graph.clear_updater();
    for (const auto& op : collective_ops) update_graph.add_op()->CopyFrom(op);
    for (const auto& op : update_ops) update_graph.add_op()->CopyFrom(op);
    return update_graph;
}

/*! Create a graph from the optimized def */

bool Graph::Create(
    const GraphDef&             optimized_graph,
    Workspace*                  ws) {
    bool has_device_option = optimized_graph.has_device_option();
    for (int i = 0; i < optimized_graph.op_size(); i++) {
        OperatorDef op_def(optimized_graph.op(i));
        LOG(DEBUG) << "Create Operator " << op_def.name()
                   << ": " << op_def.type();
        // Inherit device option if necessary
        if (!op_def.has_device_option() && has_device_option)
            op_def.mutable_device_option()->CopyFrom(
                optimized_graph.device_option());
        // For the static graph, do recomputing-aware
        Argument arg; arg.set_name("allow_recomputing");
        arg.set_i(1); op_def.add_arg()->CopyFrom(arg);
        // For the last operator, enforce the synchronization
        if (i == optimized_graph.op_size() - 1) {
            arg.set_name("do_sync");
            arg.set_i(1); op_def.add_arg()->CopyFrom(arg);
        }
        OperatorBase* op = NewOperator(op_def, ws);
        ops_.push_back(op);
    }
    return true;
}

/*! Default constructor of <Graph> */

Graph::Graph(const GraphDef& meta_graph, Workspace* ws)
    : GraphBase(meta_graph, ws) {
    GraphDef optimized_graph;
    Map< string, vector<int> > subgraph_indices;
    if (meta_graph.updater_size() > 0) {
        /*!
         * Check if existing any updaters.
         *
         * Note that the graph with update ops is not a dag,
         * we should handle them independently.
         */
        optimized_graph = this->BuildUpdateOps(meta_graph);
    } else {
        int OX = 3;  // defaults: O3
        if (this->args_.count("optimization_level"))
            OX = this->args_["optimization_level"].i();
        optimized_graph = meta_graph;
        GraphOptimizer optimizer(ws);
        GraphGradientMaker gradient_maker;
        if (OX >= 1) optimized_graph = optimizer.PruneNodes(meta_graph);
        if (OX >= 2) optimized_graph = optimizer.AddInplace(optimized_graph);
        if (OX >= 3) {
            if (this->args_["phase"].s() == "TRAIN") {
                optimized_graph = optimizer.MirrorStage(
                    optimized_graph, subgraph_indices);
                gradient_maker.Share(optimized_graph);
            } else {
                optimized_graph = optimizer.SimulateGC(optimized_graph);
            }
        }
    }

    // Try to store the final graph as a tensor for visualization
    bool could_be_serialized = true;
    for (auto& op : optimized_graph.op())
        if (op.type() == "GivenTensorFill")
            could_be_serialized = false;
    if (could_be_serialized) {
        auto* T = ws_->CreateTensor(
            "/graph_def/optimized/" +
                meta_graph.name())->Reshape({ 1 });
        T->mutable_data<string, CPUContext>()[0]
            = optimized_graph.DebugString();
    }

    // Create
    Create(optimized_graph, ws);

    // Recomputing-aware
    if (subgraph_indices.size() > 0) {
        Map< string, vector<OperatorBase*> > subgraph;
        for (const auto& it : subgraph_indices) {
            subgraph[it.first] = vector<OperatorBase*>();
            for (const auto& idx : subgraph_indices[it.first])
                subgraph[it.first].push_back(ops_[idx]);
        }
        for (const auto& op : ops_) op->set_subgraph(subgraph);
    }
}

/*! Run the graph once synchronously */

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

/*! New a graph from the raw def */

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

/*! Graph Registry */

DEFINE_REGISTRY(
    GraphRegistry,
    GraphBase,
    const GraphDef&,
    Workspace*
);

}  // namespace dragon