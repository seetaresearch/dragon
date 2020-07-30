#include <regex>

#include "dragon/core/graph.h"
#include "dragon/core/graph_gradient.h"
#include "dragon/core/graph_optimizer.h"
#include "dragon/core/workspace.h"

namespace dragon {

GraphBase::GraphBase(const GraphDef& def, Workspace* ws)
    : def_(def), ws_(ws), name_(def.name()), phase_("TEST") {
  // Collect arguments
  for (auto& arg : def_.arg()) {
    CHECK_GT(arg.name().size(), 0);
    CHECK_EQ(args_.count(arg.name()), 0);
    args_[arg.name()] = &arg;
    if (arg.name() == "phase") phase_ = arg.s();
  }

  // Collect outputs
  Set<string> outputs;
  for (const auto& op : def.op()) {
    for (const auto& input : op.input())
      CHECK(outputs.count(input) || ws_->HasTensor(input))
          << "\nThe input <" << input << "> is not in graph.";
    for (const auto& output : op.output()) {
      outputs.insert(output);
    }
  }

  // Check targets
  Set<string> targets;
  for (const auto& target : def.output()) {
    CHECK(outputs.count(target) || ws_->HasTensor(target))
        << "\nThe output <" << target << "> is not in graph.";
    targets.insert(target);
  }

  // Check gradients
  for (const auto& grad_info : def.grad_info()) {
    const auto& y = grad_info.y();
    CHECK_GT(targets.count(y), 0)
        << "\nThe derivative target <" << y << "> is not in outputs.";
    for (const auto& x : grad_info.xs()) {
      CHECK(outputs.count(x) || ws_->HasTensor(x))
          << "\nThe differentiated input <" << x << "> is not in graph.";
    }
  }
}

bool Graph::Create(const GraphDef& def) {
  this->optimized_def_ = def; // Store for debugging
  bool has_device_option = def.has_device_option();
  for (int i = 0; i < def.op_size(); i++) {
    auto op_def(def.op(i));
    LOG(DEBUG) << "Create Operator " << op_def.name() << ": " << op_def.type();
    // Inherit device option if necessary
    if (!op_def.has_device_option() && has_device_option) {
      op_def.mutable_device_option()->CopyFrom(def.device_option());
    }
    Argument arg;
    // For the last operator, enforce the synchronization
    if (i == def.op_size() - 1) {
      arg.set_name("do_sync");
      arg.set_i(1);
      op_def.add_arg()->CopyFrom(arg);
    }
    cached_ops_.push_back(NewOperator(op_def, ws_));
    cached_ops_.back()->set_output_aliases(output_aliases_);
  }
  return true;
}

Graph::Graph(const GraphDef& def, Workspace* ws) : GraphBase(def, ws) {
  // Apply the optimizations
  GraphDef def_v2(def);
  GraphOptimizer graph_optimizer(ws);
  GraphGradientMaker gradient_maker;
  Map<string, vec32_t> subgraph_indices;
  int opt = 3; // default: O3
  if (args().count("optimization")) opt = arg("optimization").i();
  if (opt >= 1) def_v2 = graph_optimizer.PruneNodes(def);
  if (opt >= 2) graph_optimizer.AddInplace(def_v2, output_aliases_);
  if (opt >= 3) {
    if (phase() == "TRAIN") {
      def_v2 = graph_optimizer.MirrorStage(def_v2, subgraph_indices);
      def_v2 = gradient_maker.Share(def_v2);
    } else {
      def_v2 = graph_optimizer.SimulateGC(def_v2);
    }
  }

  // Create
  Create(def_v2);

  // Recomputation and SubGraph
  if (subgraph_indices.size() > 0) {
    Map<string, vector<OperatorBase*>> subgraph;
    for (const auto& it : subgraph_indices) {
      subgraph[it.first] = vector<OperatorBase*>();
      for (const auto& idx : subgraph_indices[it.first])
        subgraph[it.first].push_back(cached_ops_[idx]);
    }
    for (auto* op : cached_ops_) {
      op->set_subgraph(subgraph);
    }
  }
}

bool Graph::Run(int stream, const string& include, const string& exclude) {
  unique_ptr<std::regex> regex_incl, regex_excl;
  if (!include.empty()) regex_incl.reset(new std::regex(include));
  if (!exclude.empty()) regex_excl.reset(new std::regex(exclude));
  LOG(DEBUG) << "Run Graph: " << name();
  for (auto* op : cached_ops_) {
    if (regex_incl && !regex_match(op->type(), *regex_incl)) continue;
    if (regex_excl && regex_match(op->type(), *regex_excl)) continue;
    op->SwitchToPhase(phase());
    LOG(DEBUG) << "Run Op: " << op->name();
    op->Run(stream);
    LOG(DEBUG) << "Finish Op: " << op->name();
  }
  return true;
}

GraphBase* NewGraph(const GraphDef& def, Workspace* ws) {
  if (!def.has_graph_type() || def.graph_type().empty()) {
    return new Graph(def, ws); // Sequential scheduler
  }
  return GraphRegistry()->Create(def.graph_type(), def, ws);
}

/* Graph Registry */

DEFINE_REGISTRY(GraphRegistry, GraphBase, const GraphDef&, Workspace*);

} // namespace dragon
