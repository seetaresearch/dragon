#include <regex>

#include "dragon/core/graph.h"
#include "dragon/core/graph_optimizer.h"
#include "dragon/core/workspace.h"

namespace dragon {

GraphBase::GraphBase(const GraphDef& def, Workspace* ws)
    : def_(def), ws_(ws), name_(def.name()), phase_("TEST") {
  // Collect arguments.
  for (auto& arg : def_.arg()) {
    CHECK_GT(arg.name().size(), 0);
    CHECK_EQ(args_.count(arg.name()), 0);
    args_[arg.name()] = &arg;
    if (arg.name() == "phase") phase_ = arg.s();
  }
  // Check inputs.
  Set<string> outputs;
  for (const auto& op : def.op()) {
    for (const auto& input : op.input())
      CHECK(outputs.count(input) || ws_->HasTensor(input))
          << "\nInput " << input << " is not in the graph.";
    for (const auto& output : op.output()) {
      outputs.insert(output);
    }
  }
  // Check outputs.
  for (const auto& output : def.output()) {
    CHECK(outputs.count(output) || ws_->HasTensor(output))
        << "\nOutput " << output << " is not in the graph.";
  }
}

bool Graph::Create(const GraphDef& def) {
  this->optimized_def_ = def;
  bool has_device_option = def.has_device_option();
  for (int i = 0; i < def.op_size(); i++) {
    auto op_def(def.op(i));
    // Inherit device if not provided.
    if (!op_def.has_device_option() && has_device_option) {
      op_def.mutable_device_option()->CopyFrom(def.device_option());
    }
    LOG(DEBUG) << "Create: " << op_def.name() << " [" << op_def.type() << "]";
    ops_.push_back(OperatorBase::New(op_def, ws_));
    ops_.back()->set_output_aliases(output_aliases_);
  }
  return true;
}

Graph::Graph(const GraphDef& def, Workspace* ws) : GraphBase(def, ws) {
  // Apply the optimizations.
  GraphDef def_v2(def);
  GraphOptimizer optimizer(ws);
  Map<string, vec32_t> subgraph_indices;
  int opt = 1;
  if (args().count("optimization")) opt = arg("optimization").i();
  if (opt >= 2) optimizer.PlanInplace(def_v2, output_aliases_);
  if (opt >= 3) {
    if (phase() == "TRAIN") {
      def_v2 = optimizer.PlanCheckpoint(def_v2, subgraph_indices);
      if (args().count("grad_sources")) {
        GradientTape tape(def_v2);
        auto& grad_sources = args_["grad_sources"]->strings();
        tape.Optimize({grad_sources.begin(), grad_sources.end()});
        def_v2 = tape.def();
      }
    } else {
      def_v2 = optimizer.EliminateIntermediates(def_v2);
    }
  }
  // Create graph.
  Create(def_v2);
  // Create subgraphs.
  if (subgraph_indices.size() > 0) {
    Map<string, vector<OperatorBase*>> subgraph;
    for (const auto& it : subgraph_indices) {
      subgraph[it.first] = vector<OperatorBase*>();
      for (auto op_idx : subgraph_indices[it.first])
        subgraph[it.first].push_back(ops_[op_idx]);
    }
    for (auto* op : ops_) {
      op->set_subgraph(subgraph);
    }
  }
}

bool Graph::Run(int stream, const string& include, const string& exclude) {
  unique_ptr<std::regex> regex_incl, regex_excl;
  if (!include.empty()) regex_incl.reset(new std::regex(include));
  if (!exclude.empty()) regex_excl.reset(new std::regex(exclude));
  LOG(DEBUG) << "Run: " << name();
  for (auto* op : ops_) {
    if (regex_incl && !regex_match(op->type(), *regex_incl)) continue;
    if (regex_excl && regex_match(op->type(), *regex_excl)) continue;
    op->SwitchToPhase(phase());
    LOG(DEBUG) << "Run: " << op->name();
    op->Run(stream);
    LOG(DEBUG) << "Finish: " << op->name();
  }
  LOG(DEBUG) << "Finish: " << name();
  return true;
}

GraphBase* GraphBase::New(const GraphDef& def, Workspace* ws) {
  if (!def.has_type() || def.type().empty()) {
    // Sequential scheduler.
    return new Graph(def, ws);
  }
  return GraphRegistry()->Create(def.type(), def, ws);
}

/* Graph Registry */
DEFINE_REGISTRY(GraphRegistry, GraphBase, const GraphDef&, Workspace*);

} // namespace dragon
