#include <regex>

#include "dragon/core/graph.h"
#include "dragon/core/graph_optimizer.h"
#include "dragon/core/workspace.h"

namespace dragon {

GraphBase::GraphBase(const GraphDef& def, Workspace* ws)
    : def_(def), workspace_(ws), name_(def.name()), phase_("TEST") {
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
      CHECK(outputs.count(input) || workspace_->HasTensor(input))
          << "\nInput " << input << " is not in the graph.";
    for (const auto& output : op.output()) {
      outputs.insert(output);
    }
  }
  // Check outputs.
  for (const auto& output : def.output()) {
    CHECK(outputs.count(output) || workspace_->HasTensor(output))
        << "\nOutput " << output << " is not in the graph.";
  }
}

bool Graph::Create(const GraphDef& def) {
  this->optimized_def_ = def;
  bool has_device_option = def.has_device_option();
  for (int i = 0; i < def.op_size(); i++) {
    auto op_def(def.op(i));
    if (!op_def.has_device_option() && has_device_option) {
      op_def.mutable_device_option()->CopyFrom(def.device_option());
    }
    LOG(DEBUG) << "Create: " << op_def.name() << " [" << op_def.type() << "]";
    auto* op_ptr = OperatorBase::New(op_def, workspace_);
    operators_.push_back(unique_ptr<OperatorBase>(op_ptr));
    operators_.back()->set_outputs_from(outputs_from_);
  }
  return true;
}

Graph::Graph(const GraphDef& def, Workspace* ws) : GraphBase(def, ws) {
  // Apply optimizations.
  GraphDef optimized_def(def);
  GraphOptimizer optimizer;
  int opt_level = 1;
  if (args().count("optimization")) opt_level = arg("optimization").i();
  if (opt_level >= 2) optimizer.PlanInplace(def, outputs_from_);
  if (opt_level >= 3) {
    if (phase() == "TRAIN") {
      if (args().count("grad_sources")) {
        GradientTape tape(def);
        const auto& sources = arg("grad_sources").strings();
        tape.Optimize({sources.begin(), sources.end()});
        optimized_def = tape.def();
      }
    } else {
      optimized_def = optimizer.EliminateIntermediates(def);
    }
  }
  // Create graph.
  Create(optimized_def);
}

bool Graph::Run(int stream) {
  LOG(DEBUG) << "Run: " << name();
  for (size_t op_index = 0; op_index < operators_.size(); ++op_index) {
    auto* op_ptr = operators_[op_index].get();
    op_ptr->SwitchToPhase(phase());
    LOG(DEBUG) << "Run: " << op_ptr->name();
    op_ptr->Run(stream, op_index == operators_.size() - 1);
    LOG(DEBUG) << "Finish: " << op_ptr->name();
  }
  LOG(DEBUG) << "Finish: " << name();
  return true;
}

GraphBase* GraphBase::New(const GraphDef& def, Workspace* ws) {
  if (!def.has_type() || def.type().empty()) {
    return new Graph(def, ws); // Sequential scheduler.
  }
  return GraphRegistry()->Create(def.type(), def, ws);
}

/* Graph Registry */
DEFINE_REGISTRY(GraphRegistry, GraphBase, const GraphDef&, Workspace*);

} // namespace dragon
