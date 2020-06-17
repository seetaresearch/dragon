#include "dragon/core/workspace.h"
#include "dragon/core/graph.h"
#include "dragon/core/operator.h"

namespace dragon {

vector<string> Workspace::tensors() const {
  vector<string> locals;
  // Search the local workspace
  for (const auto& it : tensor_map_)
    locals.push_back(it.first);

  // Search the remote workspaces
  for (const auto& it : external_tensor_map_) {
    locals.push_back(it.first);
  }
  return locals;
}

vector<string> Workspace::graphs() const {
  vector<string> names;
  for (const auto& it : graph_map_) {
    names.push_back(it.first);
  }
  return names;
}

void Workspace::Initialize() {
  CreateTensor(""); // Empty placeholder
  CreateTensor("/share/flag/recomputing")
      ->Reshape({1})
      ->mutable_data<bool, CPUContext>()[0] = false;
}

void Workspace::Clear() {
  // Remove and Initialize again
  tensor_map_.clear();
  Initialize();
}

void Workspace::MergeFrom(Workspace* ws) {
  CHECK(ws) << "\nThe given Workspace is invalid.";
  for (const auto& it : ws->tensor_map_) {
    if (!it.first.empty() && !str::startswith(it.first, "/")) {
      external_tensor_map_[it.first] = it.second.get();
    }
  }
}

string Workspace::GetTensorName(const string& name) const {
  const auto& it = alias_active_map_.find(name);
  if (it != alias_active_map_.end()) return it->second;
  return name;
}

Tensor* Workspace::TryGetTensor(const string& name, bool use_remote) const {
  // Check the proxy of this tensor firstly
  string query = GetTensorName(name);

  // Search the local workspace
  const auto& it = tensor_map_.find(query);
  if (it != tensor_map_.end()) return it->second.get();

  if (use_remote) {
    // Search the remote workspaces
    const auto& it = external_tensor_map_.find(query);
    if (it != external_tensor_map_.end()) return it->second;
  }
  return nullptr;
}

Tensor* Workspace::CreateTensor(const string& name) {
  Tensor* tensor = TryGetTensor(name);
  if (!tensor) {
    tensor_map_[name] = unique_ptr<Tensor>(new Tensor(name));
    return tensor_map_[name].get();
  }
  return tensor;
}

Tensor* Workspace::GetTensor(const string& name, bool use_remote) const {
  Tensor* tensor = TryGetTensor(name, use_remote);
  CHECK(tensor) << "\nTensor(" << name << ") does not "
                << "exist in current workspace.";
  return tensor;
}

void Workspace::ResetTensor(const string& name) {
  Tensor* tensor = TryGetTensor(name, false);
  CHECK(tensor) << "\nTensor(" << name << ") does not "
                << "belong to current workspace.";
  tensor->Reset();
}

bool Workspace::HasFiller(const string& name) const {
  return tensor_filler_map_.count(name) > 0;
}

void Workspace::CreateFiller(const TensorFillerProto& filler) {
  CHECK_GT(filler.tensor().size(), 0)
      << "\nTensor with an empty name can not be filled.";
  if (HasFiller(filler.tensor())) return;
  tensor_filler_map_[filler.tensor()] = filler;
}

TensorFillerProto* Workspace::GetFiller(const string& name) {
  const auto& it = tensor_filler_map_.find(name);
  if (it != tensor_filler_map_.end()) return &it->second;
  return nullptr;
}

OperatorBase* Workspace::CreateOperator(const OperatorDef& def) {
  const auto& it = operator_map_.find(def.cache_key());
  if (it == operator_map_.end()) {
    auto* new_op = NewOperator(def, this);
    operator_map_[def.cache_key()] = unique_ptr<OperatorBase>(new_op);
    return new_op;
  }
  return it->second.get();
}

void Workspace::RunOperator(const OperatorDef& def) {
  if (def.has_cache_key()) {
    CreateOperator(def)->UpdateFrom(def)->Run(0);
  } else {
    unique_ptr<OperatorBase> op(NewOperator(def, this));
    op->Run(0);
  }
}

GraphBase* Workspace::CreateGraph(const GraphDef& def) {
  CHECK(def.has_name()) << "\nGraph name is missing.";
  auto name = GetDummyName(def.name(), "", "Graph", false);
  LOG(DEBUG) << "Create Graph: " << name << "(" << def.name() << ")";
  GraphDef _def(def);
  _def.set_name(name);
  graph_map_[name] = unique_ptr<GraphBase>(NewGraph(_def, this));
  return graph_map_[name].get();
}

void Workspace::RunGraph(
    const string& graph_name,
    const string& incl,
    const string& excl,
    int stream_id) {
  if (!graph_map_.count(graph_name)) {
    LOG(FATAL) << "Graph(" << graph_name << ") does not exist.";
  }
  graph_map_[graph_name]->Run(incl, excl, stream_id);
}

bool Workspace::ActivateAlias(const string& name, const string& alias) {
  bool status = alias_active_map_.count(alias) > 0;
  alias_active_map_[alias] = name;
  return status; // True if activated otherwise false
}

string Workspace::GetDummyName(
    const string& base_name,
    const string& suffix,
    const string& domain,
    bool zero_based) {
  string accepted_name;
  int64_t index;
  const auto required_name = base_name + suffix;
  auto& dmap = dummy_name_map_[domain];
  while (1) {
    index = dmap[required_name]++;
    accepted_name = index ? base_name + "_" + str::to(index) + suffix
                          : zero_based
            ? required_name
            : base_name + "_" + str::to(dmap[required_name]++) + suffix;
    if (external_tensor_map_.empty()) break;
    if (!HasTensor(accepted_name)) break;
  }
  return accepted_name;
}

} // namespace dragon
