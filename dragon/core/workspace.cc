#include "dragon/core/workspace.h"
#include "dragon/core/graph.h"
#include "dragon/core/operator.h"

namespace dragon {

Workspace::Workspace(const string& name) : name_(name) {
  CreateTensor(""); // Empty placeholder
}

void Workspace::MergeFrom(Workspace* other) {
  if (other != nullptr) {
    // Add the external tensors
    for (const auto& it : other->tensors_) {
      if (!it.first.empty() && !str::startswith(it.first, "/")) {
        external_tensors_[it.first] = it.second.get();
      }
    }
    // Recount the unique index to avoid duplicate names
    for (const auto& i : other->scope_counters_) {
      auto& counters = scope_counters_[i.first];
      for (const auto& j : i.second) {
        counters[j.first] = std::max(counters[j.first], j.second);
      }
    }
  }
}

void Workspace::Clear() {
  // Following resources usually take large memory blob.
  // It's necessary to clear them manually if workspace referenced
  // by the frontend GC circularly.
  graphs_.clear();
  operators_.clear();
  for (const auto& iter : tensors_) {
    // The tensor pointer may be referenced by the frontend.
    // Reset memory only to avoid the dangling pointer.
    iter.second->Reset();
  }
}

Tensor* Workspace::TryGetTensor(const string& name, bool external) const {
  // Check the alias.
  const auto& alias_iter = aliases_.find(name);
  auto name_v2 = alias_iter != aliases_.end() ? alias_iter->second : name;
  // Search this workspace.
  const auto& iter = tensors_.find(name_v2);
  if (iter != tensors_.end()) return iter->second.get();
  if (external) {
    // Search external workspaces.
    const auto& iter = external_tensors_.find(name_v2);
    if (iter != external_tensors_.end()) return iter->second;
  }
  return nullptr;
}

Tensor* Workspace::CreateTensor(const string& name) {
  auto* tensor_ptr = TryGetTensor(name);
  // Create only if name not existed.
  if (tensor_ptr == nullptr) {
    tensor_ptr = new Tensor(name);
    tensors_[name] = unique_ptr<Tensor>(tensor_ptr);
  }
  return tensor_ptr;
}

Tensor* Workspace::GetTensor(const string& name, bool external) const {
  auto* tensor_ptr = TryGetTensor(name, external);
  CHECK(tensor_ptr) << "\nTensor '" << name << "' is not in workspace.";
  return tensor_ptr;
}

void Workspace::RunOperator(const OperatorDef& def) {
  string cache_key;
  OperatorBase* op_ptr = nullptr;
  if (!def.arg().empty()) {
    const auto& arg = *(def.arg().end() - 1);
    if (arg.name() == "cache_key") cache_key = arg.s();
  }
  if (cache_key.empty()) {
    op_ptr = OperatorBase::New(def, this);
    op_ptr->Run();
    delete op_ptr;
  } else {
    const auto& iter = operators_.find(cache_key);
    if (iter == operators_.end()) {
      op_ptr = OperatorBase::New(def, this);
      operators_[cache_key] = unique_ptr<OperatorBase>(op_ptr);
    } else {
      op_ptr = iter->second.get();
    }
    op_ptr->DeriveFrom(def)->Run();
  }
}

GraphBase* Workspace::CreateGraph(const GraphDef& def) {
  CHECK(def.has_name()) << "\nExcepted non-empty graph name.";
  GraphDef def_v2(def); // Copy to set an unique name
  def_v2.set_name(UniqueName(def.name(), "", "Graph", false));
  LOG(DEBUG) << "Create: " << def_v2.name();
  auto* graph_ptr = GraphBase::New(def_v2, this);
  graphs_[def_v2.name()] = unique_ptr<GraphBase>(graph_ptr);
  return graph_ptr;
}

void Workspace::RunGraph(
    const string& name,
    const string& include,
    const string& exclude,
    int stream) {
  CHECK(graphs_.count(name))
      << "\nGraph " << name << " is not in current workspace.";
  graphs_[name]->Run(stream, include, exclude);
}

string Workspace::UniqueName(
    const string& name,
    const string& suffix,
    const string& scope,
    bool zero_based) {
  auto& counters = scope_counters_[scope];
  auto target_name = name + suffix;
  auto index = counters[target_name]++;
  if (index > 0) return name + "_" + str::to(index) + suffix;
  if (zero_based) return target_name;
  return name + "_" + str::to(counters[target_name]++) + suffix;
}

vector<string> Workspace::tensors(bool external) const {
  vector<string> names;
  for (const auto& iter : tensors_) {
    names.emplace_back(iter.first);
  }
  if (external) {
    for (const auto& iter : external_tensors_) {
      names.emplace_back(iter.first);
    }
  }
  return names;
}

vector<string> Workspace::graphs() const {
  vector<string> names;
  for (const auto& iter : graphs_) {
    names.emplace_back(iter.first);
  }
  return names;
}

} // namespace dragon
