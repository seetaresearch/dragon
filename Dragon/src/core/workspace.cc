#include "core/operator.h"
#include "core/graph.h"
#include "core/workspace.h"

namespace dragon {

/*! Create some internal tensors */

void Workspace::InitWorkspace() {
    CreateTensor("ignore");
    Tensor* head = CreateTensor(
        "/opt/mirror_stage/head");
    head->Reshape({ WORKSPACE_MAX_CORRUPTED_SIZE });
    Tensor* recompute_flag = CreateTensor(
        "/opt/mirror_stage/recompute_flag");
    recompute_flag->Reshape({ 1 });
    recompute_flag->mutable_data<bool, CPUContext>()[0] = false;
    for (int i = 0; i < WORKSPACE_MAX_CORRUPTED_SIZE; i++) {
        string name = "/opt/mirror_stage/buffer_" + std::to_string(i);
        Tensor* buffer = CreateTensor(name);
        head->mutable_data<string, CPUContext>()[i] = "";
    }
}

/*! Move a external workspace into this workspace */

Workspace* Workspace::Move(Workspace* ws) {
    CHECK(ws) << "The given Workspace is invalid.";
    if (workspace_map_.count(ws->name()))
        return workspace_map_[ws->name()];
    return workspace_map_[ws->name()] = ws;
}

/*! Destory all the tensors */

void Workspace::Clear() {
    // Clear tensors, then re-initialization
    for (auto& kv : tensor_map_) kv.second->Reset();
    InitWorkspace();
}

/*! Query the real name of specified tensor */

string Workspace::GetTensorName(const string& name) const {
    const auto& it = tensor_proxy_map_.find(name);
    if (it != tensor_proxy_map_.end()) return it->second;
    return name;
}

/*! Try to serach the specified tensor in this workspace */

Tensor* Workspace::TryGetTensor(
    const string&           name,
    bool                    use_remote) const {
    // Check the proxy of this tensor firstly
    string query = GetTensorName(name);

    // Search the local workspace
    const auto& it = tensor_map_.find(query);
    if (it != tensor_map_.end()) return it->second.get();

    if (use_remote) {
        // Search the remote workspaces
        for (auto& it : workspace_map_) {
            if (it.second->HasTensor(query))
                return it.second->GetTensor(query);
        }
    }
    return nullptr;
}

/*! Create the specified tensor */

Tensor* Workspace::CreateTensor(const string& name) {
    Tensor* tensor = TryGetTensor(name);
    if (!tensor) {
        tensor_map_[name] = unique_ptr<Tensor>(new Tensor(name));
        return tensor_map_[name].get();
    }
    return tensor;
}

/*! Return the specified tensor */

Tensor* Workspace::GetTensor(
    const string&               name,
    bool                        use_remote) const {
    Tensor* tensor = TryGetTensor(name, use_remote);
    CHECK(tensor) << "\nTensor(" << name << ") does not exist "
        << "in current workspace or sub-workspace.";
    return tensor;
}

/*! Reset the specified tensor */

void Workspace::ResetTensor(const string& name) {
    Tensor* tensor = TryGetTensor(name, false);
    CHECK(tensor) << "\nTensor(" << name << ") does not "
        << "belong to current workspace, could not be reset.";
    tensor->Reset();
}

/*! Return all the stored tensor names */

vector<string> Workspace::GetTensors() const {
    vector<string> locals;
    // Search the local workspace
    for (const auto& it : tensor_map_)
        locals.push_back(it.first);

    // Serach the remote workspaces
    for (const auto& it : workspace_map_) {
        vector<string> remotes = it.second->GetTensors();
        locals.insert(locals.end(), remotes.begin(), remotes.end());
    }
    return locals;
}

/*! Whether the specified filler is in this workspace */

bool Workspace::HasFiller(
    const string&               name,
    bool                        use_remote) const {
    // Search the local workspace
    bool result = tensor_filler_map_.count(name) > 0;
    if (!use_remote) return result;

    // Search the remote workspaces
    for (auto& it : workspace_map_)
        result |= it.second->HasFiller(name);
    return result;
}

/*! Create the specified filler */
void Workspace::CreateFiller(
    const TensorFillerProto     filler) {
    CHECK_GT(filler.tensor().size(), 0)
        << "\nTensor with an empty name can not be filled.";
    if (HasFiller(filler.tensor())) return;
    tensor_filler_map_[filler.tensor()] = filler;
}

/*! Return the specified filler */

const TensorFillerProto* Workspace::GetFiller(
    const string&               name) const {
    // Search the local workspace
    const auto& it = tensor_filler_map_.find(name);
    if (it != tensor_filler_map_.end()) return &it->second;

    // Search the remote workspaces
    for (const auto& it : workspace_map_) {
        if (it.second->HasFiller(name))
            return it.second->GetFiller(name);
    }
    return nullptr;
}

/*! Creathe a persistent operator in this workspace */

void Workspace::CreatePersistentOp(const OperatorDef& def) {
    string persistent_key;
    for (auto& arg : def.arg())
        if (arg.name() == "persistent_key")
            persistent_key = arg.s();
    CHECK(persistent_key.size() > 0)
        << "\nGot empty persistent key.";
    if (!operator_map_.count(persistent_key)) {
        for (auto& input : def.input()) CreateTensor(input);
        operator_map_[persistent_key] = unique_ptr<OperatorBase>(
            CreateOperator(def, this));
    }
}

/*! Run the specified persistent operator */

void Workspace::RunPersistentOp(
    const string&               key,
    const string&               anchor,
    const vector<string>&       inputs,
    const vector<string>&       outputs) {
    const auto& it = operator_map_.find(key);
    CHECK(it != operator_map_.end())
        << "\nPersistentOp(" << key << ") does not exist.";
    it->second->MutableOp(inputs, outputs, anchor);
    it->second->Run();
}

/*! Try to run the operator in a adaptive mode */

void Workspace::RunOperator(const OperatorDef& def) {
    string persistent_key;
    for (auto& arg : def.arg()) {
        if (arg.name() == "persistent_key")
            persistent_key = arg.s();
    }
    if (persistent_key.empty()) {
        // Run op in the "ONCE" mode
        unique_ptr<OperatorBase> op(CreateOperator(def, this));
        op->Run();
    } else {
        // Run op in the "PERSISTENT" mode
        if (!operator_map_.count(persistent_key))
            operator_map_[persistent_key] = unique_ptr<OperatorBase>(
                CreateOperator(def, this));
        else operator_map_[persistent_key]->MutableOp(def);
        operator_map_[persistent_key]->Run();
    }
}

/*! Create a Graph in this workspace */

GraphBase* Workspace::CreateGraph(const GraphDef& def) {
    // A unique graph name is required
    CHECK(def.has_name())
        << "\nThe name of given GraphDef should not be empty.";
    auto unique_name = GetDummyName(def.name(), "", "Graph", false);

    LOG(DEBUG) << "Create Graph: " << unique_name 
               << "(" << def.name() << ")";

    GraphDef mutable_def(def);
    mutable_def.set_name(unique_name);

    graph_map_[unique_name] = unique_ptr<GraphBase>(
        NewGraph(mutable_def, this));

    return graph_map_[unique_name].get();
}

/*! Run the specifed graph by name and rules */

void Workspace::RunGraph(
    const string&               graph_name,
    const string&               include,
    const string&               exclude,
    const int                   stream_id) {
    if (!graph_map_.count(graph_name))
        LOG(FATAL) << "Graph(" << graph_name
        << ") does not exist.";
    graph_map_[graph_name]->Run(include, exclude, stream_id);
}

/*! Return all the stored graph names */

vector<string> Workspace::GetGraphs() const {
    vector<string> names;
    for (const auto& it : graph_map_) {
        names.push_back(it.first);
    } return names;
}

/* Set a proxy name for the tensor */

bool Workspace::SetTensorProxy(
    const string&               key,
    const string&               proxy) {
    if (key == proxy) return false;
    if (tensor_proxy_map_.count(key) > 0)
        return tensor_proxy_map_[key] == proxy;
    tensor_proxy_map_[key] = proxy;
    return true;
}

/* Return a unique dummy name within this workspace */

string Workspace::GetDummyName(
    const string&               base_name,
    const string&               suffix,
    const string&               domain,
    const bool                  zero_based) {
    string required_name = base_name + suffix;
    if (dummy_name_map_.count(domain) == 0) {
        dummy_name_map_[domain] = Map<string, int64_t>();
    }
    auto& map_this_domain = dummy_name_map_[domain];
    int64_t index = map_this_domain[required_name]++;
    return index ? base_name + "_" +
        std::to_string(index) + suffix :
        zero_based ? required_name :
        base_name + "_" + std::to_string(
            map_this_domain[required_name]++) + suffix;
}

}  // namespace dragon