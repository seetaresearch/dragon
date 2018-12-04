/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_CORE_WORKSPACE_H_
#define DRAGON_CORE_WORKSPACE_H_

#include "core/common.h"
#include "core/graph.h"
#include "utils/string.h"

namespace dragon {

#define WORKSPACE_MAX_CORRUPTED_SIZE 2

class Workspace {
 public:
    typedef Map<string, Workspace*> WorkspaceMap;
    typedef Map<string, unique_ptr<Tensor> > TensorMap;
    typedef Map<string, unique_ptr<OperatorBase> > OperatorMap;
    typedef Map<string, unique_ptr<GraphBase> > GraphMap;
    typedef Map<string, TensorFiller> FillerMap;
    typedef Map<string, string> ProxyMap;

    Workspace(const string& name) : name_(name) { InitWorkspace(); }

    ~Workspace();

    inline const string& name() { return name_; }

    /******************** Workspace ********************/

    inline void InitWorkspace() {
        CreateTensor("ignore");
        Tensor* head = CreateTensor(
            "/opt/mirror_stage/head");
        head->Reshape({ WORKSPACE_MAX_CORRUPTED_SIZE });
        Tensor* recompute_flag = CreateTensor(
            "/opt/mirror_stage/recompute_flag");
        recompute_flag->Reshape({ 1 });
        recompute_flag->mutable_data<bool, CPUContext>()[0] = false;
        for (int i = 0; i < WORKSPACE_MAX_CORRUPTED_SIZE; i++) {
            string name = "/opt/mirror_stage/buffer_"
                + std::to_string(i);
            Tensor* buffer = CreateTensor(name);
            head->mutable_data<string, CPUContext>()[i] = "";
        }
    }

    inline Workspace* MoveWorkspace(Workspace* ws) {
        CHECK(ws) << "The given Workspace is invalid.";
        if (ws_map_.count(ws->name()))
            return ws_map_[ws->name()];
        return ws_map_[ws->name()] = ws;
    }

    inline void ClearWorkspace() {
        // Clear tensors, then re-initialization
        for (auto& kv : tensor_map_) kv.second->Reset();
        InitWorkspace();
    }

    /******************** Tensor ********************/

    inline string GetTensorName(const string& name) {
        if (proxy_map_.count(name) > 0) {
            return proxy_map_[name];
        } else { return name; }
    }

    inline Tensor* TryGetTensor(
        const string&           name,
        bool                    use_remote = true) {
        string query = GetTensorName(name);
        // Search local workspace
        if (tensor_map_.count(query) > 0)
            return tensor_map_[query].get();
        if (use_remote) {
            // Search remote workspace
            for (auto& it : ws_map_) {
                if (it.second->HasTensor(query))
                    return it.second->GetTensor(query);
            }
        }
        return nullptr;
    }

    inline bool HasTensor(
        const string&           name,
        bool                    use_remote = true) {
        return TryGetTensor(name, use_remote) ? true : false;
    }

    inline Tensor* CreateTensor(const string& name) {
        Tensor* tensor = TryGetTensor(name);
        if (!tensor) {
            tensor_map_[name] = unique_ptr<Tensor>(new Tensor(name));
            return tensor_map_[name].get();
        }
        return tensor;
    }

    inline Tensor* GetTensor(
        const string&           name,
        bool                    use_remote = true) {
        Tensor* tensor = TryGetTensor(name, use_remote);
        CHECK(tensor) << "\nTensor(" << name << ") does not exist "
                      << "in current workspace or sub-workspace.";
        return tensor;
    }

    inline void ResetTensor(const string& name) {
        Tensor* tensor = TryGetTensor(name, false);
        CHECK(tensor) << "\nTensor(" << name << ") does not "
                      << "belong to current workspace, could not be reset.";
        tensor->Reset();
    }

    vector<string> GetTensors() {
        vector<string> names;
        // Search local workspace
        for (auto& it : tensor_map_)
            names.push_back(it.first);
        // Serach remote workspace
        for (auto& it : ws_map_) {
            vector<string> sub_names = it.second->GetTensors();
            names.insert(names.end(),
                sub_names.begin(), sub_names.end());
        }
        return names;
    }

    /******************** Filler ********************/

    inline bool HasFiller(
        const string&           name,
        bool                    use_remote = true) {
        // Search local workspace
        bool result = filler_map_.count(name) > 0;
        if (!use_remote) return result;

        // Search remote workspace
        for (auto& it : ws_map_)
            result |= it.second->HasFiller(name);
        return result;
    }

    inline void CreateFiller(
        const TensorFiller      filler) {
        CHECK_GT(filler.tensor().size(), 0)
            << "Tensor without a valid name can not be filled.";
        if (HasFiller(filler.tensor())) return;
        filler_map_[filler.tensor()] = filler;
    }

    inline const TensorFiller* GetFiller(
        const string&           name) {
        // Search local workspace
        if (filler_map_.count(name) > 0)
            return &filler_map_[name];

        // Search remote workspace
        for (auto& it : ws_map_) {
            if (it.second->HasFiller(name))
                return it.second->GetFiller(name);
        }
        return nullptr;
    }

    /******************** Cache ********************/

    template <class Context>
    inline vector<void*> caches(
        const vector<size_t>&   segments) {
        TIndex nbytes = 0;
        for (auto& segment : segments) nbytes += (TIndex)segment;
        Tensor* cache_t = CreateTensor("/share/cache");
        cache_t->Reshape({ nbytes });
        vector<void*> Bcaches(segments.size());
        Bcaches[0] = cache_t->template mutable_data<uint8_t, Context>();
        for (int i = 1; i < segments.size(); i++)
            Bcaches[i] = (uint8_t*)Bcaches[i - 1] + segments[i - 1];
        return Bcaches;
    }

    template <typename T, class Context>
    inline vector<T*> caches(
        const vector<TIndex>&   segments) {
        vector<size_t> Tsegments;
        for (auto& segment : segments)
            Tsegments.emplace_back(segment * sizeof(T));
        vector<void*> Bcaches = caches<Context>(Tsegments);
        vector<T*> Tcaches(segments.size());
        for (int i = 0; i < segments.size(); i++)
            Tcaches[i] = (T*)Bcaches[i];
        return Tcaches;
    }

    /******************** Operator ********************/

    inline void CreatePersistentOp(
        const OperatorDef& meta_op) {
        string persistent_key;
        for (auto& arg : meta_op.arg())
            if (arg.name() == "persistent_key")
                persistent_key = arg.s();
        CHECK(persistent_key.size() > 0)
            << "\nGot empty persistent key.";
        if (!op_map_.count(persistent_key)) {
            for (auto& input : meta_op.input()) CreateTensor(input);
            op_map_[persistent_key] = unique_ptr<OperatorBase>(
                CreateOperator(meta_op, this));
        }
    }

    inline void RunPersistentOp(
        const string&           key,
        const string&           anchor,
        const vector<string>&   inputs,
        const vector<string>&   outputs) {
        CHECK(op_map_.count(key) > 0)
            << "\nPersistentOp(" << key << ") does not exist.";
       op_map_[key]->MutableOp(inputs, outputs, anchor);
       op_map_[key]->Run();
    }

    void RunOperator(const OperatorDef& meta_op) {
        string persistent_key;
        for (auto& arg : meta_op.arg()) {
            if (arg.name() == "persistent_key")
                persistent_key = arg.s();
        }
        if (persistent_key.empty()) {
            // Run op in the "ONCE" mode
            unique_ptr<OperatorBase> op(CreateOperator(meta_op, this));
            op->Run();
        } else {
            // Run op in the "PERSISTENT" mode
            if (!op_map_.count(persistent_key))
                op_map_[persistent_key] = unique_ptr<OperatorBase>(
                    CreateOperator(meta_op, this));
            else op_map_[persistent_key]->MutableOp(meta_op);
            op_map_[persistent_key]->Run();
        }
    }

    /******************** Graph ********************/

    GraphBase* CreateGraph(const GraphDef& meta_graph);

    void RunGraph(
        const string&           graph_name,
        const string&           include,
        const string&           exclude,
        const int               stream_id = 1) {
        if (!graph_map_.count(graph_name))
            LOG(FATAL) << "Graph(" << graph_name
                       << ") does not exist.";
        graph_map_[graph_name]->Run(include, exclude, stream_id);
    }

    vector<string> GetGraphs() {
        vector<string> names;
        for (auto& it : graph_map_) names.push_back(it.first);
        return names;
    }

    /******************** Utility ********************/

    inline bool SetProxy(
        const string&           key,
        const string&           proxy) {
        if (key == proxy) return false;
        if (proxy_map_.count(key) > 0)
            return proxy_map_[key] == proxy;
        proxy_map_[key] = proxy;
        return true;
    }

 private:
    string name_;
    WorkspaceMap ws_map_;
    TensorMap tensor_map_;
    OperatorMap op_map_;
    GraphMap graph_map_;
    FillerMap filler_map_;
    ProxyMap proxy_map_;
};

}  // namespace dragon

#endif  // DRAGON_CORE_WORKSPACE_H_