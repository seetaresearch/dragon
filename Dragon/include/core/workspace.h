// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_CORE_WORKSPACE_H_
#define DRAGON_CORE_WORKSPACE_H_

#include "core/common.h"
#include "core/graph.h"
#include "utils/string.h"

namespace dragon {

#define WORKSPACE_COMMON_BUFFER_SIZE 2
#define WORKSPACE_GRAD_BUFFER_SIZE 1
#define WORKSPACE_MAX_CORRUPTED_SIZE 2

class Workspace {
 public:
    typedef Map<string, Workspace*> WorkspaceMap;
    typedef Map<string, unique_ptr<Tensor> > TensorMap;
    typedef Map<string, stack<string> > BufferMap;
    typedef Map<string, unique_ptr<mutex> > LockMap;
    typedef Map<string, unique_ptr<GraphBase> > GraphMap;
    typedef Map<string, TensorFiller> FillerMap;
    typedef Map<string, string> RenameMap;
    typedef Map<string, string> AvatarMap;

    Workspace(const string& name) : name_(name) { Init(); }
    ~Workspace();

    void Init() { 
        CreateTensor("ignore"); 
        CreateBuffer("Common", WORKSPACE_COMMON_BUFFER_SIZE);
        CreateBuffer("Grad", WORKSPACE_GRAD_BUFFER_SIZE);
    }

    inline const string& name() { return name_; }

    /******************** Workspace ********************/

    inline Workspace* MoveWorkspace(Workspace* ws) {
        CHECK(ws) << "The given Workspace is invalid.";
        if (workspace_map_.count(ws->name()))
            return workspace_map_[ws->name()];
        return workspace_map_[ws->name()] = ws;
    }

    /******************** Tensor ********************/

    inline string GetTensorName(const string& name) {
        if (rename_map_.count(name) > 0) {
            return rename_map_[name];
        } else { return name; }
    }

    bool HasTensor(const string& name, bool use_remote=true) {
        //  search local workspace
        string query = GetTensorName(name);
        bool result = tensor_map_.count(query) > 0;
        if (!use_remote) return result;

        //  search remote workspace
        for (auto& it : workspace_map_)
            result |= it.second->HasTensor(query);
        return result;
    }

    inline Tensor* CreateTensor(const string& name) {
        string query = GetTensorName(name);
        if (!HasTensor(query))
            tensor_map_[query] = unique_ptr<Tensor>(new Tensor(query));
        return tensor_map_[query].get();
    }

    Tensor* GetTensor(const string& name, bool use_remote=true) {
        string query = GetTensorName(name);
        //  search local workspace
        if (tensor_map_.count(query) > 0) 
            return tensor_map_[query].get();
        if (use_remote) {
            //  search remote workspace
            for (auto& it : workspace_map_) {
                if (it.second->HasTensor(query))
                    return it.second->GetTensor(query);
            }
        }
        LOG(FATAL) << "Tensor(" << name << ") does not exist "
                   << "in current workspace and it's sub-workspace.";
        return nullptr;
    }

    inline void LockTensor(const string& name) {
        string query = GetTensorName(name);
        if (!lock_map_.count(query))
            lock_map_[query] = unique_ptr<mutex>(new mutex);
        lock_map_[query]->lock();
    }

    inline void UnlockTensor(const string& name) {
        string query = GetTensorName(name);
        if (!lock_map_.count(query))
            lock_map_[query] = unique_ptr<mutex>(new mutex);
        lock_map_[query]->unlock();
    }

    inline void ReleaseTensor(const string& name) {
        CHECK(HasTensor(name, false)) 
            << "\nTensor(" << name << ") does not "
            << "belong to current workspace, could not release it.";
        string query = GetTensorName(name);
        tensor_map_[query]->Reset();
    }

    vector<string> GetTensors() {
        vector<string> names;
        //  search local workspace
        for (auto& it : tensor_map_) 
            names.push_back(it.first);
        //  serach remote workspace
        for (auto& it : workspace_map_) {
            vector<string> sub_names = it.second->GetTensors();
            names.insert(names.end(), sub_names.begin(), sub_names.end());
        }
        return names;
    }

    /******************** Filler ********************/

    inline void CreateFiller(const TensorFiller filler) {
        CHECK_GT(filler.tensor().size(), 0) 
            << "Tensor without a valid name can not be filled.";
        if (filler_map_.count(filler.tensor())) return;
        filler_map_[filler.tensor()] = filler;
    }

    inline const TensorFiller* GetFiller(const string& name) {
        if (filler_map_.count(name) > 0) return &filler_map_[name];
        else return nullptr;
    }

    /******************** Avatar ********************/

    inline void CreateAvatar(Tensor* orig, Tensor* avatar) {
        CHECK(tensor_map_.count(orig->name()) > 0)
            << "\nFailed to create avatar for Tensor(" << orig->name() << ")."
            << "\nAs it has not been registered in the current workspace.";
        avatar_map_[orig->name()] = avatar->name();
    }

    inline Tensor* SearchAvatar(Tensor* orig) {
        if (avatar_map_.count(orig->name()) > 0) 
            return tensor_map_[avatar_map_[orig->name()]].get();
        return orig;
    }

    /******************** Buffer ********************/

    void CreateBuffer(string category, int num) {
        CHECK(!buffer_map_.count(category));
        buffer_map_[category] = stack<string>();
        for (int i = 1; i <= num; i++) {
            string name = "/share/buffer/" + category + "_" + dragon_cast<string, int>(i);
            buffer_map_[category].push(name);
            CreateTensor(name);
        }
    }

    inline Tensor* GetBuffer(string category = "Common") {
        if (!buffer_map_[category].empty()) {
            string name = buffer_map_[category].top();
            buffer_map_[category].pop();
            return tensor_map_[name].get();
        }
        LOG(FATAL) << "Buffers of [" << category << "] "
                   << "are not enough, add more if necessary.";
        return nullptr;
    }

    void ReleaseBuffer(Tensor* tensor, 
                       string category = "Common",
                       bool enforce = false) {
        static Map<string, int> limits = {
                { "Common", WORKSPACE_COMMON_BUFFER_SIZE },
                { "Grad", WORKSPACE_GRAD_BUFFER_SIZE }
        };
        if (buffer_map_[category].size() >= limits[category] || enforce) {
            ReleaseTensor(tensor->name());
            if (buffer_map_[category].empty())
                buffer_map_[category].push(tensor->name());
        } else {
            buffer_map_[category].push(tensor->name());
        }
    }

    /******************** Graph ********************/

    GraphBase* CreateGraph(const GraphDef& meta_graph);

    bool RunGraph(const string& graph_name,
                         const string& include,
                         const string& exclude) {
        if (!graph_map_.count(graph_name)) {
            LOG(ERROR) << "Graph(" << graph_name << ") does not exist.";
            return false;
        }
        return graph_map_[graph_name]->Run(include, exclude);
    }

    vector<string> GetGraphs() {
        vector<string> names;
        for (auto& it : graph_map_) names.push_back(it.first);
        return names;
    }

    /******************** Utility ********************/

    inline void CreateRename(const string& old_tensor,
                             const string& new_tensor) {
        rename_map_[old_tensor] = new_tensor;
    }

 private:
    string name_;
    WorkspaceMap workspace_map_;
    TensorMap tensor_map_;
    BufferMap buffer_map_;
    LockMap lock_map_;
    GraphMap graph_map_;
    FillerMap filler_map_;
    RenameMap rename_map_;
    AvatarMap avatar_map_;
};

}    // namespace dragon

#endif    // DRAGON_CORE_WORKSPACE_H_