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

#define WORKSPACE_MIN_BUFFER_SIZE 3
#define WORKSPACE_MAX_BUFFER_SIZE 3

class Workspace{
 public:
    typedef Map<string, unique_ptr<Tensor> > TensorMap;
    typedef Map<string, unique_ptr<mutex> > LockMap;
    typedef Map<string, unique_ptr<GraphBase> > GraphMap;
    typedef Map<string, TensorFiller> FillerMap;
    typedef Map<string, string> RenameMap;

    Workspace(): root_folder_(".") { init(); }
    Workspace(string root_folder) : root_folder_(root_folder) { init(); }

    void init() { 
        CreateTensor("ignore"); 
        for (int i = 0; i < WORKSPACE_MIN_BUFFER_SIZE; i++) CreateBuffer();
    }

    /******************** Tensor ********************/

    inline string GetTensorName(const string& name) {
        if (rename_map_.count(name)) return rename_map_[name];
        else return name;
    }

    inline bool HasTensor(const string& name) {
        string query = GetTensorName(name);
        return tensor_map_.count(query) > 0; 
    }

    inline Tensor* CreateTensor(const string& name) {
        string query = GetTensorName(name);
        if (!HasTensor(query))
            tensor_map_[query] = unique_ptr<Tensor>(new Tensor(query));
        return tensor_map_[query].get();
    }

    inline Tensor* GetTensor(const string& name) {
        string query = GetTensorName(name);
        CHECK(HasTensor(query))
            << "Tensor(" << name << ") does not exist.";
        return tensor_map_[query].get();
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
        CHECK(HasTensor(name)) << "\nTensor(" << name << ") does not "
                               << "belong to workspace, could not release it.";
        string query = GetTensorName(name);
        tensor_map_[query]->Reset();
    }

    inline vector<string> GetTensors() {
        vector<string> names;
        for (auto& it : tensor_map_) names.push_back(it.first);
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

    /******************** Buffer ********************/

    inline Tensor* CreateBuffer() {
        int buffer_idx = 1;
        string name;
        while (1) {
            name = "_t_buffer_" + dragon_cast<string, int>(buffer_idx++);
            if (!HasTensor(name)) break;
        }
        buffer_stack_.push(name);
        return CreateTensor(name);
    }

    inline Tensor* GetBuffer() {
        if (!buffer_stack_.empty()) {
            string name = buffer_stack_.top();
            buffer_stack_.pop();
            return GetTensor(name);
        }
        LOG(FATAL) << "buffers are not enough, add more if necessary.";
        return nullptr;
    }

    inline void ReleaseBuffer(Tensor* tensor, bool force_release=false) {
        //  release directly
        if (buffer_stack_.size() >= WORKSPACE_MAX_BUFFER_SIZE || force_release) {
            ReleaseTensor(tensor->name());
        } else {    //  recover as a available buffer
            buffer_stack_.push(tensor->name());
        }
    }

    /******************** Graph ********************/

    GraphBase* CreateGraph(const GraphDef& graph_def);
    inline bool RunGraph(const string& graph_name, 
        const string& include, const string& exclude) {
        if (!graph_map_.count(graph_name)) {
            LOG(ERROR) << "Graph(" << graph_name << ") does not exist.";
            return false;
        }
        return graph_map_[graph_name]->Run(include, exclude);
    }

    inline vector<string> GetGraphs() {
        vector<string> names;
        for (auto& it : graph_map_) names.push_back(it.first);
        return names;
    }

    /******************** Utility ********************/
    
    inline const string& GetRootFolder() const { return root_folder_; }

    inline void CreateRename(const string& old_tensor,
                             const string& new_tensor) {
        rename_map_[old_tensor] = new_tensor;
    }

 private:
    TensorMap tensor_map_;
    LockMap lock_map_;
    GraphMap graph_map_;
    FillerMap filler_map_;
    RenameMap rename_map_;
    string root_folder_;
    stack<string> buffer_stack_;
};

}    // namespace dragon

#endif    // DRAGON_CORE_WORKSPACE_H_