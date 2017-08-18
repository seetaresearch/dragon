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

class Workspace{
 public:
    typedef Map<string, unique_ptr<Tensor> > TensorMap;
    typedef Map<string, stack<string> > BufferMap;
    typedef Map<string, unique_ptr<mutex> > LockMap;
    typedef Map<string, unique_ptr<GraphBase> > GraphMap;
    typedef Map<string, TensorFiller> FillerMap;
    typedef Map<string, string> RenameMap;
    typedef Map<string, vector<OperatorBase*> > RecomputeMap;

    Workspace(): root_folder_(".") { init(); }
    Workspace(string root_folder) : root_folder_(root_folder) { init(); }
    ~Workspace();

    void init() { 
        CreateTensor("ignore"); 
        CreateBuffer("Common", WORKSPACE_COMMON_BUFFER_SIZE);
        CreateBuffer("Grad", WORKSPACE_GRAD_BUFFER_SIZE);
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

    inline void CreateBuffer(string category, int num) {
        CHECK(!buffer_map_.count(category));
        buffer_map_[category] = stack<string>();
        for (int i = 1; i <= num; i++) {
            string name = "_t_" + category + "_buffer_" + dragon_cast<string, int>(i);
            buffer_map_[category].push(name);
            CreateTensor(name);
        }
    }

    inline Tensor* GetBuffer(string category = "Common") {
        if (!buffer_map_[category].empty()) {
            string name = buffer_map_[category].top();
            buffer_map_[category].pop();
            return GetTensor(name);
        }
        LOG(FATAL) << "buffers of [" << category << "] "
                   << "are not enough, add more if necessary.";
        return nullptr;
    }

    inline void ReleaseBuffer(Tensor* tensor, 
                              string category = "Common",
                              bool enforce = false) {
        static Map<string, int> limits = {
            { "Common", WORKSPACE_COMMON_BUFFER_SIZE },
            { "Grad", WORKSPACE_GRAD_BUFFER_SIZE }};
        if (buffer_map_[category].size() >= limits[category] || enforce) {
            //  release directly
            ReleaseTensor(tensor->name());
        } else {    
            //  recover as a available buffer
            buffer_map_[category].push(tensor->name());
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

    inline void AddRecompute(const string& tensor, OperatorBase* op) {
        if (!recompute_map_.count(tensor)) {
            recompute_map_[tensor] = vector<OperatorBase*>();
        }
        recompute_map_[tensor].push_back(op);
    }

    inline vector<OperatorBase*> GetRecompute(const string& tensor) {
        if (recompute_map_.count(tensor)) {
            return recompute_map_[tensor];
        } else {
            return vector<OperatorBase*>();
        }
    }

 private:
    TensorMap tensor_map_;
    BufferMap buffer_map_;
    LockMap lock_map_;
    GraphMap graph_map_;
    FillerMap filler_map_;
    RenameMap rename_map_;   
    RecomputeMap recompute_map_;
    string root_folder_;
};

}    // namespace dragon

#endif    // DRAGON_CORE_WORKSPACE_H_