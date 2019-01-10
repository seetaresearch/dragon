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
    typedef Map<string, Map<string, int64_t> > DummyNameMap;

    typedef Map<string, unique_ptr<Tensor> > TensorMap;
    typedef Map<string, string> TensorProxyMap;
    typedef Map<string, TensorFillerProto> TensorFillerMap;

    typedef Map<string, unique_ptr<OperatorBase> > OperatorMap;
    typedef Map<string, unique_ptr<GraphBase> > GraphMap;
    typedef Map<string, Workspace*> WorkspaceMap;

    /*! \brief Constructor */
    Workspace(const string& name) : name_(name) { InitWorkspace(); }

    /*! \brief Return the name of this workspace */
    const string& name() { return name_; }

    /*! \brief Create some internal tensors */
    void InitWorkspace();

    /*! \brief Move a external workspace into this workspace */
    Workspace* Move(Workspace* ws);

    /*! \brief Destory all the tensors */
    void Clear();

    /*! \brief Query the real name of specified tensor */
    string GetTensorName(const string& name) const;

    /*! \brief Try to serach the specified tensor in this workspace */
    Tensor* TryGetTensor(const string& name, bool use_remote = true) const;

    /*! \brief Whether the specified tensor is in this workspace */
    bool HasTensor(const string& name, bool use_remote = true) const {
        return TryGetTensor(name, use_remote) ? true : false;
    }

    /*! \brief Create the specified tensor */
    Tensor* CreateTensor(const string& name);

    /*! \brief Return the specified tensor */
    Tensor* GetTensor(const string& name, bool use_remote = true) const;

    /*! \brief Reset the specified tensor */
    void ResetTensor(const string& name);

    /*! \brief Return all the stored tensor names */
    vector<string> GetTensors() const;

    /* \brief Whether the specified filler is in this workspace */
    bool HasFiller(const string& name, bool use_remote = true) const;
    
    /*! \brief Create the specified filler */
    void CreateFiller(const TensorFillerProto filler);

    /*! \brief Return the specified filler */
    const TensorFillerProto* GetFiller(const string& name) const;

    /*! \brief Create temporal cache segments */
    template <class Context>
    vector<void*> caches(const vector<size_t>& segments) {
        int64_t nbytes = 0;
        for (auto& segment : segments) nbytes += (int64_t)segment;
        Tensor* cache_t = CreateTensor("/share/cache");
        cache_t->Reshape({ nbytes });
        vector<void*> Bcaches(segments.size());
        Bcaches[0] = cache_t->template mutable_data<uint8_t, Context>();
        for (int i = 1; i < segments.size(); i++)
            Bcaches[i] = (uint8_t*)Bcaches[i - 1] + segments[i - 1];
        return Bcaches;
    }

    /*! \brief Create temporal cache segments with the specified type */
    template <typename T, class Context>
    vector<T*> caches(const vector<int64_t>& segments) {
        vector<size_t> Tsegments;
        for (auto& segment : segments)
            Tsegments.emplace_back(segment * sizeof(T));
        vector<void*> Bcaches = caches<Context>(Tsegments);
        vector<T*> Tcaches(segments.size());
        for (int i = 0; i < segments.size(); i++)
            Tcaches[i] = (T*)Bcaches[i];
        return Tcaches;
    }

    /*! \brief Creathe a persistent operator in this workspace */
    void CreatePersistentOp(const OperatorDef& def);

    /*! \brief Run the specified persistent operator */
    void RunPersistentOp(
        const string&               key,
        const string&               anchor,
        const vector<string>&       inputs,
        const vector<string>&       outputs);
    
    /*! \brief Try to run the operator in a adaptive mode */
    void RunOperator(const OperatorDef& def);

    /*! \brief Create a Graph in this workspace */
    GraphBase* CreateGraph(const GraphDef& def);

    /*! \brief Run the specifed graph by name and rules */
    void RunGraph(
        const string&               graph_name,
        const string&               include,
        const string&               exclude,
        const int                   stream_id = 1);

    /*! \brief Return all the stored graph names */
    vector<string> GetGraphs() const;

    /* \brief Set a proxy name for the tensor */
    bool SetTensorProxy(const string& key, const string& proxy);

    /* \brief Return a unique dummy name within this workspace */
    string GetDummyName(
        const string&               base_name,
        const string&               suffix,
        const string&               domain = "",
        const bool                  zero_based = true);

 private:
    /*! \brief The unique workspace name */
    string name_;

    /*! \brief The dummy name indices */
    DummyNameMap dummy_name_map_;

    /*! \brief Store the created tensors */
    TensorMap tensor_map_;

    /*! \brief Store the registered tensor fillers */
    TensorFillerMap tensor_filler_map_;

    /*! \brief Store the proxy name of tensors */
    TensorProxyMap tensor_proxy_map_;

    /*! \brief Store the registered operators for dynamic graph */
    OperatorMap operator_map_;

    /*! \brief Store the registered graphs for static graph */
    GraphMap graph_map_;

    /*! \brief Store the remote workspaces */
    WorkspaceMap workspace_map_;
};

}  // namespace dragon

#endif  // DRAGON_CORE_WORKSPACE_H_