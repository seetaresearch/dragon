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

class Workspace {
 public:
    typedef Map<string, Map<string, int64_t> > DummyNameMap;

    typedef Map<string, unique_ptr<Tensor> > TensorMap;
    typedef Map<string, string> TensorAliasMap;
    typedef Map<string, TensorFillerProto> TensorFillerMap;

    typedef Map<string, unique_ptr<OperatorBase> > OperatorMap;
    typedef Map<string, unique_ptr<GraphBase> > GraphMap;

    /*! \brief Constructor */
    Workspace(const string& name) : name_(name) { Initialize(); }

    /*! \brief Return the name of this workspace */
    const string& name() { return name_; }

    /*! \brief Return the name of stored tensors */
    vector<string> tensors() const;

    /*! \brief Return the name of stored graphs */
    vector<string> graphs() const;

    /*! \brief Create some internal tensors */
    void Initialize();

    /*! \brief Destory all the tensors */
    void Clear();

    /*! \brief Merge from a external workspace */
    void MergeFrom(Workspace* ws);

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

    /* \brief Whether the specified filler is in this workspace */
    bool HasFiller(const string& name, bool use_remote = true) const;

    /*! \brief Create the specified filler */
    void CreateFiller(const TensorFillerProto& filler);

    /*! \brief Return the specified filler */
    const TensorFillerProto* GetFiller(const string& name) const;

    /*! \brief Create temporal cache segments */
    template <class Context>
    vector<void*> caches(const vector<size_t>& segments) {
        int64_t nbytes = 0;
        vector<void*> ret(segments.size());
        for (auto& segment : segments) nbytes += (int64_t)segment;
        auto* T = CreateTensor("/share/cache")->Reshape({ nbytes });
        ret[0] = T->template mutable_data<uint8_t, Context>();
        for (int i = 1; i < segments.size(); i++)
            ret[i] = (uint8_t*)ret[i - 1] + segments[i - 1];
        return ret;
    }

    /*! \brief Create temporal cache segments with the specified type */
    template <typename T, class Context>
    vector<T*> caches(const vector<int64_t>& segments) {
        vector<size_t> segments_in_byte;
        vector<T*> ret(segments.size());
        for (const auto& e : segments)
            segments_in_byte.emplace_back(e * sizeof(T));
        auto ret_in_byte = caches<Context>(segments_in_byte);
        for (int i = 0; i < segments.size(); i++)
            ret[i] = (T*)ret_in_byte[i];
        return ret;
    }

    /*! \brief Create a operator in this workspace */
    OperatorBase* CreateOperator(const OperatorDef& def);

    /*! \brief Run the specified persistent operator */
    void RunOperator(const OperatorDef& def);

    /*! \brief Try to run the operator in a adaptive mode */
    void RunOperatorOnce(const OperatorDef& def);

    /*! \brief Create a Graph in this workspace */
    GraphBase* CreateGraph(const GraphDef& def);

    /*! \brief Run the specifed graph by name and rules */
    void RunGraph(
        const string&               graph_name,
        const string&               include,
        const string&               exclude,
        int                         stream_id = 0);

    /* \brief Set an alias for the tensor */
    bool SetTensorAlias(const string& name, const string& alias);

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
    TensorAliasMap tensor_alias_map_;

    /*! \brief Store the registered operators for dynamic graph */
    OperatorMap operator_map_;

    /*! \brief Store the registered graphs for static graph */
    GraphMap graph_map_;

    /*! \brief Store the remote workspaces */
    vector<Workspace*> remote_workspaces_;
};

}  // namespace dragon

#endif  // DRAGON_CORE_WORKSPACE_H_