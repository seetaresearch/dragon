/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *    <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_CORE_WORKSPACE_H_
#define DRAGON_CORE_WORKSPACE_H_

#include "dragon/core/graph.h"

namespace dragon {

class Workspace {
 public:
  /*! \brief Constructor */
  explicit Workspace(const string& name) : name_(name) {
    Initialize();
  }

  /*! \brief Create some internal tensors */
  DRAGON_API void Initialize();

  /*! \brief Merge tensors from a external workspace */
  DRAGON_API void MergeFrom(Workspace*);

  /*! \brief Destory all the tensors */
  DRAGON_API void Clear();

  /* \brief Return a unique dummy name within this workspace */
  DRAGON_API string GetDummyName(
      const string& base_name,
      const string& suffix,
      const string& domain = "",
      bool zero_based = true);

  /*! \brief Whether the specified tensor is in this workspace */
  DRAGON_API bool HasTensor(const string& name, bool use_remote = true) const {
    return TryGetTensor(name, use_remote) ? true : false;
  }

  /*! \brief Query the real name of specified tensor */
  DRAGON_API string GetTensorName(const string&) const;

  /* \brief Activate an alias for the target */
  DRAGON_API bool ActivateAlias(const string& name, const string& alias);

  /*! \brief Create a tensor in this workspace */
  DRAGON_API Tensor* CreateTensor(const string&);

  /*! \brief Try to search the specified tensor in this workspace */
  DRAGON_API Tensor* TryGetTensor(const string&, bool = true) const;

  /*! \brief Return the specified tensor */
  DRAGON_API Tensor* GetTensor(const string&, bool = true) const;

  /*! \brief Reset the specified tensor */
  DRAGON_API void ResetTensor(const string&);

  /* \brief Whether the specified filler is existing */
  DRAGON_API bool HasFiller(const string&) const;

  /*! \brief Create a filler in this workspace */
  DRAGON_API void CreateFiller(const TensorFillerProto&);

  /*! \brief Return the specified filler */
  DRAGON_API TensorFillerProto* GetFiller(const string&);

  /*! \brief Create an operator in this workspace */
  DRAGON_API OperatorBase* CreateOperator(const OperatorDef&);

  /*! \brief Run an operator in this workspace */
  DRAGON_API void RunOperator(const OperatorDef&);

  /*! \brief Create a graph in this workspace */
  DRAGON_API GraphBase* CreateGraph(const GraphDef&);

  /*! \brief Run the specifed graph by name and rules */
  DRAGON_API void RunGraph(
      const string& graph_name,
      const string& incl = "",
      const string& excl = "",
      int stream_id = 0);

  /*! \brief Return the name of this workspace */
  const string& name() {
    return name_;
  }

  /*! \brief Return the name of stored tensors */
  DRAGON_API vector<string> tensors() const;

  /*! \brief Return the name of stored graphs */
  DRAGON_API vector<string> graphs() const;

  /*! \brief Provide a group of the shared byte data */
  template <class Context>
  vector<void*> data(const vector<size_t>& segments) {
    int64_t nbytes = 0;
    vector<void*> ret(segments.size());
    for (auto& segment : segments)
      nbytes += (int64_t)segment;
    auto* T = CreateTensor("/share/data")->Reshape({nbytes});
    ret[0] = T->template mutable_data<uint8_t, Context>();
    for (int i = 1; i < segments.size(); i++)
      ret[i] = (uint8_t*)ret[i - 1] + segments[i - 1];
    return ret;
  }

  /*! \brief Provide a group of shared typed data */
  template <typename T, class Context>
  vector<T*> data(const vector<int64_t>& segments) {
    vector<size_t> segments_in_byte;
    vector<T*> ret(segments.size());
    for (const auto& e : segments)
      segments_in_byte.emplace_back(e * sizeof(T));
    auto ret_in_byte = data<Context>(segments_in_byte);
    for (int i = 0; i < segments.size(); i++)
      ret[i] = (T*)ret_in_byte[i];
    return ret;
  }

 private:
  /*! \brief The unique workspace name */
  string name_;

  /*! \brief The dummy name indices */
  Map<string, Map<string, int64_t>> dummy_name_map_;

  /*! \brief Store the created tensors */
  Map<string, unique_ptr<Tensor>> tensor_map_;

  /*! \brief Store the external tensors */
  Map<string, Tensor*> external_tensor_map_;

  /*! \brief Store the registered tensor fillers */
  Map<string, TensorFillerProto> tensor_filler_map_;

  /*! \brief Store the active aliases */
  Map<string, string> alias_active_map_;

  /*! \brief Store the registered operators for dynamic graph */
  Map<string, unique_ptr<OperatorBase>> operator_map_;

  /*! \brief Store the registered graphs for static graph */
  Map<string, unique_ptr<GraphBase>> graph_map_;
};

} // namespace dragon

#endif // DRAGON_CORE_WORKSPACE_H_
