/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
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
  DRAGON_API explicit Workspace(const string& name);

  /*! \brief Merge resources from other */
  DRAGON_API void MergeFrom(Workspace*);

  /*! \brief Clear the cached resources */
  DRAGON_API void Clear();

  /* \brief Return an unique name */
  DRAGON_API string UniqueName(
      const string& name,
      const string& suffix,
      const string& scope = "",
      const bool zero_based = false);

  /* \brief Register an alias for the target */
  DRAGON_API void RegisterAlias(const string& target, const string& alias);

  /*! \brief Return whether tensor is existing */
  DRAGON_API bool HasTensor(const string& name, bool external = true) const {
    return TryGetTensor(name, external) == nullptr ? false : true;
  }

  /*! \brief Create the tensor */
  DRAGON_API Tensor* CreateTensor(const string&, FillerInfo* = nullptr);

  /*! \brief Try to return the tensor */
  DRAGON_API Tensor* TryGetTensor(const string&, bool = true) const;

  /*! \brief Return the tensor */
  DRAGON_API Tensor* GetTensor(const string&, bool = true) const;

  /*! \brief Reset the tensor */
  DRAGON_API void ResetTensor(const string&);

  /*! \brief Return the filler info */
  DRAGON_API FillerInfo* GetFillerInfo(const string&);

  /*! \brief Run the operator */
  DRAGON_API void RunOperator(const OperatorDef&);

  /*! \brief Create the graph */
  DRAGON_API GraphBase* CreateGraph(const GraphDef&);

  /*! \brief Run the graph */
  DRAGON_API void RunGraph(
      const string& graph_name,
      const string& include = "",
      const string& exclude = "",
      const int stream = 0);

  /*! \brief Return the workspace name */
  const string& name() {
    return name_;
  }

  /*! \brief Return the name of cached tensors */
  DRAGON_API vector<string> tensors() const;

  /*! \brief Return the name of cached graphs  */
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
  /*! \brief The workspace name */
  string name_;

  /*! \brief The external tensors */
  Map<string, Tensor*> external_tensor_map_;

  /*! \brief The unique indices */
  Map<string, Map<string, int64_t>> unique_index_map_;

  /*! \brief The registered fillers */
  Map<string, FillerInfo> filler_map_;

  /*! \brief The registered aliases */
  Map<string, string> alias_map_;

  /*! \brief The cached tensors */
  Map<string, unique_ptr<Tensor>> tensor_map_;

  /*! \brief The cached operators */
  Map<string, unique_ptr<OperatorBase>> operator_map_;

  /*! \brief The cached graphs */
  Map<string, unique_ptr<GraphBase>> graph_map_;
};

} // namespace dragon

#endif // DRAGON_CORE_WORKSPACE_H_
