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

/*!
 * \brief Sandbox to isolate the resources and computations.
 */
class DRAGON_API Workspace {
 public:
  /*! \brief Constructor with the name */
  explicit Workspace(const string& name);

  /*! \brief Merge resources from other */
  void MergeFrom(Workspace* other);

  /*! \brief Clear the cached resources */
  void Clear();

  /* \brief Return an unique name */
  string UniqueName(
      const string& name,
      const string& suffix,
      const string& scope = "",
      const bool zero_based = false);

  /* \brief Register an alias for the target */
  void RegisterAlias(const string& target, const string& alias);

  /*! \brief Return whether tensor is existing */
  bool HasTensor(const string& name, bool external = true) const {
    return TryGetTensor(name, external) == nullptr ? false : true;
  }

  /*! \brief Create the tensor */
  Tensor* CreateTensor(const string& name, FillerInfo* filler = nullptr);

  /*! \brief Try to return the tensor */
  Tensor* TryGetTensor(const string& name, bool external = true) const;

  /*! \brief Return the tensor */
  Tensor* GetTensor(const string& name, bool external = true) const;

  /*! \brief Reset the tensor */
  void ResetTensor(const string& name);

  /*! \brief Return the filler info */
  FillerInfo* GetFillerInfo(const string& name);

  /*! \brief Run the operator */
  void RunOperator(const OperatorDef& def);

  /*! \brief Create the graph */
  GraphBase* CreateGraph(const GraphDef& def);

  /*! \brief Run the graph */
  void RunGraph(
      const string& name,
      const string& include = "",
      const string& exclude = "",
      const int stream = 0);

  /*! \brief Return the workspace name */
  const string& name() {
    return name_;
  }

  /*! \brief Return the name of cached tensors */
  vector<string> tensors() const;

  /*! \brief Return the name of cached graphs  */
  vector<string> graphs() const;

  /*! \brief Return a group of the shared raw data */
  template <class Context>
  vector<void*> data(const vector<size_t>& segments) {
    vector<void*> group(segments.size());
    auto total_bytes = std::accumulate(segments.begin(), segments.end(), 0);
    group[0] = CreateTensor("/share/data")
                   ->Reshape({(int64_t)total_bytes})
                   ->template mutable_data<uint8_t, Context>();
    for (int i = 1; i < segments.size(); ++i) {
      group[i] = (uint8_t*)group[i - 1] + segments[i - 1];
    }
    return group;
  }

  /*! \brief Return a group of shared typed data */
  template <typename T, class Context>
  vector<T*> data(const vector<int64_t>& segments) {
    vector<T*> group(segments.size());
    vector<size_t> segments_v2;
    for (const auto size : segments) {
      segments_v2.push_back(size * sizeof(T));
    }
    auto group_v2 = data<Context>(segments_v2);
    for (int i = 0; i < segments.size(); ++i) {
      group[i] = (T*)group_v2[i];
    }
    return group;
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

  DISABLE_COPY_AND_ASSIGN(Workspace);
};

} // namespace dragon

#endif // DRAGON_CORE_WORKSPACE_H_
