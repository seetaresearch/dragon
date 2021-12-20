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

  /* \brief Set an alias for the target */
  void SetAlias(const string& target, const string& alias) {
    aliases_[alias] = target;
  }

  /*! \brief Return whether tensor is existing */
  bool HasTensor(const string& name, bool external = true) const {
    return TryGetTensor(name, external) == nullptr ? false : true;
  }

  /*! \brief Create the tensor */
  Tensor* CreateTensor(const string& name);

  /*! \brief Try to return the tensor */
  Tensor* TryGetTensor(const string& name, bool external = true) const;

  /*! \brief Return the tensor */
  Tensor* GetTensor(const string& name, bool external = true) const;

  /*! \brief Run the operator */
  void RunOperator(const OperatorDef& def);

  /*! \brief Create the graph */
  GraphBase* CreateGraph(const GraphDef& def);

  /*! \brief Run the graph */
  void RunGraph(
      const string& name,
      const string& include = "",
      const string& exclude = "",
      int stream = 0);

  /*! \brief Return the workspace name */
  const string& name() {
    return name_;
  }

  /*! \brief Return the name of created tensors */
  vector<string> tensors(bool external = true) const;

  /*! \brief Return the name of created graphs  */
  vector<string> graphs() const;

  /*! \brief Return a shared raw data */
  template <class Context>
  void* data(size_t size, const string& name = "BufferShared") {
    size = size > size_t(0) ? size : size_t(1);
    auto* tensor = CreateTensor(name)->Reshape({int64_t(size)});
    return (void*)tensor->template mutable_data<uint8_t, Context>();
  }

  /*! \brief Return a shared typed data */
  template <typename T, class Context>
  T* data(int64_t size, const string& name = "BufferShared") {
    return (T*)data<Context>(size_t(size) * sizeof(T), name);
  }

 private:
  /*! \brief The workspace name */
  string name_;

  /*! \brief The scope counters */
  Map<string, Map<string, int64_t>> scope_counters_;

  /*! \brief The aliases */
  Map<string, string> aliases_;

  /*! \brief The tensors */
  Map<string, unique_ptr<Tensor>> tensors_;

  /*! \brief The external tensors */
  Map<string, Tensor*> external_tensors_;

  /*! \brief The operators */
  Map<string, unique_ptr<OperatorBase>> operators_;

  /*! \brief The graphs */
  Map<string, unique_ptr<GraphBase>> graphs_;

  DISABLE_COPY_AND_ASSIGN(Workspace);
};

} // namespace dragon

#endif // DRAGON_CORE_WORKSPACE_H_
