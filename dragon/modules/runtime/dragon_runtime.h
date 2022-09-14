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

#ifndef DRAGON_MODULES_RUNTIME_DRAGON_RUNTIME_H_
#define DRAGON_MODULES_RUNTIME_DRAGON_RUNTIME_H_

#include <cstdint>
#include <string>
#include <vector>

#ifndef DRAGON_API
#define DRAGON_API
namespace dragon {
typedef struct float16* float16_t;
} // namespace dragon
#endif // DRAGON_API

namespace dragon {

typedef class GraphDef* GraphDef_t;
typedef class Workspace* Workspace_t;

/*!
 * Device API.
 */

class DRAGON_API Device {
 public:
  Device();
  explicit Device(const std::string& device_type);
  Device(const std::string& device_type, int device_index);

  const std::string& device_type() const {
    return device_type_;
  }

  const int device_index() const {
    return device_index_;
  }

 private:
  std::string device_type_;
  int device_index_;
};

/*!
 * Workspace API.
 */

DRAGON_API Workspace_t CreateWorkspace(const std::string& name);

DRAGON_API Workspace_t ResetWorkspace(Workspace_t ws);

DRAGON_API Workspace_t ResetWorkspace(const std::string& name);

DRAGON_API void MoveWorkspace(Workspace_t dest, Workspace_t src);

DRAGON_API void DestroyWorkspace(Workspace_t ws);

DRAGON_API void DestroyWorkspace(const std::string& name);

/*!
 * Graph API.
 */

DRAGON_API std::string
CreateGraph(const GraphDef_t def, const Device& device, Workspace_t ws);

DRAGON_API std::string
CreateGraph(const std::string& file, const Device& device, Workspace_t ws);

DRAGON_API void
RunGraph(const std::string& name, Workspace_t ws, int stream = 0);

/*!
 * Tensor API.
 */

DRAGON_API void CreateTensor(const std::string& name, Workspace_t ws);

template <typename T>
DRAGON_API void FeedTensor(
    const std::string& name,
    const std::vector<int64_t>& shape,
    const T* data,
    const Device& device,
    Workspace_t ws);

template <typename T>
DRAGON_API T* FetchTensor(
    const std::string& name,
    std::vector<int64_t>& shape,
    Workspace_t ws,
    bool copy = false);

/*!
 * Proto API.
 */

DRAGON_API void CreateGraphDef(GraphDef_t* def);

DRAGON_API void DestroyGraphDef(GraphDef_t def);

/*!
 * Model API.
 */

DRAGON_API void LoadONNXModel(
    const std::string& model_file,
    GraphDef_t init_def,
    GraphDef_t pred_def,
    std::vector<std::string>& inputs,
    std::vector<std::string>& outputs);

/*!
 * Config API.
 */

DRAGON_API void SetLoggingLevel(const std::string& level);

} // namespace dragon

#endif // DRAGON_MODULES_RUNTIME_DRAGON_RUNTIME_H_
