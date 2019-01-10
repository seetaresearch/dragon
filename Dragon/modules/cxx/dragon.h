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

#ifndef DRAGON_CXX_DRAGON_H_
#define DRAGON_CXX_DRAGON_H_

#include <string>
#include <cstdint>
#include <vector>

#ifdef _MSC_VER
    #ifdef DRAGON_CXX_EXPORTS
        #define DRAGON_API __declspec(dllexport)
    #else
        #define DRAGON_API __declspec(dllimport)
    #endif
#else
    #define DRAGON_API
#endif

/* * * * * * * * * * * * * * * * * * * * *
 *                                       *
 *           Internal Headers            *
 *                                       *
 * * * * * * * * * * * * * * * * * * * * */

#ifdef DRAGON_CXX_EXPORTS
#else
namespace dragon {

typedef struct float16* float16_t;

}
#endif  // DRAGON_CXX_EXPORTS

namespace dragon {

typedef struct GraphDef* GraphDef_t;
typedef struct Workspace* Workspace_t;

class DRAGON_API Device {
 public:
    Device();
    explicit Device(std::string device_type);
    Device(std::string device_type, int device_id);

    const int& device_type() const { return device_type_; }
    const int device_id() const { return device_id_; }

 private:
    int device_type_, device_id_;
};

/* * * * * * * * * * * * * * * * * * * * *
 *                                       *
 *               Workspace               *
 *                                       *
 * * * * * * * * * * * * * * * * * * * * */

DRAGON_API Workspace_t CreateWorkspace(const std::string& name);

DRAGON_API Workspace_t ResetWorkspace(Workspace_t ws);

DRAGON_API Workspace_t ResetWorkspace(const std::string& name);

DRAGON_API void MoveWorkspace(Workspace_t dst, Workspace_t src);

DRAGON_API void DestroyWorkspace(Workspace_t ws);

DRAGON_API void DestroyWorkspace(const std::string& name);

/* * * * * * * * * * * * * * * * * * * * *
 *                                       *
 *                Graph                  *
 *                                       *
 * * * * * * * * * * * * * * * * * * * * */

DRAGON_API std::string CreateGraph(
    const GraphDef_t                graph_def,
    const Device&                   device,
    Workspace_t                     ws);

DRAGON_API std::string CreateGraph(
    const std::string&              graph_file,
    const Device&                   device,
    Workspace_t                     ws);

DRAGON_API void RunGraph(
    const std::string&              graph_name,
    Workspace_t                     ws,
    const int                       stream_id = 1);

/* * * * * * * * * * * * * * * * * * * * *
 *                                       *
 *                Tensor                 *
 *                                       *
 * * * * * * * * * * * * * * * * * * * * */

DRAGON_API void CreateTensor(
    const std::string&              name,
    Workspace_t                     ws);

template <typename T>
DRAGON_API T* FetchTensor(
    const std::string&              name,
    std::vector<int64_t>&           shape,
    Workspace_t                     ws,
    const bool                      copy = false);

template <typename T>
DRAGON_API void FeedTensor(
    const std::string&              name,
    const std::vector<int64_t>&     shape,
    const T*                        data,
    const Device&                   device,
    Workspace_t                     ws);

/* * * * * * * * * * * * * * * * * * * * *
*                                        *
*                 Proto                  *
*                                        *
* * * * * * * * * * * * * * * * * * * * */

DRAGON_API void CreateGraphDef(GraphDef_t* graph_def);
DRAGON_API void DestroyGraphDef(GraphDef_t graph_def);

/* * * * * * * * * * * * * * * * * * * * *
 *                                       *
 *                I / O                  *
 *                                       *
 * * * * * * * * * * * * * * * * * * * * */

DRAGON_API void LoadCaffeModel(
    const std::string&              model_file,
    Workspace_t                     ws);

DRAGON_API void LoadONNXModel(
    const std::string&              model_file,
    GraphDef_t                      init_graph,
    GraphDef_t                      pred_graph,
    std::vector<std::string>&       inputs,
    std::vector<std::string>&       outputs);

/* * * * * * * * * * * * * * * * * * * * *
 *                                       *
 *                Config                 *
 *                                       *
 * * * * * * * * * * * * * * * * * * * * */

DRAGON_API void SetLogLevel(const std::string& level);

}  // namespace dragon

#endif  // DRAGON_CXX_DRAGON_H_