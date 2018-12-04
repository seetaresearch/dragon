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
#include "core/types.h"
#else
namespace dragon {
    struct float16;
}
#endif

namespace dragon {

typedef int64_t TIndex;

class Workspace;

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

DRAGON_API Workspace* CreateWorkspace(const std::string& name);

DRAGON_API Workspace* ResetWorkspace(const std::string& name);

DRAGON_API void ReleaseWorkspace(const std::string& name);

DRAGON_API void MoveWorkspace(Workspace* main, Workspace* sub);

/* * * * * * * * * * * * * * * * * * * * *
 *                                       *
 *                Graph                  *
 *                                       *
 * * * * * * * * * * * * * * * * * * * * */

DRAGON_API std::string CreateGraph(
    const std::string&          graph_file,
    Workspace*                  ws);

DRAGON_API std::string CreateGraph(
    const std::string&          graph_file,
    const Device&               device,
    Workspace*                  ws);

DRAGON_API void RunGraph(
    const std::string&          graph_name,
    Workspace*                  ws,
    const int                   stream_id = 1);

/* * * * * * * * * * * * * * * * * * * * *
 *                                       *
 *                Tensor                 *
 *                                       *
 * * * * * * * * * * * * * * * * * * * * */

DRAGON_API void CreateTensor(
    const std::string&          name,
    Workspace*                  ws);

template <typename T>
DRAGON_API T* FetchTensor(
    const std::string&          name,
    std::vector<TIndex>&        shape,
    Workspace*                  ws);

template <typename T>
DRAGON_API void FeedTensor(
    const std::string&          name,
    const std::vector<TIndex>&  shape,
    const T*                    data,
    const Device&               device,
    Workspace*                  ws);

/* * * * * * * * * * * * * * * * * * * * *
 *                                       *
 *                I / O                  *
 *                                       *
 * * * * * * * * * * * * * * * * * * * * */

DRAGON_API void LoadCaffemodel(
    const std::string&          model_file,
    Workspace*                  ws);

DRAGON_API void TransplantCaffeModel(
    const std::string&          input_model,
    const std::string&          output_model);

DRAGON_API void LoadDragonmodel(
    const std::string&          model_file,
    Workspace*                  ws);

/* * * * * * * * * * * * * * * * * * * * *
 *                                       *
 *                Config                 *
 *                                       *
 * * * * * * * * * * * * * * * * * * * * */

DRAGON_API void SetLogLevel(const std::string& level);

}  // namespace dragon

#endif  // DRAGON_CXX_DRAGON_H_