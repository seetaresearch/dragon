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

#ifndef DRAGON_OPERATORS_VISION_CONV_OP_ALGO_H_
#define DRAGON_OPERATORS_VISION_CONV_OP_ALGO_H_

#include "dragon/core/common.h"
#include "dragon/utils/device/common_cudnn.h"

namespace dragon {

template <typename Algo>
class ConvAlgoSearch {};

template <typename Algo>
class ConvAlgoCache {
 public:
  Algo get(
      const vec64_t& Xdims,
      const vec64_t& Wdims,
      int flag,
      std::function<Algo()> gen_func) {
    int64_t key = 0;
    std::hash<int64_t> hash_func;
    for (auto dim : Xdims) {
      key ^= hash_func(dim) + 0x9e3779b9 + (key << 6) + (key >> 2);
    }
    for (auto dim : Wdims) {
      key ^= hash_func(dim) + 0x9e3779b9 + (key << 6) + (key >> 2) + 1;
    }
    key ^= hash_func(flag) + 0x9e3779b9 + (key << 6) + (key >> 2) + 2;
    if (map_.find(key) == map_.end()) {
      auto value = gen_func();
      map_[key] = value;
    }
    return map_[key];
  }

 private:
  Map<int64_t, Algo> map_;
};

#ifdef USE_CUDNN

constexpr size_t CUDNN_CONV_WORKSPACE_LIMIT_BYTES = 512 * 1024 * 1024;

/*
 * cudnnConvolutionFwdAlgo
 */

template <>
class ConvAlgoSearch<cudnnConvolutionFwdAlgo_t> {
  using Algo = cudnnConvolutionFwdAlgo_t;
  using AlgoPerf = cudnnConvolutionFwdAlgoPerf_t;

 public:
  Algo get(
      cudnnHandle_t handle,
      const cudnnTensorDescriptor_t xDesc,
      const cudnnFilterDescriptor_t wDesc,
      const cudnnConvolutionDescriptor_t convDesc,
      const cudnnTensorDescriptor_t yDesc,
      size_t workSpaceSizeInBytes) {
    constexpr int num = 2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    AlgoPerf perfs[num];
    int num_valid;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
        handle, xDesc, wDesc, convDesc, yDesc, num, &num_valid, perfs));
    for (int i = 0; i < num_valid; ++i) {
      if (perfs[i].memory <= workSpaceSizeInBytes) {
        return perfs[i].algo;
      }
    }
    return get_default();
  }

  std::tuple<Algo, float> find(
      cudnnHandle_t handle,
      const cudnnTensorDescriptor_t xDesc,
      const cudnnFilterDescriptor_t wDesc,
      const cudnnConvolutionDescriptor_t convDesc,
      const cudnnTensorDescriptor_t yDesc,
      size_t workSpaceSizeInBytes,
      const void* x,
      const void* w,
      void* y,
      void* workSpace) {
    constexpr int num = 2 * CUDNN_CONVOLUTION_FWD_ALGO_COUNT;
    AlgoPerf perfs[num];
    int num_valid;
    CUDNN_CHECK(cudnnFindConvolutionForwardAlgorithmEx(
        handle,
        xDesc,
        x,
        wDesc,
        w,
        convDesc,
        yDesc,
        y,
        num,
        &num_valid,
        perfs,
        workSpace,
        workSpaceSizeInBytes));
    return std::tuple<Algo, float>(perfs[0].algo, perfs[0].time);
  }

  Algo get_default() {
    return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  }

  Algo get_deterministic() {
    return CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  }
};

/*
 * cudnnConvolutionBwdDataAlgo
 */

template <>
class ConvAlgoSearch<cudnnConvolutionBwdDataAlgo_t> {
  using Algo = cudnnConvolutionBwdDataAlgo_t;
  using AlgoPerf = cudnnConvolutionBwdDataAlgoPerf_t;

 public:
  Algo get(
      cudnnHandle_t handle,
      const cudnnFilterDescriptor_t wDesc,
      const cudnnTensorDescriptor_t dyDesc,
      const cudnnConvolutionDescriptor_t convDesc,
      const cudnnTensorDescriptor_t dxDesc,
      size_t workSpaceSizeInBytes) {
    constexpr int num = 2 * CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
    AlgoPerf perfs[num];
    int num_valid;
    CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm_v7(
        handle, wDesc, dyDesc, convDesc, dxDesc, num, &num_valid, perfs));
    for (int i = 0; i < num_valid; ++i) {
      if (perfs[i].memory <= workSpaceSizeInBytes) {
        return perfs[i].algo;
      }
    }
    return get_default();
  }

  std::tuple<Algo, float> find(
      cudnnHandle_t handle,
      const cudnnFilterDescriptor_t wDesc,
      const cudnnTensorDescriptor_t dyDesc,
      const cudnnConvolutionDescriptor_t convDesc,
      const cudnnTensorDescriptor_t dxDesc,
      size_t workSpaceSizeInBytes,
      const void* w,
      const void* dy,
      void* dx,
      void* workSpace) {
    constexpr int num = 2 * CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT;
    AlgoPerf perfs[num];
    int num_valid;
    CUDNN_CHECK(cudnnFindConvolutionBackwardDataAlgorithmEx(
        handle,
        wDesc,
        w,
        dyDesc,
        dy,
        convDesc,
        dxDesc,
        dx,
        num,
        &num_valid,
        perfs,
        workSpace,
        workSpaceSizeInBytes));
    return std::tuple<Algo, float>(perfs[0].algo, perfs[0].time);
  }

  Algo get_default() {
    return CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
  }

  Algo get_deterministic() {
    return CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
  }
};

/*
 * cudnnConvolutionBwdFilterAlgo
 */

template <>
class ConvAlgoSearch<cudnnConvolutionBwdFilterAlgo_t> {
  using Algo = cudnnConvolutionBwdFilterAlgo_t;
  using AlgoPerf = cudnnConvolutionBwdFilterAlgoPerf_t;

 public:
  Algo get(
      cudnnHandle_t handle,
      const cudnnTensorDescriptor_t xDesc,
      const cudnnTensorDescriptor_t dyDesc,
      const cudnnConvolutionDescriptor_t convDesc,
      const cudnnFilterDescriptor_t dwDesc,
      size_t workSpaceSizeInBytes) {
    constexpr int num = 2 * CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
    AlgoPerf perfs[num];
    int num_valid;
    CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
        handle, xDesc, dyDesc, convDesc, dwDesc, num, &num_valid, perfs));
    for (int i = 0; i < num_valid; ++i) {
      if (perfs[i].memory <= workSpaceSizeInBytes) {
        return perfs[i].algo;
      }
    }
    return get_default();
  }

  std::tuple<Algo, float> find(
      cudnnHandle_t handle,
      const cudnnTensorDescriptor_t xDesc,
      const cudnnTensorDescriptor_t dyDesc,
      const cudnnConvolutionDescriptor_t convDesc,
      const cudnnFilterDescriptor_t dwDesc,
      size_t workSpaceSizeInBytes,
      const void* x,
      const void* dy,
      void* dw,
      void* workSpace) {
    constexpr int num = 2 * CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT;
    AlgoPerf perfs[num];
    int num_valid;
    CUDNN_CHECK(cudnnFindConvolutionBackwardFilterAlgorithmEx(
        handle,
        xDesc,
        x,
        dyDesc,
        dy,
        convDesc,
        dwDesc,
        dw,
        num,
        &num_valid,
        perfs,
        workSpace,
        workSpaceSizeInBytes));
    return std::tuple<Algo, float>(perfs[0].algo, perfs[0].time);
  }

  Algo get_default() {
    return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
  }

  Algo get_deterministic() {
    return CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  }
};

#endif // USE_CUDNN

} // namespace dragon

#endif // DRAGON_OPERATORS_VISION_CONV_OP_ALGO_H_
