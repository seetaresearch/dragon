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

#ifndef DRAGON_KERNELS_VISION_OP_KERNELS_H_
#define DRAGON_KERNELS_VISION_OP_KERNELS_H_

#include "dragon/core/context.h"
#include "dragon/core/context_cuda.h"
#include "dragon/core/context_mps.h"

namespace dragon {

namespace kernels {

template <typename T, class Context>
void AvgPool2d(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const string& data_format,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void AvgPool2dGrad(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const string& data_format,
    const T* dy,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void AvgPool3d(
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const int out_d,
    const int out_h,
    const int out_w,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const string& data_format,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void AvgPool3dGrad(
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const int out_d,
    const int out_h,
    const int out_w,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const string& data_format,
    const T* dy,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void BiasAdd(
    const int N,
    const int S,
    const int C,
    const T* x,
    const T* bias,
    T* y,
    Context* ctx);

template <typename T, class Context>
void Col2Im2d(
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const string& data_format,
    const T* col,
    T* im,
    Context* ctx);

template <typename T, class Context>
void Col2ImNd(
    const int num_dims,
    const int channels,
    const int* in_shape,
    const int* out_shape,
    const int* kshape,
    const int* strides,
    const int* pads,
    const int* dilations,
    const string& data_format,
    const T* col,
    T* im,
    Context* ctx);

template <typename T, class Context>
void Im2Col2d(
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const string& data_format,
    const T* im,
    T* col,
    Context* ctx);

template <typename T, class Context>
void Im2ColNd(
    const int num_dims,
    const int channels,
    const int* in_shape,
    const int* out_shape,
    const int* kshape,
    const int* strides,
    const int* pads,
    const int* dilations,
    const string& data_format,
    const T* im,
    T* col,
    Context* ctx);

template <typename T, class Context>
void DepthwiseConv2d(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const string& data_format,
    const T* x,
    const T* filter,
    T* y,
    Context* ctx);

template <typename T, class Context>
void DepthwiseConv2dGrad(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const string& data_format,
    const T* dy,
    const T* filter,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void DepthwiseConv2dWGrad(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const int dilation_h,
    const int dilation_w,
    const string& data_format,
    const T* dy,
    const T* x,
    T* dfilter,
    Context* ctx);

template <typename T, class Context>
void MaxPool2d(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const string& data_format,
    const T* x,
    int* mask,
    T* y,
    Context* ctx);

template <typename T, class Context>
void MaxPool2dGrad(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int kernel_h,
    const int kernel_w,
    const int stride_h,
    const int stride_w,
    const int pad_h,
    const int pad_w,
    const string& data_format,
    const T* dy,
    int* mask,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void MaxPool3d(
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const int out_d,
    const int out_h,
    const int out_w,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const string& data_format,
    const T* x,
    int* mask,
    T* y,
    Context* ctx);

template <typename T, class Context>
void MaxPool3dGrad(
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const int out_d,
    const int out_h,
    const int out_w,
    const int kernel_d,
    const int kernel_h,
    const int kernel_w,
    const int stride_d,
    const int stride_h,
    const int stride_w,
    const int pad_d,
    const int pad_h,
    const int pad_w,
    const string& data_format,
    const T* dy,
    int* mask,
    T* dx,
    Context* ctx);

template <typename T, class Context>
void ResizeLinear2d(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const bool align_corners,
    const string& data_format,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void ResizeLinear2dGrad(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const bool align_corners,
    const string& data_format,
    const T* dy,
    float* dx,
    Context* ctx);

template <typename T, class Context>
void ResizeNearest2d(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const string& data_format,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void ResizeNearest2dGrad(
    const int N,
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const string& data_format,
    const T* dy,
    float* dx,
    Context* ctx);

template <typename T, class Context>
void ResizeNearest3d(
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const int out_d,
    const int out_h,
    const int out_w,
    const string& data_format,
    const T* x,
    T* y,
    Context* ctx);

template <typename T, class Context>
void ResizeNearest3dGrad(
    const int N,
    const int C,
    const int D,
    const int H,
    const int W,
    const int out_d,
    const int out_h,
    const int out_w,
    const string& data_format,
    const T* dy,
    float* dx,
    Context* ctx);

template <typename T, class Context>
void RoiAlign(
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int num_rois,
    const float spatial_scale,
    const int sampling_ratio,
    const bool aligned,
    const T* x,
    const float* rois,
    T* y,
    Context* ctx);

template <typename T, class Context>
void RoiAlignGrad(
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int num_rois,
    const float spatial_scale,
    const int sampling_ratio,
    const bool aligned,
    const T* dy,
    const float* rois,
    float* dx,
    Context* ctx);

template <typename T, class Context>
void RoiPool(
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int num_rois,
    const float spatial_scale,
    const T* x,
    const float* rois,
    int* index,
    T* y,
    Context* ctx);

template <typename T, class Context>
void RoiPoolGrad(
    const int C,
    const int H,
    const int W,
    const int out_h,
    const int out_w,
    const int num_rois,
    const float spatial_scale,
    const T* dy,
    const float* rois,
    int* index,
    float* dx,
    Context* ctx);

} // namespace kernels

} // namespace dragon

#endif // DRAGON_KERNELS_VISION_OP_KERNELS_H_
