// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_UTILS_OP_KERNEL_H_
#define DRAGON_UTILS_OP_KERNEL_H_

#include "core/context.h"

namespace dragon {

namespace kernel {

typedef int64_t TIndex;

template <typename T, class Context>
void Empty();

/******************** activation.dropout ********************/

template <typename T, class Context>
void Dropout(const int count, 
             T prob, 
             T scale, 
             const T* x, 
             uint32_t* mask,
             T* y, 
             Context* context);

template <typename T, class Context>
void DropoutGrad(const int count, 
                 T prob, 
                 T scale, 
                 const T* dy, 
                 const uint32_t* mask,
                 T* dx);

/******************** activation.elu ********************/

template <typename T, class Context>
void Elu(const int count, const T* x, const float alpha, T* y);

template <typename T, class Context>
void EluGrad(const int count,
             const T* dy,
             const T* y,
             const float alpha,
             T* dx);

/******************** activation.relu ********************/

template <typename T, class Context>
void Relu(const int count, const T* x, const float slope, T* y);

template <typename T, class Context>
void ReluGrad(const int count,
              const T* dy,
              const T* y,
              const float slope,
              T* dx);

/******************** activation.sigmoid ********************/

template <typename T, class Context>
void Sigmoid(const int count, const T* x, T* y);

template <typename T, class Context>
void SigmoidGrad(const int count, const T* dy, const T* y, T* dx);

/******************** activation.softmax ********************/

template <typename T, class Context>
void Softmax(const int count, 
             const int classes, 
             const int outer_dim, 
             const int inner_dim, 
             const T* sum_multiplier, 
             const T* x, 
             T* scale, 
             T* y, 
             Context* context);

template <typename T, class Context>
void SoftmaxGrad(const int count, 
                 const int classes, 
                 const int outer_dim, 
                 const int inner_dim,
                 const T* sum_multiplier, 
                 const T* dy, 
                 const T* y, 
                 T* scale, 
                 T* dx);

/******************** activation.tanh ********************/

template <typename T, class Context>
void Tanh(const int count, const T* x, T* y);

template <typename T, class Context>
void TanhGrad(const int count, const T* dy, const T* y, T* dx);

/******************** arithmetic.bias_add ********************/

template <typename T, class Context>
void BiasAdd(const int count, 
             const int outer_dim, 
             const int dim, 
             const int inner_dim,
             const string& format, 
             const T* bias, 
             const T* bias_multiplier, 
             T* y);

/******************** arithmetic.clip ********************/

template <typename T, class Context>
void Clip(const int count, 
          const float low, 
          const float high,
          const T* x, 
          T* mask, 
          T* y);

/******************** arithmetic.scale ********************/

template <typename T, class Context>
void Scale(const int axis, 
           Tensor* x, 
           Tensor* gamma, 
           Tensor* beta, 
           Tensor* BMul, 
           Tensor* y);

template <typename T, class Context>
void ScaleGrad(const int axis, Tensor* dy, Tensor* gamma, Tensor* dx);

/******************** cast.float2half ********************/

template <typename T, class Context>
void Float2Half(const int count, const float* x, float16* y);

/******************** control_flow.compare ********************/

template <typename T, class Context>
void Equal(const int count, const T* a, const T* b, T* y);

/******************** loss.l1_loss ********************/

template <typename T, class Context>
void AbsGrad(const int count, const T* dy, T* dx);

/******************** loss.sigmoid_cross_entropy ********************/

template <typename T, class Context>
void SigmoidCrossEntropy(const int count, const T* x, const T* target, T* loss);

/******************** loss.smooth_l1_loss ********************/

template <typename T, class Context>
void SmoothL1(const int count, const float sigma2, const T* x,  T* y);

template <typename T, class Context>
void SmoothL1Grad(const int count, const float sigma2, const T* dy, T* dx);

/******************** loss.softmax_cross_entropy ********************/

template <typename T, class Context>
void SoftmaxCrossEntropy(const int count, const T* prob, const T* target, T* loss);

/******************** loss.sparse_softmax_cross_entropy ********************/

template <typename T, class Context>
void SparseSoftmaxCrossEntropy(const int count, 
                               const int classes, 
                               const int outer_dim, 
                               const int inner_dim, 
                               const T* prob, 
                               const T* labels, 
                               T* loss, 
                               T* valid,
                               Tensor* ignore);

template <typename T, class Context>
void SparseSoftmaxCrossEntropyGrad(const int count,
                                   const int classes, 
                                   const int outer_dim, 
                                   const int inner_dim, 
                                   const T* prob, 
                                   const T* labels,
                                   T* valid, 
                                   Tensor* ignore, 
                                   T* dx);

/******************** loss.sparse_softmax_focal_loss ********************/

template <typename T, class Context>
void SparseSoftmaxFocalLoss(const int count, 
                            const int classes, 
                            const int outer_dim, 
                            const int inner_dim, 
                            const float pos_alpha,
                            const float neg_alpha,
                            const float gamma,
                            const int neg_id,
                            const T* prob, 
                            const T* labels,
                            T* scale,
                            T* loss, 
                            T* valid,
                            Tensor* ignore);

template <typename T, class Context>
void SparseSoftmaxFocalLossGrad(const int count,
                                const int classes, 
                                const int outer_dim, 
                                const int inner_dim,
                                const float gamma,
                                const int neg_id,
                                const float eps,
                                const T* scale,
                                const T* prob,
                                const T* labels,
                                T* valid, 
                                Tensor* ignore, 
                                T* dx);

/******************** misc.memory_data ********************/

template <typename Tx, typename Ty, class Context>
void MemoryData(const int count, 
                const int num, 
                const int channels,
                const int height, 
                const int width, 
                const Tx* x, 
                Ty* y);

/******************** ndarray.arange ********************/

template <typename T, class Context>
void Arange(const int count,
            const int start,
            const int step,
            T* y);

/******************** ndarray.argmax ********************/

template <typename T, class Context>
void Argmax(const int count, 
            const int axis_dim,
            const int inner_dim, 
            const int top_k, 
            const T* x, 
            T* y);

/******************** ndarray.argmin ********************/

template <typename T, class Context>
void Argmin(const int count, 
            const int axis_dim,
            const int inner_dim, 
            const int top_k, 
            const T* x, 
            T* y);

/******************** ndarray.at ********************/

template <typename T, class Context>
void CanonicalAxis(const int count, const int dim, T* y);

template <typename T, class Context>
void At(const int count, 
        const int outer_dim, 
        const int inner_dim,
        const int x_slice_dim, 
        const int y_slice_dim, 
        const T* indices, 
        const T* x,
        T* y,
        Context* context);

template <typename T, class Context>
void AtGrad(const int count, 
            const int outer_dim, 
            const int inner_dim,
            const int x_slice_dim, 
            const int y_slice_dim, 
            const T* indices,
            const T* dy, 
            T* dx, 
            Context* context);

/******************** ndarray.concat ********************/

template <typename T, class Context>
void Concat(const int count, 
            const int outer_dim, 
            const int inner_dim,
            const int x_concat_dim, 
            const int y_concat_dim, 
            const int concat_offset,
            const T* x, 
            T* y, 
            Context* context);

template <typename T, class Context>
void ConcatGrad(const int count, 
                const int outer_dim,
                const int inner_dim,
                const int x_concat_dim, 
                const int y_concat_dim, 
                const int concat_offset,
                const T* dy, 
                T* dx, 
                Context* context);

/******************** ndarray.crop ********************/

template <typename T, class Context>
void Crop2D(vector<TIndex> idxs,
            const vector<TIndex>& offsets,
            const int cur_dim,
            Tensor* x, 
            Tensor* y, 
            Context* context);

template <typename T, class Context>
void Crop2DGrad(vector<TIndex> idxs,
                const vector<TIndex>& offsets,
                const int cur_dim,
                Tensor* dy,
                Tensor* dx,
                Context* context);

/******************** ndarray.one_hot ********************/

template <typename T, class Context>
void OneHot(const int count, 
            const int depth, 
            const int on_value, 
            const T* x, 
            T* y);

/******************** ndarray.reduce ********************/

template <typename T, class Context>
void Sum(const int count, 
         const int axis_dim, 
         const int inner_dim,
         const T* x, 
         T* y);

template <typename T, class Context>
void SumGrad(const int count, 
             const int axis_dim, 
             const int inner_dim,
             const T coeff, 
             const T* dy, 
             T* dx);

/******************** ndarray.repeat ********************/

template <typename T, class Context>
void Repeat(const int count,
            const int outer_dim,
            const int dim,
            const int inner_dim,
            const int repeats,
            const T* x,
            T* y,
            Context* context);

template <typename T, class Context>
void RepeatGrad(const int count,
                const int outer_dim,
                const int dim,
                const int inner_dim,
                const int repeats,
                const T* dy,
                T* dx,
                Context* context);

/******************** ndarray.slice ********************/

template <typename T, class Context>
void Slice(const int count, 
           const int outer_dim, 
           const int inner_dim,
           const int x_slice_dim, 
           const int y_slice_dim, 
           const int slice_offset,
           const T* x, 
           T* y, 
           Context* context);

template <typename T, class Context>
void SliceGrad(const int count, 
               const int outer_dim, 
               const int inner_dim,
               const int x_slice_dim, 
               const int y_slice_dim, 
               const int slice_offset,
               const T* dy, 
               T* x, 
               Context* context);

/******************** ndarray.tile ********************/

template <typename T, class Context>
void Tile(const int count, 
          const int outer_dim, 
          const int ex_inner_dim,
          const int multiple, 
          const T* x, 
          T* y, 
          Context* context);

template <typename T, class Context>
void TileGrad(const int count, 
              const int outer_dim, 
              const int ex_inner_dim,
              const int multiple, 
              const T* dy, 
              T* dx, 
              Context* context);

/******************** ndarray.transpose ********************/

template <typename T, class Context>
void Transpose(const int count, 
               const int ndim, 
               const int* order, 
               const int* old_steps, 
               const int* new_steps,
               const T* x, 
               T* y);

template <typename T, class Context>
void TransposeGrad(const int count, 
                   const int ndim, 
                   const int* order,
                   const int* old_steps,
                   const int* new_steps, 
                   const T* dy, 
                   T* dx);

/******************** recurrent.lstm_uint ********************/

template <typename T, class Context>
void LSTMUnit(const int count, 
              const int num, 
              const int channels,
              const T* c_1, 
              const T* x, 
              const T* cont,
              T* x_act, 
              T* c, 
              T* h);

template <typename T, class Context>
void LSTMUnitGrad(const int count, 
                  const int num, 
                  const int channels,
                  const T* c_1, 
                  const T* x_act, 
                  const T* c, 
                  const T* dc, 
                  const T* dh, 
                  T* dc_1, 
                  T* dx);

/******************** update.adam_update ********************/

template <typename T, class Context>
void AdamUpdate(Tensor* x, 
                Tensor* m, 
                Tensor* v, 
                Tensor* t,
                const float beta1, 
                const float beta2, 
                const float eps, 
                const float lr);

/******************** update.nesterov_update ********************/

template <typename T, class Context>
void NesterovUpdate(const int count,
                    T* x,
                    T* h,
                    Tensor* t,
                    const float momentum,
                    const float lr,
                    Context* ctx);

/******************** update.rmsprop_update ********************/

template <typename T, class Context>
void RMSPropUpdate(const int count,
                   T* x,
                   T* h,
                   Tensor* t,
                   const float decay,
                   const float eps,
                   const float lr);

/******************** vision.bilinear_resize ********************/

template <typename T, class Context>
void BilinearResize(const int count, 
              const int num, 
              const int channels,
              const int h_in,
              const int w_in, 
              const int h_out, 
              const int w_out, 
              const T* x, 
              T* y);

template <typename T, class Context>
void BilinearResizeGrad(const int count,
                        const int num, 
                        const int channels,
                        const int h_in, 
                        const int w_in, 
                        const int h_out, 
                        const int w_out,
                        const T* dy, 
                        T* dx);

/******************** vision.conv ********************/

template <typename T, class Context>
void Im2Col(const int channels, 
            const int height, 
            const int width,
            const int kernel_h, 
            const int kernel_w, 
            const int stride_h, 
            const int stride_w, 
            const int pad_h,
            const int pad_w,
            const int dilation_h, 
            const int dilation_w, 
            const T* im,
            T* col);

template <typename T, class Context>
void Col2Im(const int channels, 
            const int height, 
            const int width,
            const int kernel_h, 
            const int kernel_w, 
            const int stride_h, 
            const int stride_w, 
            const int pad_h,
            const int pad_w,
            const int dilation_h, 
            const int dilation_w,
            const T* col,
            T* im);

/******************** vision.nn_resize ********************/

template <typename T, class Context>
void NNResize(const int count, 
              const int num, 
              const int channels,
              const int h_in,
              const int w_in, 
              const int h_out, 
              const int w_out, 
              const T* x, 
              T* y);

template <typename T, class Context>
void NNResizeGrad(const int count, 
                  const int num, 
                  const int channels,
                  const int h_in, 
                  const int w_in, 
                  const int h_out, 
                  const int w_out,
                  const T* dy, 
                  T* dx);

/******************** vision.pooling ********************/

template <typename T, class Context>
void MAXPooling(const int count, 
                const int num, 
                const int channels, 
                const int height, 
                const int width, 
                const int pool_height, 
                const int pool_width,
                const int kernel_h, 
                const int kernel_w,
                const int stride_h,
                const int stride_w,
                const int pad_h, 
                const int pad_w,
                const T* x, 
                int* mask, 
                T* y);

template <typename T, class Context>
void AVEPooling(const int count, 
                const int num, 
                const int channels,
                const int height, 
                const int width, 
                const int pool_height, 
                const int pool_width,
                const int kernel_h, 
                const int kernel_w, 
                const int stride_h, 
                const int stride_w, 
                const int pad_h,
                const int pad_w,
                const T* x, 
                T* y);

template <typename T, class Context>
void MAXPoolingGrad(const int count, 
                    const int num, 
                    const int channels,
                    const int height, 
                    const int width, 
                    const int pool_height, 
                    const int pool_width,
                    const int kernel_h, 
                    const int kernel_w, 
                    const int stride_h, 
                    const int stride_w, 
                    const int pad_h,
                    const int pad_w,
                    const T* dy, 
                    const int* mask, 
                    T* dx);

template <typename T, class Context>
void AVEPoolingGrad(const int count, 
                    const int num, 
                    const int channels,
                    const int height, 
                    const int width, 
                    const int pool_height, 
                    const int pool_width,
                    const int kernel_h, 
                    const int kernel_w, 
                    const int stride_h, 
                    const int stride_w, 
                    const int pad_h,
                    const int pad_w,
                    const T* dy, 
                    T* dx);

/******************** vision.roi_pooling ********************/

template <typename T, class Context>
void ROIPooling(const float spatial_scale, 
                const int pool_h,
                const int pool_w,
                Tensor* x,
                Tensor* roi,
                Tensor* mask,
                Tensor* y);

template <typename T, class Context>
void ROIPoolingGrad(const float spatial_scale, 
                    const int pool_h, 
                    const int pool_w,
                    Tensor* dy,
                    Tensor* roi,
                    Tensor* mask,
                    Tensor* dx);

/******************** vision.roi_align ********************/

template <typename T, class Context>
void ROIAlign(const float spatial_scale, 
              const int pool_h, 
              const int pool_w,
              Tensor* x,
              Tensor* roi,
              Tensor* mask_h,
              Tensor* mask_y,
              Tensor* y);

template <typename T, class Context>
void ROIAlignGrad(const float spatial_scale, 
                  const int pool_h, 
                  const int pool_w,
                  Tensor* dy,
                  Tensor* roi,
                  Tensor* mask_h,
                  Tensor* mask_y,
                  Tensor* dx);

}    // namespace kernel

}    // namepsace dragon

#endif    // DRAGON_UTILS_OP_KERNEL_H_