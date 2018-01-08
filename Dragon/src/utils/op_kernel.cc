#include <algorithm>
#include <functional>

#include "core/tensor.h"
#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"
#include "utils/sse_alternative.h"
#include "utils/math_functions.h"

bool judge(int a, int b)  { return unsigned(a) < unsigned(b); }

namespace dragon {

namespace kernel {
  
template<> void Empty<float, CPUContext>() {}

/******************** activation.dropout ********************/

template<> void Dropout<float, CPUContext>(const int count, 
                                           float prob, 
                                           float scale, 
                                           const float* x, 
                                           uint32_t* mask,
                                           float* y, 
                                           CPUContext* context) {
    uint32_t thresh = static_cast<uint32_t>(UINT_MAX * prob);
    math::RandomBernoulli<float, CPUContext>(count, 1 - prob, mask);
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) y[i] = x[i] * mask[i] * scale;
}

template<> void DropoutGrad<float, CPUContext>(const int count, 
                                               float prob, 
                                               float scale, 
                                               const float* dy, 
                                               const uint32_t* mask,
                                               float* dx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) dx[i] = dy[i] * mask[i] * scale;
}

/******************** activation.elu ********************/

template<> void Elu<float, CPUContext>(const int count,
                                       const float* x,
                                       const float alpha,
                                       float* y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = std::max(x[i], float(0))
            + alpha * (std::exp(std::min(x[i], float(0))) - float(1));
    }
}

template<> void EluGrad<float, CPUContext>(const int count,
                                           const float* dy,
                                           const float* y,
                                           const float alpha,
                                           float* dx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        dx[i] = dy[i] * ((y[i] > 0) + (alpha + y[i]) * (y[i] <= 0));
    }
}

/******************** activation.prelu ********************/

template<> void PRelu<float, CPUContext>(const int count,
                                         const int channels,
                                         const int dim,
                                         const bool channel_shared,
                                         const string& data_format,
                                         const float* x,
                                         const float* w,
                                         float* y) {
    if (channel_shared) {
#ifdef WITH_OMP
        #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
        for (int i = 0; i < count; ++i) {
            y[i] = std::max(x[i], float(0)) + w[0] * std::min(x[i], float(0));
        }
    } else {
        if (data_format == "NCHW") {
#ifdef WITH_OMP
            #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
            for (int i = 0; i < count; ++i) {
                int c = (i / dim) % channels;
                y[i] = std::max(x[i], float(0)) + w[c] * std::min(x[i], float(0));
            }
        } else if (data_format == "NHWC") {
#ifdef WITH_OMP
            #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
            for (int i = 0; i < count; ++i) {
                int c = i % channels;
                y[i] = std::max(x[i], float(0)) + w[c] * std::min(x[i], float(0));
            }
        } else LOG(FATAL) << "Unknown data format: " << data_format;
    }
}

template<> void PReluGrad<float, CPUContext>(const int count,
                                             const int channels, 
                                             const int dim,
                                             const bool channel_shared,
                                             const string& data_format,
                                             const float* dy,
                                             const float* x,
                                             const float* w,
                                             float* dx) {
    if (channel_shared) {
#ifdef WITH_OMP
        #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
        for (int i = 0; i < count; ++i) {
            dx[i] = dy[i] * ((x[i] > 0) + w[0] * (x[i] <= 0));
        }
    } else {
        if (data_format == "NCHW") {
#ifdef WITH_OMP
            #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
            for (int i = 0; i < count; ++i) {
                int c = (i / dim) % channels;
                dx[i] = dy[i] * ((x[i] > 0) + w[c] * (x[i] <= 0));
            }
        } else if (data_format == "NHWC") {
#ifdef WITH_OMP
            #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
            for (int i = 0; i < count; ++i) {
                int c = i % channels;
                dx[i] = dy[i] * ((x[i] > 0) + w[c] * (x[i] <= 0));
            }
        } else LOG(FATAL) << "Unknown data format: " << data_format;
    }
}

template<> void PReluWGrad<float, CPUContext>(const int rows,
                                              const int row_offset,
                                              const int channels,
                                              const int dim,
                                              const bool channel_shared,
                                              const string& data_format,
                                              const float* dy,
                                              const float* x,
                                              const float* multiplier,
                                              float* bcast_dw,
                                              float* dw) {
    const int cdim = channels * dim;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(cdim))
#endif
    for (int i = 0; i < cdim; ++i) {
        bcast_dw[i] = dy[i] * x[i] * (x[i] <= 0);
        for (int n = 1; n < rows; n++) {
            const int cur_idx = i + n * row_offset;
            bcast_dw[i] += dy[cur_idx] * x[cur_idx] * (x[cur_idx] <= 0);
        }
    }
    if (channel_shared) {
        float w_sum = math::Dot<float, CPUContext>(channels * dim, bcast_dw, multiplier);
        math::AddScalar<float, CPUContext>(1, w_sum, dw);
    } else {
        if (data_format == "NCHW") {
            math::Gemv<float, CPUContext>(CblasNoTrans, channels, dim,
                                                                  1.0,
                                                 bcast_dw, multiplier,
                                                                  1.0,
                                                                   dw);
        } else if (data_format == "NHWC") {
            math::Gemv<float, CPUContext>(CblasTrans, dim, channels,
                                                                1.0,
                                               bcast_dw, multiplier,
                                                                1.0,
                                                                 dw);

        } else LOG(FATAL) << "Unknown data format: " << data_format;
    }
}

/******************** activation.relu ********************/

template<> void Relu<float, CPUContext>(const int count, 
                                        const float* x, 
                                        const float slope, 
                                        float* y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = std::max(x[i], 0.f) + slope * std::min(x[i], 0.f);
    }
}

template<> void ReluGrad<float, CPUContext>(const int count, 
                                            const float* dy, 
                                            const float* y, 
                                            const float slope, 
                                            float* dx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        dx[i] = dy[i] * ((y[i] > 0) + slope * (y[i] <= 0));
    }
}

/******************** activation.selu ********************/

template<> void SElu<float, CPUContext>(const int count,
                                        const float* x,
                                        float* y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = 1.0507 * std::max(x[i], float(0))
             + 1.7581 * (std::exp(std::min(x[i], float(0))) - float(1));
    }
}

template<> void SEluGrad<float, CPUContext>(const int count,
                                            const float* dy,
                                            const float* y,
                                            float* dx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        dx[i] = y[i] > 0 ? 1.0507 * dy[i] : (1.7581 + y[i]) * dy[i];
    }
}

/******************** activation.sigmoid ********************/

template <typename T>
T _sigmoid(T x) { return T(1) / (T(1) + exp(-x)); }

template<> void Sigmoid<float, CPUContext>(const int count, const float* x, float* y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i)  y[i] = _sigmoid<float>(x[i]);
}

template<> void SigmoidGrad<float, CPUContext>(const int count, 
                                               const float* dy, 
                                               const float* y, 
                                               float* dx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        dx[i] = dy[i] * y[i] * (1 - y[i]);
    }
}

/******************** activation.softmax ********************/

template<> void Softmax<float, CPUContext>(const int count, 
                                           const int classes, 
                                           const int outer_dim, 
                                           const int inner_dim,
                                           const float* sum_multiplier, 
                                           const float* x, 
                                           float* scale, 
                                           float* y, 
                                           CPUContext* context) {
    const int dim = count / outer_dim;
    for (int i = 0; i < outer_dim; ++i) {
        context->Copy<float, CPUContext, CPUContext>(inner_dim, scale, x + i*dim);
        for (int j = 0; j < classes; ++j) {
            for (int k = 0; k < inner_dim; k++)
                scale[k] = std::max(scale[k], x[i * dim + j * inner_dim + k]);
        }
        math::Gemm<float, CPUContext>(CblasNoTrans, CblasNoTrans,
                                           classes, inner_dim, 1,
                                                            -1.0,
                                           sum_multiplier, scale,
                                                             1.0,
                                                               y);
        math::Exp<float, CPUContext>(dim, y, y);
        math::Gemv<float, CPUContext>(CblasTrans, classes, inner_dim,
                                                                 1.0,
                                                   y, sum_multiplier,
                                                                 0.0,
                                                               scale);
        for (int j = 0; j < classes; ++j) {
            math::Div<float, CPUContext>(inner_dim, y, scale, y);
            y += inner_dim;
        }
    }
}

template<> void SoftmaxGrad<float, CPUContext>(const int count, 
                                               const int classes, 
                                               const int outer_dim, 
                                               const int inner_dim,
                                               const float* sum_multiplier, 
                                               const float* dy, 
                                               const float* y, 
                                               float* scale, 
                                               float* dx) {
    const int dim = count / outer_dim;
    for (int i = 0; i < outer_dim; ++i) {
        for (int k = 0; k < inner_dim; ++k)
            scale[k] = math::StridedDot<float, CPUContext>(classes,
                                       dx + i * dim + k, inner_dim,
                                          y + i*dim + k, inner_dim);
         math::Gemm<float, CPUContext>(CblasNoTrans, CblasNoTrans,
                                            classes, inner_dim, 1,
                                                             -1.0,
                                            sum_multiplier, scale,
                                                              1.0,
                                                       dx + i*dim);
    }
    math::Mul<float, CPUContext>(count, dx, y, dx);
}

/******************** activation.tanh ********************/

template<> void Tanh<float, CPUContext>(const int count, const float* x, float* y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = std::tanh(x[i]);
    }
}

template<> void TanhGrad<float, CPUContext>(const int count, 
                                            const float* dy, 
                                            const float* y, 
                                            float* dx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        dx[i] = dy[i] * (1 - y[i] * y[i]);
    }
}

/******************** arithmetic.bias_add ********************/

template<> void BiasAdd<float, CPUContext>(const int count,
                                           const int outer_dim,
                                           const int dim,
                                           const int inner_dim,
                                           const string& data_format,
                                           const float* bias,
                                           const float* bias_multiplier,
                                           float* y) {
    const int y_offset = dim * inner_dim;
    for (int n = 0; n < outer_dim; ++n) {
        if (data_format == "NCHW") {
            math::Gemm<float, CPUContext>(CblasNoTrans, CblasNoTrans,
                                                   dim, inner_dim, 1,
                                                                 1.0,
                                               bias, bias_multiplier,
                                                                 1.0,
                                                                   y);
        } else if (data_format == "NHWC") {
            math::Gemm<float, CPUContext>(CblasNoTrans, CblasNoTrans,
                                                   inner_dim, dim, 1,
                                                                 1.0,
                                               bias_multiplier, bias,
                                                                 1.0,
                                                                   y);
        } else LOG(FATAL) << "Unknown data format: " << data_format;
        y += y_offset;
    }
}

/******************** arithmetic.clip ********************/

template <> void Clip<float, CPUContext>(const int count,
                                         const float low,
                                         const float high,
                                         const float* x,
                                         float* mask,
                                         float* y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        mask[i] = 1.0;
        if (x[i] < low || x[i] > high) mask[i] = 0.0;
        y[i] = std::max(low, std::min(x[i], high));
    }
}

/******************** arithmetic.scale ********************/

template<> void Scale<float, CPUContext>(const int axis, 
                                         Tensor* x, 
                                         Tensor* gamma, 
                                         Tensor* beta, 
                                         Tensor* BMul, 
                                         Tensor* y) {
    int outer_dim = x->count(0, axis);
    int inner_dim = x->count(axis + gamma->ndim());
    int scale_dim = gamma->count();
    auto* Xdata = x->data<float, CPUContext>();
    auto* Ydata = y->mutable_data<float, CPUContext>();
    auto* Sdata = gamma->data<float, CPUContext>();
    auto* Bdata = beta != nullptr ? 
                          beta->data<float, CPUContext>() : 
                          nullptr;
    auto* BMul_data = BMul != nullptr ?
                      BMul->data<float, CPUContext>() : 
                      nullptr;
    for (int n = 0; n < outer_dim; ++n) {
        for (int d = 0; d < scale_dim; ++d) {
            const float factor = Sdata[d];
            math::Scale<float, CPUContext>(inner_dim, factor, Xdata, Ydata);
            Xdata += inner_dim; 
            Ydata += inner_dim;
        }
    }
    if (Bdata != nullptr) {
        int dim = scale_dim * inner_dim;
        Ydata = y->mutable_data<float, CPUContext>();
        for (int n = 0; n < outer_dim; ++n) {
            math::Gemm<float, CPUContext>(CblasNoTrans, CblasNoTrans,
                                             scale_dim, inner_dim, 1,
                                                                 1.0,
                                                    Bdata, BMul_data,
                                                                 1.0,
                                                               Ydata);
             Ydata += dim;
        }
    }
}

template<> void Scale<float16, CPUContext>(const int axis, 
                                           Tensor* x, 
                                           Tensor* gamma, 
                                           Tensor* beta, 
                                           Tensor* BMul, 
                                           Tensor* y) {
    LOG(FATAL) << "float16 is unsupported for CPUContext.";
}

template <> void ScaleGrad<float, CPUContext>(const int axis, 
                                              Tensor* dy, 
                                              Tensor* gamma, 
                                              Tensor* dx) {
    int outer_dim = dx->count(0, axis);
    int inner_dim = dx->count(axis + gamma->ndim());
    int scale_dim = gamma->count();
    auto* dYdata = dy->data<float, CPUContext>();
    auto* dXdata = dx->mutable_data<float, CPUContext>();
    auto* Sdata = gamma->data<float, CPUContext>();
    for (int n = 0; n < outer_dim; ++n) {
        for (int d = 0; d < scale_dim; ++d) {
            const float factor = Sdata[d];
            math::Scale<float, CPUContext>(inner_dim, factor, dYdata, dXdata);
            dYdata += inner_dim; dXdata += inner_dim;
        }
    }
}

/******************** control_flow.compare ********************/

template <> void Equal<float, CPUContext>(const int count,
                                          const float* a,
                                          const float* b,
                                          float* y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i)
        y[i] = fabs(a[i] - b[i]) < FLT_EPSILON ? 1.0 : 0.0;
}

/******************** loss.l1_loss ********************/

template<> void AbsGrad<float, CPUContext>(const int count, const float* dy, float* dx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const float val = dy[i];
        //  val > 0: 1 | val == 0: 0 | val < 0: -1
        dx[i] = (val > float(0)) - (val < float(0));
    }
}

/******************** loss.sigmoid_cross_entropy ********************/

template <> void SigmoidCrossEntropy<float, CPUContext>(const int count, 
                                                        const float* x, 
                                                        const float* target, 
                                                        float* loss) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        loss[i] = std::log(1 + std::exp(x[i] - 2 * x[i] * (x[i] >= 0)))
                      + x[i] * ((x[i] >= 0) - target[i]);
    }
}

/******************** loss.smooth_l1_loss ********************/

template<> void SmoothL1<float, CPUContext>(const int count, 
                                            const float sigma2, 
                                            const float* x, 
                                            float* y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const float val = x[i];
        const float abs_val = abs(val);
        if (abs_val < 1.0 / sigma2) y[i] = 0.5 * val * val * sigma2;
        else y[i] = abs_val - 0.5 / sigma2;
    }
}

template<> void SmoothL1Grad<float, CPUContext>(const int count, 
                                                const float sigma2, 
                                                const float* dy, 
                                                float* dx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const float val = dy[i];
        const float abs_val = abs(val);
        if (abs_val < 1.0 / sigma2) dx[i] = val * sigma2;
        //  val > 0: 1 | val == 0: 0 | val < 0: -1
        else dx[i] = (val > float(0)) - (val < float(0));
    }
}

/******************** loss.softmax_cross_entropy ********************/

template <> void SoftmaxCrossEntropy<float, CPUContext>(const int count, 
                                                        const float* prob, 
                                                        const float* target, 
                                                        float* loss) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        loss[i] = - target[i] * std::log(std::max(prob[i], FLT_MIN));
    }
}

/******************** loss.sparse_softmax_cross_entropy ********************/

template <> void SparseSoftmaxCrossEntropy<float, CPUContext>(const int count, 
                                                              const int classes, 
                                                              const int outer_dim, 
                                                              const int inner_dim,
                                                              const float* prob, 
                                                              const float* labels, 
                                                              float* loss, 
                                                              float* valid, 
                                                              Tensor* ignore) {
    const int* ignores = ignore->count() > 0 ? 
                         ignore->data<int, CPUContext>() : nullptr;
    const int dim = count / outer_dim;
    for (int i = 0; i < outer_dim; ++i) {
        for (int j = 0; j < inner_dim; ++j) {
            const int idx = i * inner_dim + j;
            const int label = labels[idx];
            int k;
            for (k = 0; k < ignore->count(); ++k) {
                if (label == ignores[k]) {
                    loss[idx] = valid[idx] = 0;
                    break;
                }
            }
            if (k == ignore->count()) {
                float labeled_prob = prob[i * dim + label * inner_dim + j];
                loss[idx] = -std::log(std::max(labeled_prob, FLT_MIN));
                valid[idx] = 1;
            }
        }
    }
}

template<> void SparseSoftmaxCrossEntropyGrad<float, CPUContext>(const int count,
                                                                 const int classes,
                                                                 const int outer_dim,
                                                                 const int inner_dim,
                                                                 const float* prob,
                                                                 const float* labels,
                                                                 float* valid,
                                                                 Tensor* ignore,
                                                                 float* dx) {
    int dim = count / outer_dim;
    const int* ignores = ignore->count() > 0 ?
                         ignore->data <int, CPUContext>() : nullptr;
    valid[0] = 0;
    for (int i = 0; i < outer_dim; ++i) {
        for (int j = 0; j < inner_dim; ++j) {
            const int label = labels[i * inner_dim + j];
            int k;
            for (k = 0; k < ignore->count(); ++k)
                if (label == ignores[k]) break;
            if (k != ignore->count()) {
                for (int c = 0; c < classes; ++c)
                    dx[i * dim + c * inner_dim + j] = 0;
            } else {
                dx[i * dim + label * inner_dim + j] -= 1;
                valid[0]++;
            }
        }
    }
}

/******************** loss.sparse_softmax_focal_loss ********************/

template <> void SparseSoftmaxFocalLoss<float, CPUContext>(const int count,
                                                           const int classes,
                                                           const int outer_dim,
                                                           const int inner_dim,
                                                           const float pos_alpha,
                                                           const float neg_alpha,
                                                           const float gamma,
                                                           const int neg_id,
                                                           const float* prob,
                                                           const float* labels,
                                                           float* scale,
                                                           float* loss,
                                                           float* valid,
                                                           Tensor* ignore) {
    const int* ignores = ignore->count() > 0 ? 
                         ignore->data<int, CPUContext>() : nullptr;
    const int dim = count / outer_dim;

#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) 
        scale[i] = std::pow((1.0f - prob[i]), gamma);

    for (int i = 0; i < outer_dim; ++i) {
        for (int j = 0; j < inner_dim; ++j) {
            const int idx = i * inner_dim + j;
            const int label = labels[idx];
            int k;
            for (k = 0; k < ignore->count(); ++k) {
                if (label == ignores[k]) {
                    loss[idx] = valid[idx] = 0;
                    break;
                }
            }
            if (k == ignore->count()) {
                const int t_ = i * dim + label * inner_dim + j;
                float labeled_prob = std::max(labeled_prob, FLT_MIN);
                scale[t_] = label > neg_id ? pos_alpha * scale[t_] :
                                             neg_alpha * scale[t_];
                loss[idx] = -scale[t_] * std::log(labeled_prob);
                valid[idx] = label > neg_id ? 1 : 0;
            }
        }
    }
}

template<> void SparseSoftmaxFocalLossGrad<float, CPUContext>(const int count,
                                                              const int classes, 
                                                              const int outer_dim, 
                                                              const int inner_dim,
                                                              const float gamma,
                                                              const int neg_id,
                                                              const float eps,
                                                              const float* scale,
                                                              const float* prob, 
                                                              const float* labels, 
                                                              float* valid, 
                                                              Tensor* ignore, 
                                                              float* dx) {
    int dim = count / outer_dim;
    const int* ignores = ignore->count() > 0 ?
                         ignore->data <int, CPUContext>() : nullptr;
    valid[0] = 0;
    for (int i = 0; i < outer_dim; ++i) {
        for (int j = 0; j < inner_dim; ++j) {
            const int label = labels[i * inner_dim + j];
            int k;
            for (k = 0; k < ignore->count(); ++k)
                if (label == ignores[k]) break;
            if (k != ignore->count()) {
                for (int c = 0; c < classes; ++c)
                    dx[i * dim + c * inner_dim + j] = 0;
            } else {
                const int t_ = i * dim + label * inner_dim + j;
                float grad = -gamma * (scale[t_] / std::max((1.0f - prob[t_]), eps))
                                    * std::log(std::max(prob[t_], FLT_MIN)) 
                                    * prob[t_] + scale[t_];
                for (int c = 0; c < classes; ++c) {
                    const int i_ = i * dim + c * inner_dim + j;
                    if (c == label) {
                        dx[i_] = grad * (prob[t_] - 1);
                    } else {
                        dx[i_] = grad * prob[i_];
                    }
                }
                if (label > neg_id) valid[0]++;
            }
        }
    }
}

/******************** misc.image_data ********************/

template <typename Tx, typename Ty>
void _ImageData_NCHW(const int N, const int C,
                     const int H, const int W,
                     const float* mean_values,
                     const float* std_values,
                     const Tx* x,
                     Ty* y) {
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                const int NH = n * H + h;
                for (int w = 0; w < W; ++w) {
                    Ty raw_value = x[(NH * W + w) * C + c];
                    if (mean_values != nullptr) raw_value -= mean_values[c];
                    if (std_values != nullptr) raw_value /= std_values[c];
                    *(y++) = raw_value;
                }
            }
        }
    }
}

template <typename Tx, typename Ty>
void _ImageData_NHWC(const int N, const int C,
                     const int H, const int W,
                     const float* mean_values,
                     const float* std_values,
                     const Tx* x,
                     Ty* y) {
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                for (int c = 0; c < C; ++c) {
                    Ty raw_value = *(x++);
                    if (mean_values != nullptr) raw_value -= mean_values[c];
                    if (std_values != nullptr) raw_value /= std_values[c]; 
                    *(y++) = raw_value;
                }
            }
        }
    }
}

template <> void ImageData<float, float, CPUContext>(const int count,
                                                     const int N, const int C,
                                                     const int H, const int W,
                                                     const float* mean_values,
                                                     const float* std_values,
                                                     const string& data_format,
                                                     const float* x,
                                                     float* y) {
    if (data_format == "NCHW") {
        _ImageData_NCHW<float, float>(N, C, H, W,
                                     mean_values,
                                      std_values,
                                               x,
                                               y);
    } else if (data_format == "NHWC") {
        _ImageData_NHWC<float, float>(N, C, H, W,
                                     mean_values,
                                      std_values,
                                               x,
                                               y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template <> void ImageData<uint8_t, float, CPUContext>(const int count,
                                                       const int N, const int C,
                                                       const int H, const int W,
                                                       const float* mean_values,
                                                       const float* std_values,
                                                       const string& data_format,
                                                       const uint8_t* x,
                                                       float* y) {
    if (data_format == "NCHW") {
        _ImageData_NCHW<uint8_t, float>(N, C, H, W,
                                     mean_values,
                                      std_values,
                                               x,
                                               y);
    } else if (data_format == "NHWC") {
        _ImageData_NHWC<uint8_t, float>(N, C, H, W,
                                     mean_values,
                                      std_values,
                                               x,
                                               y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template <> void ImageData<float, float16, CPUContext>(const int count,
                                                       const int N, const int C,
                                                       const int H, const int W,
                                                       const float* mean_values,
                                                       const float* std_values,
                                                       const string& data_format,
                                                       const float* x,
                                                       float16* y) {
    LOG(FATAL) << "float16 is unsupported for CPUContext.";
}

template <> void ImageData<uint8_t, float16, CPUContext>(const int count,
                                                         const int N, const int C,
                                                         const int H, const int W,
                                                         const float* mean_values,
                                                         const float* std_values,
                                                         const string& data_format,
                                                         const uint8_t* x,
                                                         float16* y) {
    LOG(FATAL) << "float16 is unsupported for CPUContext.";
}

/******************** ndarray.arange ********************/

template<> void Arange<float, CPUContext>(const int count,
                                          const int start,
                                          const int step,
                                          float* y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) y[i] = start + i * step;
}

template<> void Arange<int, CPUContext>(const int count,
                                        const int start,
                                        const int step,
                                        int* y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) y[i] = start + i * step;
}

/******************** ndarray.argmax ********************/

template<> void Argmax<float, CPUContext>(const int count, 
                                          const int axis_dim,
                                          const int inner_dim, 
                                          const int top_k, 
                                          const float* x,
                                          float* y) {
    vector<pair<float, int> > vec(axis_dim);
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < axis_dim; ++j) 
            vec[j] = std::make_pair(x[(i / inner_dim * axis_dim + j) * 
                                    inner_dim + i % inner_dim], j);
        std::partial_sort(vec.begin(), 
                          vec.begin() + top_k, 
                          vec.end(), 
                          std::greater< pair<float, int> >());
        for (int j = 0; j < top_k; ++j) 
            y[(i / inner_dim * top_k + j) * inner_dim + i % inner_dim] = vec[j].second;
    }
}

/******************** ndarray.argmin ********************/

template<> void Argmin<float, CPUContext>(const int count, 
                                          const int axis_dim,
                                          const int inner_dim, 
                                          const int top_k, 
                                          const float* x,
                                          float* y) {
    vector<pair<float, int> > vec(axis_dim);
#ifdef WITH_OMP
#pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < axis_dim; ++j) 
            vec[j] = std::make_pair(x[(i / inner_dim * axis_dim + j) * 
                                    inner_dim + i % inner_dim], j);
        std::partial_sort(vec.begin(), 
                          vec.begin() + top_k, 
                          vec.end());
        for (int j = 0; j < top_k; ++j) 
            y[(i / inner_dim * top_k + j) * inner_dim + i % inner_dim] = vec[j].second;
    }
}

/******************** ndarray.at ********************/

template <> void CanonicalAxis<int, CPUContext>(const int count, const int dim, int* y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) if (y[i] < 0) y[i] += dim;
}

template <typename T>
void _At(const int count,
         const int outer_dim,
         const int inner_dim,
         const int x_slice_dim,
         const int y_slice_dim,
         const int* indices,
         const T* x,
         T* y,
         CPUContext* ctx) {
    TIndex x_offset, y_offset, x_idx_offset, y_idx_offset;
    for (int i = 0; i < y_slice_dim; ++i) {
        y_idx_offset = i;
        x_idx_offset = indices[y_idx_offset];
        for (int n = 0; n < outer_dim; ++n) {
            x_offset = (n * x_slice_dim + x_idx_offset) * inner_dim;
            y_offset = (n * y_slice_dim + y_idx_offset) * inner_dim;
            ctx->Copy<T, CPUContext, CPUContext>(inner_dim,
                                                 y + y_offset,
                                                 x + x_offset);
        }
    }
}

template <> void At<float, CPUContext>(const int count,
                                       const int outer_dim,
                                       const int inner_dim,
                                       const int x_slice_dim,
                                       const int y_slice_dim,
                                       const int* indices,
                                       const float* x,
                                       float* y,
                                       CPUContext* ctx) {
    _At<float>(count, outer_dim, inner_dim,
                  x_slice_dim, y_slice_dim,
                       indices, x, y, ctx);
    
}

template <> void At<int, CPUContext>(const int count,
                                     const int outer_dim,
                                     const int inner_dim,
                                     const int x_slice_dim,
                                     const int y_slice_dim,
                                     const int* indices,
                                     const int* x,
                                     int* y,
                                     CPUContext* ctx) {
    _At<int>(count, outer_dim, inner_dim,
                x_slice_dim, y_slice_dim,
                     indices, x, y, ctx);
}

template <typename T>
void _AtGrad(const int count,
             const int outer_dim,
             const int inner_dim,
             const int x_slice_dim,
             const int y_slice_dim,
             const int* indices,
             const T* dy,
             T* dx) {
    TIndex x_offset, y_offset, x_idx_offset, y_idx_offset;
    for (int i = 0; i < y_slice_dim; ++i) {
        y_idx_offset = i;
        x_idx_offset = indices[y_idx_offset];
        for (int n = 0; n < outer_dim; ++n) {
            x_offset = (n * x_slice_dim + x_idx_offset) * inner_dim;
            y_offset = (n * y_slice_dim + y_idx_offset) * inner_dim;
            math::Add<T, CPUContext>(inner_dim,
                                     dy + y_offset,
                                     dx + x_offset,
                                     dx + x_offset);
        }
    }
}

template <> void AtGrad<float, CPUContext>(const int count,
                                           const int outer_dim,
                                           const int inner_dim,
                                           const int x_slice_dim,
                                           const int y_slice_dim,
                                           const int* indices,
                                           const float* dy,
                                           float* dx) {
    _AtGrad<float>(count, outer_dim, inner_dim,
                      x_slice_dim, y_slice_dim,
                              indices, dy, dx);
}

template <> void AtGrad<int, CPUContext>(const int count,
                                         const int outer_dim,
                                         const int inner_dim,
                                         const int x_slice_dim,
                                         const int y_slice_dim,
                                         const int* indices,
                                         const int* dy,
                                         int* dx) {
    _AtGrad<int>(count, outer_dim, inner_dim,
                    x_slice_dim, y_slice_dim,
                            indices, dy, dx);
}

/******************** ndarray.concat ********************/

template <> void Concat<float, CPUContext>(const int count, 
                                           const int outer_dim, 
                                           const int inner_dim,
                                           const int x_concat_dim, 
                                           const int y_concat_dim, 
                                           const int concat_offset,
                                           const float* x, 
                                           float* y, 
                                           CPUContext* context) {
    TIndex x_offset, y_offset;
    for (int n = 0; n < outer_dim; ++n) {
        x_offset = n * x_concat_dim * inner_dim;
        y_offset = (n * y_concat_dim + concat_offset) * inner_dim;
        context->Copy<float, CPUContext, CPUContext>(x_concat_dim * inner_dim,
                                                     y + y_offset,
                                                     x + x_offset);
    }
}

template <> void Concat<float16, CPUContext>(const int count,
                                             const int outer_dim,
                                             const int inner_dim,
                                             const int x_concat_dim,
                                             const int y_concat_dim,
                                             const int concat_offset,
                                             const float16* x,
                                             float16* y,
                                             CPUContext* context) {
    TIndex x_offset, y_offset;
    for (int n = 0; n < outer_dim; ++n) {
        x_offset = n * x_concat_dim * inner_dim;
        y_offset = (n * y_concat_dim + concat_offset) * inner_dim;
        context->Copy<float16, CPUContext, CPUContext>(x_concat_dim * inner_dim, 
                                                       y + y_offset, 
                                                       x + x_offset);
    }
}

template <> void ConcatGrad<float, CPUContext>(const int count, 
                                               const int outer_dim, 
                                               const int inner_dim,
                                               const int x_concat_dim, 
                                               const int y_concat_dim, 
                                               const int concat_offset,
                                               const float* dy, 
                                               float* dx, 
                                               CPUContext* context) {
    TIndex x_offset, y_offset;
    for (int n = 0; n < outer_dim; ++n) {
        x_offset = n * x_concat_dim * inner_dim;
        y_offset = (n * y_concat_dim + concat_offset) * inner_dim;
        context->Copy<float, CPUContext, CPUContext>(x_concat_dim * inner_dim,
                                                     dx + x_offset,
                                                     dy + y_offset);
    }
}

template <> void ConcatGrad<float16, CPUContext>(const int count, 
                                               const int outer_dim, 
                                               const int inner_dim,
                                               const int x_concat_dim, 
                                               const int y_concat_dim, 
                                               const int concat_offset,
                                               const float16* dy, 
                                               float16* dx, 
                                               CPUContext* context) {
    TIndex x_offset, y_offset;
    for (int n = 0; n < outer_dim; ++n) {
        x_offset = n * x_concat_dim * inner_dim;
        y_offset = (n * y_concat_dim + concat_offset) * inner_dim;
        context->Copy<float16, CPUContext, CPUContext>(x_concat_dim * inner_dim,
                                                       dx + x_offset,
                                                       dy + y_offset);
    }
}

/******************** ndarray.crop ********************/

template<> void Crop1D<float, CPUContext>(const int count,
                                          const int dim,
                                          const int ex_dim,
                                          const int inner_dim,
                                          const int start,
                                          const float* x,
                                          float* y, 
                                          CPUContext* context) {
    const int count_v2 = count / inner_dim;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count_v2))
#endif
    for (int idx = 0; idx < count_v2; ++idx) {
        const int ex_d = idx % ex_dim;
        const int o = idx / ex_dim;
        const float* x_ptr = x + (o * dim + ex_d + start) * inner_dim;
        float* y_ptr = y + (o * ex_dim + ex_d) * inner_dim;
        context->Copy<float, CPUContext, CPUContext>(inner_dim, y_ptr, x_ptr);
    }
}

template<> void Crop1DGrad<float, CPUContext>(const int count,
                                              const int dim,
                                              const int ex_dim,
                                              const int inner_dim,
                                              const int start,
                                              const int end,
                                              const float* dy,
                                              float* dx, 
                                              CPUContext* context) {
    const int count_v2 = count / inner_dim;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count_v2))
#endif
    for (int idx = 0; idx < count_v2; ++idx) {
        const int d = idx % dim;
        const int o = idx / dim;
        float* dx_ptr = dx + (o * dim + d) * inner_dim;
        if (d < start || d >= end) {
            for (int i = 0; i < inner_dim; ++i) dx_ptr[i] = 0;
        } else {
            const float* dy_ptr = dy + (o * ex_dim + d - start) * inner_dim;
            context->Copy<float, CPUContext, CPUContext>(inner_dim, dx_ptr, dy_ptr);
        }
    }
}

/******************** ndarray.pad ********************/

template <> void ConstPad1D<float, CPUContext>(const int count,
                                               const int dim,
                                               const int ex_dim,
                                               const int inner_dim,
                                               const int pad_l,
                                               const float value,
                                               const float* x,
                                               float* y,
                                               CPUContext* context) {
    const int count_v2 = count / inner_dim;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count_v2))
#endif
    for (int idx = 0; idx < count_v2; ++idx) {
        const int ex_d = idx % ex_dim;
        const int o = idx / ex_dim;
        const int d = ex_d - pad_l;
        float* y_ptr = y + (o * ex_dim + ex_d) * inner_dim;
        if (d < 0 || d >= dim) {
            for (int i = 0; i < inner_dim; ++i) y_ptr[i] = value;
        } else {
            const float* x_ptr = x + (o * dim + d) * inner_dim;
            context->Copy<float, CPUContext, CPUContext>(inner_dim, y_ptr, x_ptr);
        }
    }
}

template <> void ReflectPad1D<float, CPUContext>(const int count,
                                                 const int dim,
                                                 const int ex_dim,
                                                 const int inner_dim,
                                                 const int pad_l,
                                                 const float* x,
                                                 float* y,
                                                 CPUContext* context) {
    const int count_v2 = count / inner_dim;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count_v2))
#endif
    for (int idx = 0; idx < count_v2; ++idx) {
        const int ex_d = idx % ex_dim;
        const int o = idx / ex_dim;
        int d = ex_d - pad_l;
        d = std::max(d, -d);
        d = std::min(d, 2 * dim - d - 2);
        float* y_ptr = y + (o * ex_dim + ex_d) * inner_dim;
        if (d < 0 || d >= dim) {
            for (int i = 0; i < inner_dim; ++i) 
                y_ptr[i] = x[(o * dim + d) * inner_dim + i];
        } else {
            const float* x_ptr = x + (o * dim + d) * inner_dim;
            context->Copy<float, CPUContext, CPUContext>(inner_dim, y_ptr, x_ptr);
        }
    }
}

template <> void EdgePad1D<float, CPUContext>(const int count,
                                              const int dim,
                                              const int ex_dim,
                                              const int inner_dim,
                                              const int pad_l,
                                              const float* x,
                                              float* y, 
                                              CPUContext* context) {
    const int count_v2 = count / inner_dim;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count_v2))
#endif
    for (int idx = 0; idx < count_v2; ++idx) {
        const int ex_d = idx % ex_dim;
        const int o = idx / ex_dim;
        const int d = std::min(dim - 1, std::max(ex_d - pad_l, 0));
        float* y_ptr = y + (o * ex_dim + ex_d) * inner_dim;
        if (d < 0 || d >= dim) {
            for (int i = 0; i < inner_dim; ++i) 
                y_ptr[i] = x[(o * dim + d) * inner_dim + i];
        } else {
            const float* x_ptr = x + (o * dim + d) * inner_dim;
            context->Copy<float, CPUContext, CPUContext>(inner_dim, y_ptr, x_ptr);
        }
    }
}

template <> void ConstPad1DGrad<float, CPUContext>(const int count,
                                                   const int dim,
                                                   const int ex_dim,
                                                   const int inner_dim,
                                                   const int pad_l,
                                                   const float* dy,
                                                   float* dx,
                                                   CPUContext* context) {
    const int count_v2 = count / inner_dim;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count_v2))
#endif
    for (int idx = 0; idx < count_v2; ++idx) {
        const int d = idx % dim;
        const int o = idx / dim;
        const int ex_d = d + pad_l;
        const float* dy_ptr = dy + (o * ex_dim + ex_d) * inner_dim;
        float* dx_ptr = dx + (o * dim + d) * inner_dim;
        context->Copy<float, CPUContext, CPUContext>(inner_dim, dx_ptr, dy_ptr);
    }
}

template <> void ReflectPad1DGrad<float, CPUContext>(const int count,
                                                     const int dim,
                                                     const int ex_dim,
                                                     const int inner_dim,
                                                     const int pad_l,
                                                     const float* dy,
                                                     float* dx) {
    for (int idx = 0; idx < count; ++idx) {
        const int i = idx % inner_dim;
        const int ex_d = (idx / inner_dim) % ex_dim;
        const int o = idx / inner_dim / ex_dim;
        int d = ex_d - pad_l;
        d = std::max(d, -d);
        d = std::min(d, 2 * dim - d - 2);
        dx[(o * dim + d) * inner_dim + i] += dy[idx];
    }
}

template <> void EdgePad1DGrad<float, CPUContext>(const int count,
                                                  const int dim,
                                                  const int ex_dim,
                                                  const int inner_dim,
                                                  const int pad_l,
                                                  const float* dy,
                                                  float* dx,
                                                  CPUContext* context) {
    const int count_v2 = count / inner_dim;
    for (int idx = 0; idx < count_v2; ++idx) {
        const int ex_d = idx % ex_dim;
        const int o = idx / ex_dim;
        const int d = std::min(dim - 1, std::max(ex_d - pad_l, 0));
        const float* dy_ptr = dy + (o * ex_dim + ex_d) * inner_dim;
        if (d == 0 || d == dim - 1) {
            for (int i = 0; i < inner_dim; ++i)
                dx[(o * dim + d) * inner_dim + i] += dy_ptr[i];
        } else {
            float* dx_ptr = dx + (o * dim + d) * inner_dim;
            context->Copy<float, CPUContext, CPUContext>(inner_dim, dx_ptr, dy_ptr);
        }
    }
}

/******************** ndarray.one_hot ********************/

template <> void OneHot<float, CPUContext>(const int count,
                                           const int depth,
                                           const int on_value,
                                           const float* x,
                                           float* y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const int val = x[i];
        y[i * depth + val] = on_value;
    }
}

/******************** ndarray.reduce ********************/

template<> void Sum<float, CPUContext>(const int count, 
                                       const int axis_dim,
                                       const int inner_dim, 
                                       const float* x, 
                                       float* y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        float sum_val = 0.0;
        for (int j = 0; j < axis_dim; ++j)
            sum_val += x[(i / inner_dim * axis_dim + j) * inner_dim + i % inner_dim];
        y[i] = sum_val;
    }
}

template<> void SumGrad<float, CPUContext>(const int count, 
                                           const int axis_dim, 
                                           const int inner_dim,
                                           const float coeff, 
                                           const float* dy, 
                                           float* dx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < axis_dim; ++j)
            dx[(i / inner_dim * axis_dim + j) * inner_dim + i % inner_dim] = dy[i] * coeff;
    }
}

/******************** ndarray.repeat ********************/

template <> void Repeat<float, CPUContext>(const int count, 
                                           const int outer_dim,
                                           const int dim,
                                           const int inner_dim,
                                           const int repeats,
                                           const float* x,
                                           float* y,
                                           CPUContext* context) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < outer_dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            for (int k = 0; k < repeats; ++k) {
                context->Copy<float, CPUContext, CPUContext>(inner_dim, y, x);
                y += inner_dim;
            }
            x += inner_dim;
        }
    }
}

template <> void RepeatGrad<float, CPUContext>(const int count,
                                               const int outer_dim,
                                               const int dim,
                                               const int inner_dim,
                                               const int repeats,
                                               const float* dy,
                                               float* dx,
                                               CPUContext* context) {
    for (int i = 0; i < outer_dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            context->Copy<float, CPUContext, CPUContext>(inner_dim, dx, dy);
            dy += inner_dim;
            for (int k = 1; k < repeats; ++k) {
                math::Axpy<float, CPUContext>(inner_dim, 1.0, dy, dx);
                dy += inner_dim;
            }
            dx += inner_dim;
        }
    }
} 

/******************** ndarray.slice ********************/

template <> void Slice<float, CPUContext>(const int count, 
                                          const int outer_dim, 
                                          const int inner_dim,
                                          const int x_slice_dim, 
                                          const int y_slice_dim, 
                                          const int slice_offset,
                                          const float* x, 
                                          float* y, 
                                          CPUContext* context) {
    TIndex x_offset, y_offset;
    for (int n = 0; n < outer_dim; ++n) {
        x_offset = (n * x_slice_dim + slice_offset) * inner_dim;
        y_offset = n * y_slice_dim * inner_dim;
        context->Copy<float, CPUContext, CPUContext>(y_slice_dim * inner_dim, 
                                                     y + y_offset, 
                                                     x + x_offset);
    }
}

template <> void SliceGrad<float, CPUContext>(const int count, 
                                              const int outer_dim, 
                                              const int inner_dim,
                                              const int x_slice_dim, 
                                              const int y_slice_dim, 
                                              const int slice_offset,
                                              const float* dy, 
                                              float* dx, 
                                              CPUContext* context) {
    TIndex x_offset, y_offset;
    for (int n = 0; n < outer_dim; ++n) {
        x_offset = (n * x_slice_dim + slice_offset) * inner_dim;
        y_offset = n * y_slice_dim * inner_dim;
        context->Copy<float, CPUContext, CPUContext>(y_slice_dim * inner_dim, 
                                                     dx + x_offset, 
                                                     dy + y_offset);
    }
}

/******************** ndarray.tile ********************/

template <> void Tile<float, CPUContext>(const int count,
                                         const int outer_dim,
                                         const int ex_inner_dim,
                                         const int multiple,
                                         const float* x,
                                         float* y,
                                         CPUContext* context) {
    for (int i = 0; i < outer_dim; ++i) {
        for (int t = 0; t < multiple; ++t) {
            context->Copy<float, CPUContext, CPUContext>(ex_inner_dim, y, x);
            y += ex_inner_dim;
        }
        x += ex_inner_dim;
    }
}

template <> void TileGrad<float, CPUContext>(const int count, 
                                             const int outer_dim, 
                                             const int ex_inner_dim, 
                                             const int multiple, 
                                             const float* dy, 
                                             float* dx, 
                                             CPUContext* context) {
    for (int i = 0; i < outer_dim; ++i) {
        context->Copy<float, CPUContext, CPUContext>(ex_inner_dim, dx, dy);
        dy += ex_inner_dim;
        for (int t = 1; t < multiple; ++t) {
            math::Axpy<float, CPUContext>(ex_inner_dim, 1.0, dy, dx);
            dy += ex_inner_dim;
        }
        dx += ex_inner_dim;
    }
}

/******************** ndarray.transpose ********************/

template <> void Transpose<float, CPUContext>(const int count, 
                                              const int ndim, 
                                              const int* order, 
                                              const int* old_steps,
                                              const int* new_steps, 
                                              const float* x, 
                                              float* y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
       int x_idx = 0, y_idx = i;
       for (int j = 0; j < ndim; ++j) {
           int k = order[j];
           x_idx += (y_idx / new_steps[j]) * old_steps[k];
           y_idx %= new_steps[j];
       }
       y[i] = x[x_idx];
    }
}

template <> void Transpose<float16, CPUContext>(const int count, 
                                                const int ndim, 
                                                const int* order, 
                                                const int* old_steps,
                                                const int* new_steps, 
                                                const float16* x, 
                                                float16* y) {
    LOG(FATAL) << "float16 is unsupported for CPUContext.";
}

template <> void TransposeGrad<float, CPUContext>(const int count, 
                                                  const int ndim,
                                                  const int* order, 
                                                  const int* old_steps,
                                                  const int* new_steps, 
                                                  const float* dy, 
                                                  float* dx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        int x_idx = 0, y_idx = i;
        for (int j = 0; j < ndim; ++j) {
            int k = order[j];
            x_idx += (y_idx / new_steps[j]) * old_steps[k];
            y_idx %= new_steps[j];
        }
        dx[x_idx] = dy[i];
    }
}

template <> void TransposeGrad<float16, CPUContext>(const int count, 
                                                    const int ndim,
                                                    const int* order, 
                                                    const int* old_steps,
                                                    const int* new_steps, 
                                                    const float16* dy, 
                                                    float16* dx) {
    LOG(FATAL) << "float16 is unsupported for CPUContext.";
}

/******************** recurrent.lstm_uint ********************/

template <> void LSTMUnit<float, CPUContext>(const int count, 
                                             const int num, 
                                             const int channels,
                                             const float* c_1, 
                                             const float* x, 
                                             const float* cont, 
                                             float* x_act, 
                                             float* c, 
                                             float* h) {
    float i, f, o, g;
    int f_offset = channels, o_offset = 2 * channels, 
        g_offset = 3 * channels, x_offset = 4 * channels;
    for (int n = 0; n < num; ++n) {
        for (int ch = 0; ch < channels; ++ch) {
            i = _sigmoid<float>(x[ch]);
            if (cont != nullptr) {
                f = (cont[n] == 0) ? 
                     0 : (cont[n] * _sigmoid<float>(x[f_offset + ch]));
            } else { 
                f = _sigmoid<float>(x[f_offset + ch]); 
            }
            o = _sigmoid<float>(x[o_offset + ch]);
            g = tanh(x[g_offset + ch]);
            c[ch] = f * c_1[ch] + i * g;
            h[ch] = o * tanh(c[ch]);
            x_act[ch] = i;
            x_act[f_offset + ch] = f;
            x_act[o_offset + ch] = o;
            x_act[g_offset + ch] = g;
        }
        c_1 += channels;
        c += channels;
        h += channels;
        x += x_offset;
    }
}

template <> void LSTMUnitGrad<float, CPUContext>(const int count, 
                                                 const int num, 
                                                 const int channels,
                                                 const float* c_1, 
                                                 const float* x_act,
                                                 const float* c, 
                                                 const float* dc, 
                                                 const float* dh, 
                                                 float* dc_1, 
                                                 float* dx) {
    float i, f, o, g, tanh_c_t, dc_1_sum_term;
    float* p_di, *p_df, *p_do, *p_dg;
    int f_offset = channels, o_offset = 2 * channels,
        g_offset = 3 * channels, x_offset = 4 * channels;
    for (int n = 0; n < num; ++n) {
        for (int ch = 0; ch < channels; ++ch) {
            i = x_act[ch];
            f = x_act[f_offset + ch];
            o = x_act[o_offset + ch];
            g = x_act[g_offset + ch];
            p_di = dx + ch;
            p_df = dx + f_offset + ch;
            p_do = dx + o_offset + ch;
            p_dg = dx + g_offset + ch;
            //  BPTT compute the dc_{t-1} at the time of t
            //  dc_{t-1} =  dl / d(h_{t}) * d(h_{t}) / d(c_{t}) * d(c_{t}) / d(c_{t-1})
            //                  + d(c_{t+1}) / d(c_{t}) * d(c_{t}) / d(c_{t-1})
            //           =  (dl / d(h_{t}) * d(h_{t}) / d(c_{t}) + d(c_{t+1}) / d(c_{t}))
            //                  * d(c_{t}) / d(c_{t-1})
            tanh_c_t = tanh(c[ch]);
            dc_1_sum_term = dh[ch] * o * (1 - tanh_c_t * tanh_c_t) + dc[ch];
            dc_1[ch] = dc_1_sum_term * f;
            *p_di = dc_1_sum_term * g * i * (1 - i);
            *p_df = dc_1_sum_term * c_1[ch] * f * (1 - f);
            *p_do = dh[ch] * tanh_c_t * o * (1 - o);
            *p_dg = dc_1_sum_term * i * (1 - g * g);
        }
        c_1 += channels;
        c += channels;
        x_act += x_offset;
        dx += x_offset;
        dc += channels;
        dc_1 += channels;
        dh += channels;
    }
}

/******************** update.adam_update ********************/

template <> void AdamUpdate<float, CPUContext>(Tensor* x, 
                                               Tensor* m, 
                                               Tensor* v, 
                                               Tensor* t,
                                               const float beta1, 
                                               const float beta2, 
                                               const float eps, 
                                               const float lr) {
    TIndex count = x->count();
    t->Reshape(vector<TIndex>(1, count));
    auto* Xdata = x->mutable_data<float, CPUContext>();
    auto* Mdata = m->mutable_data<float, CPUContext>();
    auto* Vdata = v->mutable_data<float, CPUContext>();
    auto* Tdata = t->mutable_data<float, CPUContext>();
    math::Axpby<float, CPUContext>(count, 1.0 - beta1, Xdata, beta1, Mdata);
    math::Mul<float, CPUContext>(count, Xdata, Xdata, Tdata);
    math::Axpby<float, CPUContext>(count, 1.0 - beta2, Tdata, beta2, Vdata);
    math::Sqrt<float, CPUContext>(count, Vdata, Tdata);
    math::AddScalar<float, CPUContext>(count, eps, Tdata);
    math::Div<float, CPUContext>(count, Mdata, Tdata, Tdata);
    math::Scale<float, CPUContext>(count, lr, Tdata, Xdata);
}

/******************** update.nesterov_update ********************/

template <> void NesterovUpdate<float,  CPUContext>(const int count,
                                                    float* x,
                                                    float* h,
                                                    Tensor* t,
                                                    const float momentum,
                                                    const float lr,
                                                    CPUContext* ctx) {
    t->Reshape(vector<TIndex>(1, count));
    float* Tdata = t->mutable_data<float, CPUContext>();
    ctx->Copy<float, CPUContext, CPUContext>(count, Tdata, h);
    math::Axpby<float, CPUContext>(count, lr, x, momentum, h);
    math::Axpby<float, CPUContext>(count, 1.0 + momentum, h, -momentum, Tdata);
    ctx->Copy<float, CPUContext, CPUContext>(count, x, Tdata);
}

/******************** update.rmsprop_update ********************/

template <> void RMSPropUpdate<float, CPUContext>(const int count, 
                                                  float* x, 
                                                  float* h,
                                                  Tensor* t,
                                                  const float decay, 
                                                  const float eps, 
                                                  const float lr) {
    t->Reshape(vector<TIndex>(1, count));
    float* Tdata = t->mutable_data<float, CPUContext>();
    math::Square<float, CPUContext>(count, x, Tdata);
    math::Axpby<float, CPUContext>(count, 1.0 - decay, Tdata, decay, h);
    math::Sqrt<float, CPUContext>(count, h, Tdata);
    math::AddScalar<float, CPUContext>(count, eps, Tdata);
    math::Div<float, CPUContext>(count, x, Tdata, Tdata);
    math::Axpby<float, CPUContext>(count, lr, Tdata, 0.0, x);
}

/******************** vision.bilinear_resize ********************/

template <typename T>
void _BilinearResize_NCHW(const int N, const int C,
                          const int H, const int W,
                          const int out_h, const int out_w,
                          const float scale_h, const float scale_w,
                          const T* x,
                          T* y) {
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const int NC = n * C + c;
            for (int h = 0; h < out_h; ++h) {
                const float h_in = h * scale_h;
                const int top_y_idx = floorf(h_in);
                const int bottom_y_idx = (h_in < H - 1) ? ceilf(h_in) : H - 1;
                const int NCHT = NC * H + top_y_idx;
                const int NCHB = NC * H + bottom_y_idx;
                const float y_lerp = h_in - top_y_idx;
                for (int w = 0; w < out_w; ++w) {
                    const float w_in = w * scale_w;
                    const int left_x_idx = floorf(w_in);
                    const int right_x_idx = (w_in < W - 1) ? ceilf(w_in) : W - 1;
                    const float x_lerp = w_in - left_x_idx;

                    const float top_left(x[NCHT * W + left_x_idx]);
                    const float top_right(x[NCHT * W + right_x_idx]);
                    const float bottom_left(x[NCHB * W + left_x_idx]);
                    const float bottom_right(x[NCHB * W + right_x_idx]);

                    const float top = top_left + (top_right - top_left) * x_lerp;
                    const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
                    *(y++) = top + (bottom - top) * y_lerp;
                }
            }
        }
    }
}

template <typename T>
void _BilinearResize_NHWC(const int N, const int C,
                          const int H, const int W,
                          const int out_h, const int out_w,
                          const float scale_h, const float scale_w,
                          const T* x,
                          T* y) {
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < out_h; ++h) {
            const float h_in = h * scale_h;
            const int top_y_idx = floorf(h_in);
            const int bottom_y_idx = (h_in < H - 1) ? ceilf(h_in) : H - 1;
            const int NHT = n * H + top_y_idx;
            const int NHB = n * H + bottom_y_idx;
            const float y_lerp = h_in - top_y_idx;
            for (int w = 0; w < out_w; ++w) {
                const float w_in = w * scale_w;
                const int left_x_idx = floorf(w_in);
                const int right_x_idx = (w_in < W - 1) ? ceilf(w_in) : W - 1;
                const float x_lerp = w_in - left_x_idx;
                for (int c = 0; c < C; ++c) {
                    const float top_left(x[(NHT * W + left_x_idx) * C + c]);
                    const float top_right(x[(NHT * W + right_x_idx) * C + c]);
                    const float bottom_left(x[(NHB * W + left_x_idx) * C + c]);
                    const float bottom_right(x[(NHB * W + right_x_idx) * C + c]);
                    const float top = top_left + (top_right - top_left) * x_lerp;
                    const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
                    *(y++) = top + (bottom - top) * y_lerp;
                }
            }
        }
    }
}

template <> void BilinearResize<float, CPUContext>(const int count,
                                                   const int N, const int C,
                                                   const int H, const int W,
                                                   const int out_h, const int out_w,
                                                   const string& data_format,
                                                   const float* x, 
                                                   float* y) {
    const float scale_h = (float)H / out_h;
    const float scale_w = (float)W / out_w;
    if (data_format == "NCHW") {
        _BilinearResize_NCHW<float>(N, C, H, W,
                                  out_h, out_w,
                              scale_h, scale_w,
                                             x,
                                             y);
    } else if (data_format == "NHWC"){
        _BilinearResize_NHWC<float>(N, C, H, W,
                                  out_h, out_w,
                              scale_h, scale_w,
                                             x,
                                             y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template <typename T>
void _BilinearResizeGrad_NCHW(const int N, const int C,
                              const int H, const int W,
                              const int out_h, const int out_w,
                              const float scale_h, const float scale_w,
                              const T* dy,
                              T* dx) {
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const int NC = n * C + c;
            for (int h = 0; h < out_h; ++h) {
                const float h_in = h * scale_h;
                const int top_y_idx = floorf(h_in);
                const int bottom_y_idx = (h_in < H - 1) ? ceilf(h_in) : H - 1;
                const int NCHT = NC * H + top_y_idx;
                const int NCHB = NC * H + bottom_y_idx;
                const float y_lerp = h_in - top_y_idx;
                for (int w = 0; w < out_w; ++w) {
                    const float w_in = w * scale_w;
                    const int left_x_idx = floorf(w_in);
                    const int right_x_idx = (w_in < W - 1) ? ceilf(w_in) : W - 1;
                    const float x_lerp = w_in - left_x_idx;
                    const float dtop = (1 - y_lerp) * (*(dy));
                    const float dbottom = y_lerp * (*(dy++));
                    dx[NCHT * W + left_x_idx] += static_cast<T>((1 - x_lerp) * dtop);
                    dx[NCHT * W + right_x_idx] += static_cast<T>(x_lerp * dtop);
                    dx[NCHB * W + left_x_idx] += static_cast<T>((1 - x_lerp) * dbottom);
                    dx[NCHB * W + right_x_idx] += static_cast<T>(x_lerp * dbottom);
                }
            }
        }
    }
}

template <typename T>
void _BilinearResizeGrad_NHWC(const int N, const int C,
                              const int H, const int W,
                              const int out_h, const int out_w,
                              const float scale_h, const float scale_w,
                              const T* dy,
                              T* dx) {
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < out_h; ++h) {
            const float h_in = h * scale_h;
            const int top_y_idx = floorf(h_in);
            const int bottom_y_idx = (h_in < H - 1) ? ceilf(h_in) : H - 1;
            const int NHT = n * H + top_y_idx;
            const int NHB = n * H + bottom_y_idx;
            const float y_lerp = h_in - top_y_idx;
            for (int w = 0; w < out_w; ++w) {
                const float w_in = w * scale_w;
                const int left_x_idx = floorf(w_in);
                const int right_x_idx = (w_in < W - 1) ? ceilf(w_in) : W - 1;
                const float x_lerp = w_in - left_x_idx;
                const float dtop = (1 - y_lerp) * (*(dy));
                const float dbottom = y_lerp * (*(dy++));
                for (int c = 0; c < C; ++c) {
                    dx[(NHT * W + left_x_idx) * C + c] += static_cast<T>((1 - x_lerp) * dtop);
                    dx[(NHT * W + right_x_idx) * C + c] += static_cast<T>(x_lerp * dtop);
                    dx[(NHB * W + left_x_idx) * C + c] += static_cast<T>((1 - x_lerp) * dbottom);
                    dx[(NHB * W + right_x_idx) * C + c] += static_cast<T>(x_lerp * dbottom);
                }
            }
        }
    }
}

template <> void BilinearResizeGrad<float, CPUContext>(const int count,
                                                       const int N, const int C,
                                                       const int H, const int W,
                                                       const int out_h, const int out_w,
                                                       const string& data_format,
                                                       const float* dy,
                                                       float* dx) {
    const float scale_h = (float)H / out_h;
    const float scale_w = (float)W / out_w;
    math::Set<float, CPUContext>(N * C * H * W, 0, dx);
    if (data_format == "NCHW") {
        _BilinearResizeGrad_NCHW<float>(N, C, H, W,
                                      out_h, out_w,
                                  scale_h, scale_w,
                                                dy,
                                                dx);
    } else if (data_format == "NHWC"){
        _BilinearResizeGrad_NHWC<float>(N, C, H, W,
                                      out_h, out_w,
                                  scale_h, scale_w,
                                                dy,
                                                dx);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/******************** vision.conv ********************/

template<typename T>
void _Im2Col2d_NCHW(const int C, const int H, const int W,
                    const int col_h, const int col_w,
                    const int kernel_h, const int kernel_w,
                    const int stride_h, const int stride_w,
                    const int pad_h, const int pad_w,
                    const int dilation_h, const int dilation_w,
                    const T* im,
                    T* col) {
    const int im_offset = H * W;
    for (int c = 0; c < C; ++c, im += im_offset) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h = -pad_h + kh * dilation_h;
                for (int output_h = 0; output_h < col_h; ++output_h) {
                    if (!judge(h, H)) {
                        for (int output_w = 0; output_w < col_w; ++output_w) *(col++) = 0;
                    } else {
                        int w = -pad_w + kw * dilation_w;
                        for (int output_w = 0; output_w < col_w; ++output_w) {
                            if (!judge(w, W)) *(col++) = 0;
                            else *(col++) = im[h * W + w];
                            w += stride_w;
                        }
                    }
                    h += stride_h;
                }
            }
        }
    }
}

template<typename T>
void _Im2Col2d_NHWC(const int C, const int H, const int W,
                    const int col_h, const int col_w,
                    const int kernel_h, const int kernel_w,
                    const int stride_h, const int stride_w,
                    const int pad_h, const int pad_w,
                    const int dilation_h, const int dilation_w,
                    const T* im,
                    T* col) {
    for (int output_h = 0; output_h < col_h; ++output_h) {
        const int base_h = -pad_h + stride_h * output_h;
        for (int output_w = 0; output_w < col_w; ++output_w) {
            const int base_w = -pad_w + stride_w * output_w;
            for (int kh = 0; kh < kernel_h; ++kh) {
                int h = base_h + kh * dilation_h;
                if (!judge(h, H)) {
                    for (int kw = 0; kw < kernel_w; ++kw)
                        for (int c = 0; c < C; ++c) *(col++) = 0;
                } else {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int w = base_w + kw * dilation_w;
                        for (int c = 0; c < C; ++c) {
                            if (!judge(w, W)) *(col++) = 0;
                            else *(col++) = im[(h * W + w) * C + c];
                        }
                    }
                }
            }
        }
    }
}

template <> void Im2Col2d<float, CPUContext>(const int C, const int H, const int W,
                                             const int col_h, const int col_w,
                                             const int kernel_h, const int kernel_w, 
                                             const int stride_h, const int stride_w, 
                                             const int pad_h, const int pad_w,
                                             const int dilation_h, const int dilation_w,
                                             const string& data_format,
                                             const float* im,
                                             float* col) {
    if (data_format == "NCHW") {
        const int count = (C * col_h * col_w);
        _Im2Col2d_NCHW<float>(C, H, W, col_h, col_w,
                                 kernel_h, kernel_w,
                                 stride_h, stride_w,
                                       pad_h, pad_w,
                             dilation_h, dilation_w,
                                                 im,
                                                col);
    } else if (data_format == "NHWC") {
        const int count = (col_h * col_w * C);
        _Im2Col2d_NHWC<float>(C, H, W, col_h, col_w,
                                 kernel_h, kernel_w,
                                 stride_h, stride_w,
                                       pad_h, pad_w,
                             dilation_h, dilation_w,
                                                 im,
                                                col);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template<typename T>
void _Col2Im2d_NCHW(const int C, const int H, const int W,
                    const int col_h, const int col_w,
                    const int kernel_h, const int kernel_w,
                    const int stride_h, const int stride_w,
                    const int pad_h, const int pad_w,
                    const int dilation_h, const int dilation_w,
                    const T* col,
                    T* im) {
    math::Set<float, CPUContext>(C * H * W, 0, im);
    const int im_offset = H * W;
    for (int c = 0; c < C; ++c, im += im_offset) {
        for (int kh = 0; kh < kernel_h; ++kh) {
            for (int kw = 0; kw < kernel_w; ++kw) {
                int h = -pad_h + kh * dilation_h;
                for (int output_h = 0; output_h < col_h; ++output_h) {
                    if (!judge(h, H)) {
                        col += col_w;
                    } else {
                        int w = -pad_w + kw * dilation_w;
                        for (int output_w = 0; output_w < col_w; ++output_w) {
                            if (judge(w, W)) im[h * W + w] += *col;
                            ++col;
                            w += stride_w;
                        }
                    }
                    h += stride_h;
                }
            }
        } 
    }
}

template<typename T>
void _Col2Im2d_NHWC(const int C, const int H, const int W,
                    const int col_h, const int col_w,
                    const int kernel_h, const int kernel_w,
                    const int stride_h, const int stride_w,
                    const int pad_h, const int pad_w,
                    const int dilation_h, const int dilation_w,
                    const T* col,
                    T* im) {
    math::Set<float, CPUContext>(C * H * W, 0, im);
    for (int output_h = 0; output_h < col_h; ++output_h) {
        const int base_h = -pad_h + stride_h * output_h;
        for (int output_w = 0; output_w < col_w; ++output_w) {
            const int base_w = -pad_w + stride_w * output_w;
            for (int kh = 0; kh < kernel_h; ++kh) {
                int h = base_h + kh * dilation_h;
                if (!judge(h, H)) {
                    col += (kernel_w * C);
                } else {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        int w = base_w + kw * dilation_w;
                        for (int c = 0; c < C; ++c) {
                            if (judge(w, W)) im[(h * W + w) * C + c] += *(col);
                            ++col;
                        }
                    }
                }
            }
        }
    }
}

template<> void Col2Im2d<float, CPUContext>(const int C, const int H, const int W,
                                            const int col_h, const int col_w,
                                            const int kernel_h, const int kernel_w,
                                            const int stride_h, const int stride_w,
                                            const int pad_h, const int pad_w,
                                            const int dilation_h, const int dilation_w,
                                            const string& data_format,
                                            const float* col,
                                            float* im) {
    if (data_format == "NCHW") {
        const int count = (C * H * W);
        _Col2Im2d_NCHW<float>(C, H, W, col_h, col_w,
                                 kernel_h, kernel_w,
                                 stride_h, stride_w,
                                       pad_h, pad_w,
                             dilation_h, dilation_w,
                                                col,
                                                 im);
    } else if (data_format == "NHWC") {
        const int count = (H * W * C);
        _Col2Im2d_NHWC<float>(C, H, W, col_h, col_w,
                                 kernel_h, kernel_w,
                                 stride_h, stride_w,
                                       pad_h, pad_w,
                             dilation_h, dilation_w,
                                                col,
                                                 im);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/******************** vision.nn_resize ********************/

template <typename T>
void _NNResize_NCHW(const int N, const int C,
                    const int H, const int W,
                    const int out_h, const int out_w,
                    const float scale_h, const float scale_w,
                    const T* x,
                    T* y) {
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const int NC = n * C + c;
            for (int h = 0; h < out_h; ++h) {
                const int h_in = std::min(int(floorf(h * scale_h)), H - 1);
                const int NCH = NC * H + h_in;
                for (int w = 0; w < out_w; ++w) {
                    const int w_in = std::min(int(floorf(w * scale_w)), W - 1);
                    *(y++) = x[NCH * W + w_in];
                }
            }
        }
    }
}

template <typename T>
void _NNResize_NHWC(const int N, const int C,
                    const int H, const int W,
                    const int out_h, const int out_w,
                    const float scale_h, const float scale_w,
                    const T* x,
                    T* y) {
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < out_h; ++h) {
            const int h_in = std::min(int(floorf(h * scale_h)), H - 1);
            const int NH = n * H + h_in;
            for (int w = 0; w < out_w; ++w) {
                const int w_in = std::min(int(floorf(w * scale_w)), W - 1);
                const int NHW = NH * W + w_in;
                for (int c = 0; c < C; ++c) *(y++) = x[NHW * C + c];
            }
        }
    }
}

template <> void NNResize<float, CPUContext>(const int count, 
                                             const int N, const int C,
                                             const int H, const int W, 
                                             const int out_h, const int out_w,
                                             const string& data_format,
                                             const float* x,
                                             float* y) {
    const float scale_h = (float)H / out_h;
    const float scale_w = (float)W / out_w;
    if (data_format == "NCHW") {
        _NNResize_NCHW<float>(N, C, H, W, out_h, out_w,
                                      scale_h, scale_w,
                                                     x,
                                                     y);
    } else if (data_format == "NHWC"){
        _NNResize_NHWC<float>(N, C, H, W, out_h, out_w,
                                      scale_h, scale_w,
                                                     x,
                                                     y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template <typename T>
void _NNResizeGrad_NCHW(const int N, const int C,
                        const int H, const int W,
                        const int out_h, const int out_w,
                        const float scale_h, const float scale_w,
                        const T* dy,
                        T* dx) {
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            const int NC = n * C + c;
            for (int h = 0; h < out_h; ++h) {
                const int h_in = std::min(int(floorf(h * scale_h)), H - 1);
                const int NCH = NC * H + h_in;
                for (int w = 0; w < out_w; ++w) {
                    const int w_in = std::min(int(floorf(w * scale_w)), W - 1);
                    dx[NCH * W + w_in] += *(dy++);
                }
            }
        }
    }
}

template <typename T>
void _NNResizeGrad_NHWC(const int N, const int C,
                    const int H, const int W,
                    const int out_h, const int out_w,
                    const float scale_h, const float scale_w,
                    const T* dy,
                    T* dx) {
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < out_h; ++h) {
            const int h_in = std::min(int(floorf(h * scale_h)), H - 1);
            const int NH = n * H + h_in;
            for (int w = 0; w < out_w; ++w) {
                const int w_in = std::min(int(floorf(w * scale_w)), W - 1);
                const int NHW = NH * W + w_in;
                for (int c = 0; c < C; ++c) dx[NHW * C + c] += *(dy++);
            }
        }
    }
}

template <> void NNResizeGrad<float, CPUContext>(const int count,
                                                 const int N, const int C,
                                                 const int H, const int W, 
                                                 const int out_h, const int out_w,
                                                 const string& data_format,
                                                 const float* dy,
                                                 float* dx) {
    const float scale_h = (float)H / out_h;
    const float scale_w = (float)W / out_w;
    math::Set<float, CPUContext>(N * C * H * W, 0, dx);
    if (data_format == "NCHW") {
        _NNResizeGrad_NCHW<float>(N, C, H, W, out_h, out_w,
                                          scale_h, scale_w,
                                                        dy,
                                                        dx);
    } else if (data_format == "NHWC"){
        _NNResizeGrad_NHWC<float>(N, C, H, W, out_h, out_w,
                                          scale_h, scale_w,
                                                        dy,
                                                        dx);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/******************** vision.pooling ********************/

template <typename T>
void _MAXPooling2d_NCHW(const int N, const int C,
                        const int H, const int W,
                        const int pool_h, const int pool_w,
                        const int kernel_h, const int kernel_w, 
                        const int stride_h, const int stride_w,
                        const int pad_h, const int pad_w,
                        const float* x,
                        int* mask,
                        float* y) {
    int x_offset = H * W;
    int y_offset = pool_h * pool_w;
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int ph = 0; ph < pool_h; ++ph) {
                for (int pw = 0; pw < pool_w; ++pw) {
                    int start_h = ph * stride_h - pad_h;
                    int start_w = pw * stride_w - pad_w;
                    int end_h = std::min(start_h + kernel_h, H);
                    int end_w = std::min(start_w + kernel_w, W);
                    start_h = std::max(start_h, 0);
                    start_w = std::max(start_w, 0);
                    const int pool_idx = ph * pool_w + pw;
                    float max_val = -FLT_MAX;
                    int max_idx = -1;
                    for (int h = start_h; h < end_h; ++h) {
                        for (int w = start_w; w < end_w; ++w) {
                            const int idx = h * W + w;
                            if (x[idx] > max_val) {
                                max_val = x[idx];
                                max_idx = idx;
                            }
                        }
                    }
                    y[pool_idx] = max_val;
                    mask[pool_idx] = max_idx;
                }
            } 
            x += x_offset;
            y += y_offset;
            mask += y_offset;
        }
    }
}

template <typename T>
void _MAXPooling2d_NHWC(const int N, const int C,
                        const int H, const int W,
                        const int pool_h, const int pool_w,
                        const int kernel_h, const int kernel_w,
                        const int stride_h, const int stride_w,
                        const int pad_h, const int pad_w,
                        const float* x,
                        int* mask,
                        float* y) {
    int x_offset = H * W * C;
    int y_offset = pool_h * pool_w * C;
    for (int n = 0; n < N; ++n) {
        for (int ph = 0; ph < pool_h; ph++) {
            for (int pw = 0; pw < pool_w; ++pw) {
                int start_h = ph * stride_h - pad_h;
                int start_w = pw * stride_w - pad_w;
                int end_h = std::min(start_h + kernel_h, H);
                int end_w = std::min(start_w + kernel_w, W);
                start_h = std::max(start_h, 0);
                start_w = std::max(start_w, 0);
                const int base_pool_idx = ph * pool_w + pw;
                for (int c = 0; c < C; ++c) {
                    const int pool_idx = base_pool_idx * C + c;
                    float max_val = -FLT_MAX;
                    int max_idx = -1;
                    for (int h = start_h; h < end_h; ++h) {
                        for (int w = start_w; w < end_w; ++w) {
                            const int idx = (h * W + w) * C + c;
                            if (x[idx] > max_val) {
                                max_val = x[idx];
                                max_idx = idx;
                            }
                        }
                    }
                    y[pool_idx] = max_val;
                    mask[pool_idx] = max_idx;
                }
            }
        }
        x += x_offset;
        y += y_offset;
        mask += y_offset;
    }
}

template<> void MAXPooling2d<float, CPUContext>(const int count, 
                                                const int N, const int C,
                                                const int H, const int W,
                                                const int pool_h, const int pool_w,
                                                const int kernel_h, const int kernel_w,
                                                const int stride_h, const int stride_w,
                                                const int pad_h, const int pad_w,
                                                const string& data_format,
                                                const float* x, 
                                                int* mask, 
                                                float* y) {
    if (data_format == "NCHW") {
        _MAXPooling2d_NCHW<float>(N, C, H, W, pool_h, pool_w,
                                          kernel_h, kernel_w,
                                          stride_h, stride_w,
                                                pad_h, pad_w,
                                                           x,
                                                        mask,
                                                          y);

    } else if (data_format == "NHWC") {
        _MAXPooling2d_NHWC<float>(N, C, H, W, pool_h, pool_w,
                                          kernel_h, kernel_w,
                                          stride_h, stride_w,
                                                pad_h, pad_w,
                                                           x,
                                                        mask,
                                                          y);

    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template<typename T>
void _AVGPooling2d_NCHW(const int N, const int C,
                        const int H, const int W,
                        const int pool_h, const int pool_w,
                        const int kernel_h, const int kernel_w,
                        const int stride_h, const int stride_w,
                        const int pad_h, const int pad_w,
                        const float* x,
                        float* y) {
    int x_offset = H * W;
    int y_offset = pool_h * pool_w;
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int ph = 0; ph < pool_h; ++ph) {
                for (int pw = 0; pw < pool_w; ++pw) {
                    int start_h = ph * stride_h - pad_h;
                    int start_w = pw * stride_w - pad_w;
                    int end_h = std::min(start_h + kernel_h, H + pad_h);
                    int end_w = std::min(start_w + kernel_w, W + pad_w);
                    int pool_area = (end_h - start_h) * (end_w - start_w);
                    end_h = std::min(end_h, H);
                    end_w = std::min(end_w, W);
                    start_h = std::max(start_h, 0);
                    start_w = std::max(start_w, 0);
                    const int pool_idx = ph * pool_w + pw;
                    T sum_val = 0;
                    for (int h = start_h; h < end_h; ++h)
                        for (int w = start_w; w < end_w; ++w)
                            sum_val += x[h * W + w];
                    y[pool_idx] = sum_val / pool_area;
                } 
            }
            x += x_offset;
            y += y_offset;
        }
    }
}

template<typename T>
void _AVGPooling2d_NHWC(const int N, const int C,
                        const int H, const int W,
                        const int pool_h, const int pool_w,
                        const int kernel_h, const int kernel_w,
                        const int stride_h, const int stride_w,
                        const int pad_h, const int pad_w,
                        const float* x,
                        float* y) {
    int x_offset = H * W * C;
    int y_offset = pool_h * pool_w * C;
    for (int n = 0; n < N; ++n) {
        for (int ph = 0; ph < pool_h; ph++) {
            for (int pw = 0; pw < pool_w; ++pw) {
                int start_h = ph * stride_h - pad_h;
                int start_w = pw * stride_w - pad_w;
                int end_h = std::min(start_h + kernel_h, H + pad_h);
                int end_w = std::min(start_w + kernel_w, W + pad_w);
                int pool_area = (end_h - start_h) * (end_w - start_w);
                end_h = std::min(end_h, H);
                end_w = std::min(end_w, W);
                start_h = std::max(start_h, 0);
                start_w = std::max(start_w, 0);
                const int base_pool_idx = ph * pool_w + pw;
                for (int c = 0; c < C; ++c) {
                    const int pool_idx = base_pool_idx * C + c;
                    T sum_val = 0;
                    for (int h = start_h; h < end_h; ++h)
                        for (int w = start_w; w < end_w; ++w)
                            sum_val += x[(h * W + w) * C + c];
                    y[pool_idx] = sum_val / pool_area;
                }
            }
        }
        x += x_offset;
        y += y_offset;
    }
}

template<> void AVGPooling2d<float, CPUContext>(const int count,
                                                const int N, const int C,
                                                const int H, const int W,
                                                const int pool_h, const int pool_w,
                                                const int kernel_h, const int kernel_w,
                                                const int stride_h, const int stride_w,
                                                const int pad_h, const int pad_w,
                                                const string& data_format,
                                                const float* x,
                                                float* y) {
    if (data_format == "NCHW") {
        _AVGPooling2d_NCHW<float>(N, C, H, W, pool_h, pool_w,
                                          kernel_h, kernel_w,
                                          stride_h, stride_w,
                                                pad_h, pad_w,
                                                           x,
                                                          y);

    } else if (data_format == "NHWC") {
        _AVGPooling2d_NHWC<float>(N, C, H, W, pool_h, pool_w,
                                          kernel_h, kernel_w,
                                          stride_h, stride_w,
                                                pad_h, pad_w,
                                                           x,
                                                          y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template <typename T>
void _MAXPooling2dGrad_NCHW(const int N, const int C,
                            const int H, const int W,
                            const int pool_h, const int pool_w,
                            const int kernel_h, const int kernel_w,
                            const int stride_h, const int stride_w,
                            const int pad_h, const int pad_w,
                            const float* dy,
                            const int* mask,
                            float* dx) {
    int x_offset = H * W;
    int y_offset = pool_h * pool_w;
    math::Set<float, CPUContext>(N * C * H * W, 0, dx);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int ph = 0; ph < pool_h; ++ph) {
                for (int pw = 0; pw < pool_w; ++pw) {
                    const int pool_idx = ph * pool_w + pw;
                    const int idx = mask[pool_idx];
                    dx[idx] += dy[pool_idx];
                }
            }
            dx += x_offset;
            dy += y_offset;
            mask += y_offset;
        }
    }
}

template <typename T>
void _MAXPooling2dGrad_NHWC(const int N, const int C,
                            const int H, const int W,
                            const int pool_h, const int pool_w,
                            const int kernel_h, const int kernel_w,
                            const int stride_h, const int stride_w,
                            const int pad_h, const int pad_w,
                            const float* dy,
                            const int* mask,
                            float* dx) {
    int x_offset = H * W * C;
    int y_offset = pool_h * pool_w * C;
    math::Set<float, CPUContext>(N * H * W * C, 0, dx);
    for (int n = 0; n < N; ++n) {
        for (int ph = 0; ph < pool_h; ph++) {
            for (int pw = 0; pw < pool_w; ++pw) {
                const int base_pool_idx = ph * pool_w + pw;
                for (int c = 0; c < C; ++c) {
                    const int pool_idx = base_pool_idx * C + c;
                    const int idx = mask[pool_idx];
                    dx[idx] += dy[pool_idx];
                }
            }
        }
        dx += x_offset;
        dy += y_offset;
    }
}

template<> void MAXPooling2dGrad<float, CPUContext>(const int count,
                                                    const int N, const int C,
                                                    const int H, const int W,
                                                    const int pool_h, const int pool_w,
                                                    const int kernel_h, const int kernel_w,
                                                    const int stride_h, const int stride_w,
                                                    const int pad_h, const int pad_w,
                                                    const string& data_format,
                                                    const float* dy,
                                                    const int* mask,
                                                    float* dx) {


   if (data_format == "NCHW") {
        _MAXPooling2dGrad_NCHW<float>(N, C, H, W, pool_h, pool_w,
                                              kernel_h, kernel_w,
                                              stride_h, stride_w,
                                                    pad_h, pad_w,
                                                              dy,
                                                            mask,
                                                             dx);

    } else if (data_format == "NHWC") {
        _MAXPooling2dGrad_NHWC<float>(N, C, H, W, pool_h, pool_w,
                                              kernel_h, kernel_w,
                                              stride_h, stride_w,
                                                    pad_h, pad_w,
                                                              dy,
                                                            mask,
                                                             dx);

    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template <typename T>
void _AVGPooling2dGrad_NCHW(const int N, const int C,
                            const int H, const int W,
                            const int pool_h, const int pool_w,
                            const int kernel_h, const int kernel_w,
                            const int stride_h, const int stride_w,
                            const int pad_h, const int pad_w,
                            const float* dy,
                            float* dx) {
    int x_offset = H * W;
    int y_offset = pool_h * pool_w;
    math::Set<float, CPUContext>(N * C * H * W, 0, dx);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int ph = 0; ph < pool_h; ++ph) {
                for (int pw = 0; pw < pool_w; ++pw) {
                    int start_h = ph * stride_h - pad_h;
                    int start_w = pw * stride_w - pad_w;
                    int end_h = std::min(start_h + kernel_h, H + pad_h);
                    int end_w = std::min(start_w + kernel_w, W + pad_w);
                    int pool_area = (end_h - start_h) * (end_w - start_w);
                    end_h = std::min(end_h, H);
                    end_w = std::min(end_w, W);
                    start_h = std::max(start_h, 0);
                    start_w = std::max(start_w, 0);
                    const int pool_idx = ph * pool_w + pw;
                    for (int h = start_h; h < end_h; ++h) {
                        for (int w = start_w; w < end_w; ++w) {
                            const int idx = h * W + w;
                            dx[idx] += (dy[pool_idx] / pool_area);
                        }
                    }
                }
            } 
            dx += x_offset;
            dy += y_offset;
        }
    }
}

template <typename T>
void _AVGPooling2dGrad_NHWC(const int N, const int C,
                            const int H, const int W,
                            const int pool_h, const int pool_w,
                            const int kernel_h, const int kernel_w,
                            const int stride_h, const int stride_w,
                            const int pad_h, const int pad_w,
                            const float* dy,
                            float* dx) {
    int x_offset = H * W * C;
    int y_offset = pool_h * pool_w * C;
    math::Set<float, CPUContext>(N * H * W * C, 0, dx);
    for (int n = 0; n < N; ++n) {
        for (int ph = 0; ph < pool_h; ph++) {
            for (int pw = 0; pw < pool_w; ++pw) {
                int start_h = ph * stride_h - pad_h;
                int start_w = pw * stride_w - pad_w;
                int end_h = std::min(start_h + kernel_h, H + pad_h);
                int end_w = std::min(start_w + kernel_w, W + pad_w);
                int pool_area = (end_h - start_h) * (end_w - start_w);
                end_h = std::min(end_h, H);
                end_w = std::min(end_w, W);
                start_h = std::max(start_h, 0);
                start_w = std::max(start_w, 0);
                const int base_pool_idx = ph * pool_w + pw;
                for (int c = 0; c < C; ++c) {
                    const int pool_idx = base_pool_idx * C + c;
                    for (int h = start_h; h < end_h; ++h)
                        for (int w = start_w; w < end_w; ++w)
                            dx[(h * W + w) * C + c] += (dy[pool_idx] / pool_area);
                }
            }
        }
        dx += x_offset;
        dy += y_offset;
    }
}

template<> void AVGPooling2dGrad<float, CPUContext>(const int count, 
                                                    const int N, const int C,
                                                    const int H, const int W,
                                                    const int pool_h, const int pool_w,
                                                    const int kernel_h, const int kernel_w,
                                                    const int stride_h, const int stride_w,
                                                    const int pad_h, const int pad_w,
                                                    const string& data_format,
                                                    const float* dy, 
                                                    float* dx) {
    if (data_format == "NCHW") {
        _AVGPooling2dGrad_NCHW<float>(N, C, H, W, pool_h, pool_w,
                                              kernel_h, kernel_w,
                                              stride_h, stride_w,
                                                    pad_h, pad_w,
                                                              dy,
                                                             dx);

    } else if (data_format == "NHWC") {
        _AVGPooling2dGrad_NHWC<float>(N, C, H, W, pool_h, pool_w,
                                              kernel_h, kernel_w,
                                              stride_h, stride_w,
                                                    pad_h, pad_w,
                                                              dy,
                                                             dx);

    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/******************** vision.roi_pooling ********************/

template<> void ROIPooling<float, CPUContext>(const float spatial_scale, 
                                              const int pool_h, const int pool_w,
                                              Tensor* x,
                                              Tensor* roi,
                                              Tensor* mask,
                                              Tensor* y) {
    auto* Xdata = x->data<float, CPUContext>();
    auto* Rdata = roi->data<float, CPUContext>();
    auto* Ydata = y->mutable_data<float, CPUContext>();
    auto* Mdata = mask->mutable_data<int, CPUContext>();
    int num_rois = roi->dim(0), batch_size = x->dim(0);
    int channels = x->dim(1), count = y->count();
    int height = x->dim(2), width = x->dim(3);
    math::Set<float, CPUContext>(count, -FLT_MAX, Ydata);
    math::Set<int, CPUContext>(count, -1, Mdata);
    for (int n = 0; n < num_rois; ++n) {
        int im_idx = Rdata[0];
        int x1 = round(Rdata[1] * spatial_scale);
        int y1 = round(Rdata[2] * spatial_scale);
        int x2 = round(Rdata[3] * spatial_scale);
        int y2 = round(Rdata[4] * spatial_scale);
        int roi_height = std::max(y2 - y1 + 1, 1);
        int roi_width = std::max(x2 - x1 + 1, 1);
        const float unit_h = (float)roi_height / (float)pool_h;
        const float unit_w = (float)roi_width / (float)pool_w;
        const float* Idata = Xdata + x->offset(im_idx);
        for (int c = 0; c < channels; ++c) {
            for (int ph = 0; ph < pool_h; ++ph) {
                for (int pw = 0; pw < pool_w; ++pw) {
                    int start_h = floor(unit_h * ph);
                    int start_w = floor(unit_w * pw);
                    int end_h = ceil(unit_h*(ph + 1));
                    int end_w = ceil(unit_w*(pw + 1));
                    start_h = std::max(start_h + y1, 0);
                    start_w = std::max(start_w + x1, 0);
                    end_h = std::max(end_h + y1, 0);
                    end_w = std::max(end_w + x1, 0);
                    start_h = std::min(start_h, height);
                    start_w = std::min(start_w, width);
                    end_h = std::min(end_h, height);
                    end_w = std::min(end_w, width);
                    bool is_empty = (end_h == start_h) || (end_w == start_w);
                    const int pool_idx = ph * pool_w + pw;
                    if (is_empty) {
                        Ydata[pool_idx] = 0;
                        Mdata[pool_idx] = -1;
                    }
                    for (int h = start_h; h < end_h; ++h) {
                        for (int w = start_w; w < end_w; ++w) {
                            const int idx = h * width + w;
                            if (Idata[idx] > Ydata[pool_idx]) {
                                    Ydata[pool_idx] = Idata[idx];
                                    Mdata[pool_idx] = idx;
                            }
                        }    //  end w
                    }    //  end h
                }    //  end pw
            }    //  end ph
            //  offset image channels
            Idata += x->offset(0, 1);
            Ydata += y->offset(0, 1);
            Mdata += mask->offset(0, 1);
        }    //  end c
        //  offset roi region
        Rdata += 5;
    }    //  end n
}

template<> void ROIPoolingGrad<float, CPUContext>(const float spatial_scale, 
                                                  const int pool_h, const int pool_w,
                                                  Tensor* dy,
                                                  Tensor* roi,
                                                  Tensor* mask,
                                                  Tensor* dx) {
    NOT_IMPLEMENTED;
}

/******************** vision.roi_align ********************/

template<> void ROIAlign<float, CPUContext>(const float spatial_scale, 
                                            const int pool_h, const int pool_w,
                                            Tensor* x,
                                            Tensor* roi,
                                            Tensor* mask_h,
                                            Tensor* mask_w,
                                            Tensor* y) {
    NOT_IMPLEMENTED;
}

template<> void ROIAlignGrad<float, CPUContext>(const float spatial_scale, 
                                                const int pool_h, const int pool_w,
                                                Tensor* dy,
                                                Tensor* roi,
                                                Tensor* mask_h,
                                                Tensor* mask_w,
                                                Tensor* dx) {
    NOT_IMPLEMENTED;
}

}    // namespace kernel

}    // namespace dragon