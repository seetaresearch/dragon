#include <algorithm>
#include <functional>

#include "core/tensor.h"
#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"
#include "utils/sse_alternative.h"
#include "utils/math_functions.h"
#include "utils/cast.h"

bool judge(int a, int b)  { return unsigned(a) < unsigned(b); }

namespace dragon {

namespace kernel {

/******************** activation.dropout ********************/

template<> void Dropout<float, CPUContext>(
    const int               count,
    float                   prob,
    float                   scale,
    const float*            x,
    uint32_t*               mask,
    float*                  y,
    CPUContext*             ctx) {
    uint32_t thresh = static_cast<uint32_t>(UINT_MAX * prob);
    math::RandomBernoulli<float, CPUContext>(count, 1 - prob, mask, ctx);
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) y[i] = x[i] * mask[i] * scale;
}

template<> void DropoutGrad<float, CPUContext>(
    const int               count,
    float                   prob,
    float                   scale,
    const float*            dy,
    const uint32_t*         mask,
    float*                  dx,
    CPUContext*             ctx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i)
        dx[i] = dy[i] * mask[i] * scale;
}

/******************** activation.elu ********************/

template<> void Elu<float, CPUContext>(
    const int               count,
    const float             alpha,
    const float*            x,
    float*                  y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = std::max(x[i], 0.f) + alpha *
            (std::exp(std::min(x[i], 0.f)) - 1.f);
    }
}

template<> void EluGrad<float, CPUContext>(
    const int               count,
    const float             alpha,
    const float*            dy,
    const float*            y,
    float*                  dx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        dx[i] = dy[i] * (
            (y[i] > 0) + (alpha + y[i]) * (y[i] <= 0)
        );
    }
}

/******************** activation.prelu ********************/

template<> void PRelu<float, CPUContext>(
    const int               count,
    const int               channels,
    const int               dim,
    const bool              channel_shared,
    const string&           data_format,
    const float*            x,
    const float*            w,
    float*                  y) {
    if (channel_shared) {
#ifdef WITH_OMP
        #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
        for (int i = 0; i < count; ++i) {
            y[i] = std::max(x[i], 0.f) +
                w[0] * std::min(x[i], 0.f);
        }
    } else {
        if (data_format == "NCHW") {
#ifdef WITH_OMP
            #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
            for (int i = 0; i < count; ++i) {
                int c = (i / dim) % channels;
                y[i] = std::max(x[i], 0.f) +
                    w[c] * std::min(x[i], 0.f);
            }
        } else if (data_format == "NHWC") {
#ifdef WITH_OMP
            #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
            for (int i = 0; i < count; ++i) {
                int c = i % channels;
                y[i] = std::max(x[i], 0.f) +
                    w[c] * std::min(x[i], 0.f);
            }
        } else LOG(FATAL) << "Unknown data format: " << data_format;
    }
}

template<> void PReluGrad<float, CPUContext>(
    const int               count,
    const int               channels,
    const int               dim,
    const bool              channel_shared,
    const string&           data_format,
    const float*            dy,
    const float*            x,
    const float*            w,
    float*                  dx) {
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

template<> void PReluWGrad<float, CPUContext>(
    const int               rows,
    const int               row_offset,
    const int               channels,
    const int               dim,
    const bool              channel_shared,
    const string&           data_format,
    const float*            dy,
    const float*            x,
    const float*            multiplier,
    float*                  bcast_dw,
    float*                  dw,
    CPUContext*             ctx) {
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
        float w_sum = math::Dot<float, CPUContext>(
            channels * dim, bcast_dw, multiplier, ctx);
        math::AddScalar<float, CPUContext>(1, w_sum, dw);
    } else {
        if (data_format == "NCHW") {
            math::Gemv<float, CPUContext>(
                CblasNoTrans, channels, dim,
                    1.0, bcast_dw, multiplier,
                        1.0, dw, ctx);
        } else if (data_format == "NHWC") {
            math::Gemv<float, CPUContext>(
                CblasTrans, dim, channels,
                    1.0, bcast_dw, multiplier,
                        1.0, dw, ctx);
        } else LOG(FATAL) << "Unknown data format: " << data_format;
    }
}

/******************** activation.relu ********************/

template<> void Relu<float, CPUContext>(
    const int               count,
    const float             slope,
    const float*            x,
    float*                  y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = std::max(x[i], 0.f) + slope * std::min(x[i], 0.f);
    }
}

template<> void Relu<float16, CPUContext>(
    const int               count,
    const float             slope,
    const float16*          x,
    float16*                y) {
    CPU_FP16_NOT_SUPPORTED;
}

template<> void ReluGrad<float, CPUContext>(
    const int               count,
    const float             slope,
    const float*            dy,
    const float*            y,
    float*                  dx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        dx[i] = dy[i] * ((y[i] > 0) + slope * (y[i] <= 0));
    }
}

/******************** activation.selu ********************/

template<> void SElu<float, CPUContext>(
    const int               count,
    const float*            x,
    float*                  y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = 1.0507 * std::max(x[i], 0.f)
             + 1.7581 * (std::exp(std::min(x[i], 0.f)) - 1.f);
    }
}

template<> void SEluGrad<float, CPUContext>(
    const int               count,
    const float*            dy,
    const float*            y,
    float*                  dx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        dx[i] = y[i] > 0 ? 1.0507 * dy[i] :
            (1.7581 + y[i]) * dy[i];
    }
}

/******************** activation.sigmoid ********************/

template <typename T>
T _sigmoid(T x) { return T(1) / (T(1) + exp(-x)); }

template<> void Sigmoid<float, CPUContext>(
    const int               count,
    const float*            x,
    float*                  y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i)  y[i] = _sigmoid<float>(x[i]);
}

template<> void SigmoidGrad<float, CPUContext>(
    const int               count,
    const float*            dy,
    const float*            y,
    float*                  dx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        dx[i] = dy[i] * y[i] * (1 - y[i]);
    }
}

/******************** activation.softmax ********************/

template<> void Softmax<float, CPUContext>(
    const int               count,
    const int               classes,
    const int               outer_dim,
    const int               inner_dim,
    const float*            sum_multiplier,
    const float*            x,
    float*                  scale,
    float*                  y,
    CPUContext*             ctx) {
    const int dim = count / outer_dim;
    for (int i = 0; i < outer_dim; ++i) {
       CPUContext::Copy<float, CPUContext, CPUContext>(
            inner_dim, scale, x + i*dim);
        for (int j = 0; j < classes; ++j) {
            for (int k = 0; k < inner_dim; k++)
                scale[k] = std::max(
                    scale[k], x[i * dim + j * inner_dim + k]
                );
        }
        math::Gemm<float, CPUContext>(
            CblasNoTrans, CblasNoTrans, 
                classes, inner_dim, 1,
                    -1.0, sum_multiplier, scale, 1.0, y, ctx);
        math::Exp<float, CPUContext>(dim, y, y);
        math::Gemv<float, CPUContext>(
            CblasTrans, classes, inner_dim,
                1.0, y, sum_multiplier,
                    0.0, scale, ctx);
        for (int j = 0; j < classes; ++j) {
            math::Div<float, CPUContext>(inner_dim, y, scale, y);
            y += inner_dim;
        }
    }
}

template<> void SoftmaxGrad<float, CPUContext>(
    const int               count,
    const int               classes,
    const int               outer_dim,
    const int               inner_dim,
    const float*            sum_multiplier,
    const float*            dy,
    const float*            y,
    float*                  scale,
    float*                  dx,
    CPUContext*             ctx) {
    const int dim = count / outer_dim;
    for (int i = 0; i < outer_dim; ++i) {
        for (int k = 0; k < inner_dim; ++k)
            scale[k] = math::StridedDot<float, CPUContext>(
                classes,
                    dx + i * dim + k, inner_dim,
                        y + i*dim + k, inner_dim, ctx);
         math::Gemm<float, CPUContext>(
             CblasNoTrans, CblasNoTrans,
                classes, inner_dim, 1,
                    -1.0, sum_multiplier, scale,
                        1.0, dx + i * dim, ctx);
    }
    math::Mul<float, CPUContext>(count, dx, y, dx);
}

/******************** activation.tanh ********************/

template<> void Tanh<float, CPUContext>(
    const int               count,
    const float*            x,
    float*                  y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        y[i] = std::tanh(x[i]);
    }
}

template<> void TanhGrad<float, CPUContext>(
    const int               count,
    const float*            dy,
    const float*            y,
    float*                  dx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        dx[i] = dy[i] * (1 - y[i] * y[i]);
    }
}

/******************** arithmetic.affine ********************/

template<> void Affine<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               scale_dim,
    const int               inner_dim,
    const float*            x,
    const float*            alpha,
    const float*            beta,
    const float*            beta_multiplier,
    float*                  y,
    CPUContext*             ctx) {
    //  Ax
    auto* Xdata = x; auto* Ydata = y;
    for (int n = 0; n < outer_dim; ++n) {
        for (int d = 0; d < scale_dim; ++d) {
            math::Scale<float, CPUContext>(
                inner_dim, alpha[d], Xdata, Ydata, ctx);
            Xdata += inner_dim; 
            Ydata += inner_dim;
        }
    }
    //  Pb
    if (beta != nullptr && beta_multiplier != nullptr) {
        int dim = scale_dim * inner_dim;
        Ydata = y;
        for (int n = 0; n < outer_dim; ++n) {
            math::Gemm<float, CPUContext>(
                CblasNoTrans, CblasNoTrans,
                    scale_dim, inner_dim, 1,
                        1.0, beta, beta_multiplier,
                            1.0, Ydata, ctx);
             Ydata += dim;
        }
    }
}

template<> void Affine<float16, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               scale_dim,
    const int               inner_dim,
    const float16*          x,
    const float16*          alpha,
    const float16*          beta,
    const float16*          beta_multiplier,
    float16*                y,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void AffineGrad<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               scale_dim,
    const int               inner_dim,
    const float*            dy,
    const float*            alpha,
    float*                  dx,
    CPUContext*             ctx) {
    auto* dYdata = dy; auto* dXdata = dx;
    for (int n = 0; n < outer_dim; ++n) {
        for (int d = 0; d < scale_dim; ++d) {
            math::Scale<float, CPUContext>(
                inner_dim, alpha[d], dYdata, dXdata, ctx);
            dYdata += inner_dim; dXdata += inner_dim;
        }
    }
}

/******************** arithmetic.clip ********************/

template <> void Clip<float, CPUContext>(
    const int               count,
    const float             low,
    const float             high,
    const float*            x,
    float*                  mask,
    float*                  y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        mask[i] = 1.0;
        if (x[i] < low || x[i] > high) mask[i] = 0.0;
        y[i] = std::max(low, std::min(x[i], high));
    }
}

/******************** control_flow.compare ********************/

template <> void Equal<float, CPUContext>(
    const int               count,
    const float*            a,
    const float*            b,
    float*                  y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i)
        y[i] = fabs(a[i] - b[i]) < FLT_EPSILON ? 1.0 : 0.0;
}

/******************** loss.l1_loss ********************/

template<> void AbsGrad<float, CPUContext>(
    const int               count,
    const float*            dy,
    float*                  dx) {
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

template <> void SigmoidCrossEntropy<float, CPUContext>(
    const int               count,
    const float*            x,
    const float*            target,
    float*                  loss,
    float*                  valid) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        if (target[i] < 0) {
            loss[i] = valid[i] = 0.;
        } else {
            loss[i] = std::log(
                1 + std::exp(x[i] - 2 * x[i] * (x[i] >= 0))
            ) + x[i] * ((x[i] >= 0) - target[i]);
            valid[i] = 1.;
        }
    }
}

template <> void SigmoidCrossEntropyGrad<float, CPUContext>(
    const int               count,
    const float*            x,
    const float*            target,
    float*                  dx,
    float*                  valid) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        if (target[i] < 0) {
            dx[i] = valid[i] = 0.;
        } else {
            dx[i] = 1. / (1. + expf(-x[i])) - target[i];
            valid[i] = 1.;
        }
    }
}

/******************** loss.smooth_l1_loss ********************/

template<> void SmoothL1<float, CPUContext>(
    const int               count,
    const float             beta,
    const float*            x,
    float*                  y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const float val = x[i];
        const float abs_val = abs(val);
        if (abs_val < beta) y[i] = 0.5 * val * val / beta;
        else y[i] = abs_val - 0.5 * beta;
    }
}

template<> void SmoothL1Grad<float, CPUContext>(
    const int               count,
    const float             beta,
    const float*            dy,
    float*                  dx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const float val = dy[i];
        const float abs_val = abs(val);
        if (abs_val < beta) dx[i] = val / beta;
        //  val > 0: 1 | val == 0: 0 | val < 0: -1
        else dx[i] = (val > float(0)) - (val < float(0));
    }
}

/******************** loss.softmax_cross_entropy ********************/

template <> void SoftmaxCrossEntropy<float, CPUContext>(
    const int               count,
    const float*            prob,
    const float*            target,
    float*                  loss) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        loss[i] = - target[i] * std::log(std::max(prob[i], FLT_MIN));
    }
}

/******************** loss.sparse_softmax_cross_entropy ********************/

template <typename Tx, typename Ty>
void _SparseSoftmaxCrossEntropy(
    const int               count,
    const int               classes,
    const int               outer_dim,
    const int               inner_dim,
    const Tx*               prob,
    const Ty*               labels,
    Tx*                     loss,
    Tx*                     valid,
    Tensor*                 ignore) {
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
                Tx labeled_prob = prob[i * dim + label * inner_dim + j];
                loss[idx] = -std::log(std::max(labeled_prob, FLT_MIN));
                valid[idx] = 1;
            }
        }
    }
}

template <> void SparseSoftmaxCrossEntropy<float, float, CPUContext>(
    const int               count,
    const int               classes,
    const int               outer_dim,
    const int               inner_dim,
    const float*            prob,
    const float*            labels,
    float*                  loss,
    float*                  valid,
    Tensor*                 ignore,
    CPUContext*             ctx) {
    _SparseSoftmaxCrossEntropy<float, float>(
        count, classes, outer_dim, inner_dim,
            prob, labels, loss, valid, ignore);
}

template <> void SparseSoftmaxCrossEntropy<float, int64_t, CPUContext>(
    const int               count,
    const int               classes,
    const int               outer_dim,
    const int               inner_dim,
    const float*            prob,
    const int64_t*          labels,
    float*                  loss,
    float*                  valid,
    Tensor*                 ignore,
    CPUContext*             ctx) {
    _SparseSoftmaxCrossEntropy<float, int64_t>(
        count, classes, outer_dim, inner_dim,
            prob, labels, loss, valid, ignore);
}

template <typename Tx, typename Ty>
void _SparseSoftmaxCrossEntropyGrad(
    const int               count,
    const int               classes,
    const int               outer_dim,
    const int               inner_dim,
    const Tx*               prob,
    const Ty*               labels,
    Tx*                     valid,
    Tensor*                 ignore,
    Tx*                     dx) {
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

template<> void SparseSoftmaxCrossEntropyGrad<float, float, CPUContext>(
    const int               count,
    const int               classes,
    const int               outer_dim,
    const int               inner_dim,
    const float*            prob,
    const float*            labels,
    float*                  valid,
    Tensor*                 ignore,
    float*                  dx,
    CPUContext*             ctx) {
    _SparseSoftmaxCrossEntropyGrad<float, float>(
        count, classes, outer_dim, inner_dim,
            prob, labels, valid, ignore, dx);
}

template<> void SparseSoftmaxCrossEntropyGrad<float, int64_t, CPUContext>(
    const int               count,
    const int               classes,
    const int               outer_dim,
    const int               inner_dim,
    const float*            prob,
    const int64_t*          labels,
    float*                  valid,
    Tensor*                 ignore,
    float*                  dx,
    CPUContext*             ctx) {
    _SparseSoftmaxCrossEntropyGrad<float, int64_t>(
        count, classes, outer_dim, inner_dim,
            prob, labels, valid, ignore, dx);
}

/******************** loss.sparse_softmax_focal_loss ********************/

template <> void SparseSoftmaxFocalLoss<float, CPUContext>(
    const int               count,
    const int               classes,
    const int               outer_dim,
    const int               inner_dim,
    const float             pos_alpha,
    const float             neg_alpha,
    const float             gamma,
    const int               neg_id,
    const float*            prob,
    const float*            labels,
    float*                  scale,
    float*                  loss,
    float*                  valid,
    Tensor*                 ignore) {
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
                scale[t_] = label > neg_id ?
                    pos_alpha * scale[t_] :  neg_alpha * scale[t_];
                loss[idx] = -scale[t_] * std::log(labeled_prob);
                valid[idx] = label > neg_id ? 1 : 0;
            }
        }
    }
}

template<> void SparseSoftmaxFocalLossGrad<float, CPUContext>(
    const int               count,
    const int               classes,
    const int               outer_dim,
    const int               inner_dim,
    const float             gamma,
    const int               neg_id,
    const float             eps,
    const float*            scale,
    const float*            prob,
    const float*            labels,
    float*                  valid,
    Tensor*                 ignore,
    float*                  dx) {
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
                float grad = -gamma
                    * (scale[t_] / std::max((1.0f - prob[t_]), eps))
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

/******************** misc.astype ********************/

template <typename Ta, typename Tb>
void _TypeA2B(const int count, const Ta* a, Tb* b) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) b[i] = a[i];
}

template <typename Ta, typename Tb>
void _TypeA2B_v2(const int count, const Ta* a, Tb* b) {
#ifdef WITH_OMP
#pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) b[i] = dragon_cast<Tb, Ta>(a[i]);
}

#define DEFINE_TYPE_A2B(type_a, type_b) \
    template <> void TypeA2B<type_a, type_b, CPUContext>( \
        const int           count, \
        const type_a*       a, \
        type_b*             b) { \
        _TypeA2B<type_a, type_b>(count, a, b); \
    }

#define DEFINE_TYPE_A2B_V2(type_a, type_b) \
    template <> void TypeA2B<type_a, type_b, CPUContext>( \
        const int           count, \
        const type_a*       a, \
        type_b*             b) { \
        _TypeA2B_v2<type_a, type_b>(count, a, b); \
    }

#define DEFINE_TYPE_DISABLE_FP16(type) \
    template <> void TypeA2B<float16, type, CPUContext>( \
        const int           count, \
        const float16*      a, \
        type*               b) { \
        CPU_FP16_NOT_SUPPORTED; \
    } \
    template <> void TypeA2B<type, float16, CPUContext>( \
        const int           count, \
        const type*         a, \
        float16*            b) { \
        CPU_FP16_NOT_SUPPORTED; \
    }

#define DEFINE_TYPE_A2ALL(type_a) \
    DEFINE_TYPE_A2B(type_a, float); \
    DEFINE_TYPE_A2B(type_a, double); \
    DEFINE_TYPE_A2B(type_a, int); \
    DEFINE_TYPE_A2B(type_a, int64_t); \
    DEFINE_TYPE_A2B(type_a, uint8_t);

DEFINE_TYPE_A2B_V2(float16, float);
DEFINE_TYPE_A2B_V2(float, float16);
DEFINE_TYPE_A2B_V2(float16, float16);
DEFINE_TYPE_A2ALL(float);
DEFINE_TYPE_A2ALL(double); DEFINE_TYPE_DISABLE_FP16(double);
DEFINE_TYPE_A2ALL(int); DEFINE_TYPE_DISABLE_FP16(int);
DEFINE_TYPE_A2ALL(int64_t); DEFINE_TYPE_DISABLE_FP16(int64_t);
DEFINE_TYPE_A2ALL(uint8_t); DEFINE_TYPE_DISABLE_FP16(uint8_t);

/******************** misc.image_data ********************/

template <typename Tx, typename Ty>
void _ImageData_NCHW(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean_values,
    const float*            std_values,
    const Tx*               x,
    Ty*                     y) {
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int h = 0; h < H; ++h) {
                const int NH = n * H + h;
                for (int w = 0; w < W; ++w) {
                    Ty raw_value = x[(NH * W + w) * C + c];
                    if (mean_values) raw_value -= mean_values[c];
                    if (std_values) raw_value /= std_values[c];
                    *(y++) = raw_value;
                }
            }
        }
    }
}

template <typename Tx, typename Ty>
void _ImageData_NHWC(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean_values,
    const float*            std_values,
    const Tx*               x,
    Ty*                     y) {
    for (int n = 0; n < N; ++n) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                for (int c = 0; c < C; ++c) {
                    Ty raw_value = *(x++);
                    if (mean_values) raw_value -= mean_values[c];
                    if (std_values) raw_value /= std_values[c]; 
                    *(y++) = raw_value;
                }
            }
        }
    }
}

template <> void ImageData<float, float, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean_values,
    const float*            std_values,
    const string&           data_format,
    const float*            x,
    float*                  y) {
    if (data_format == "NCHW") {
        _ImageData_NCHW<float, float>(
            N, C, H, W, mean_values, std_values, x, y);
    } else if (data_format == "NHWC") {
        _ImageData_NHWC<float, float>(
            N, C, H, W, mean_values, std_values, x, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template <> void ImageData<uint8_t, float, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean_values,
    const float*            std_values,
    const string&           data_format,
    const uint8_t*          x,
    float*                  y) {
    if (data_format == "NCHW") {
        _ImageData_NCHW<uint8_t, float>(
            N, C, H, W, mean_values, std_values, x, y);
    } else if (data_format == "NHWC") {
        _ImageData_NHWC<uint8_t, float>(
            N, C, H, W, mean_values, std_values, x, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template <> void ImageData<float, float16, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean_values,
    const float*            std_values,
    const string&           data_format,
    const float*            x,
    float16*                y) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void ImageData<uint8_t, float16, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const float*            mean_values,
    const float*            std_values,
    const string&           data_format,
    const uint8_t*          x,
    float16*                y) {
    CPU_FP16_NOT_SUPPORTED;
}

/******************** ndarray.arange ********************/

template<> void Arange<float, CPUContext>(
    const int               count,
    const int               start,
    const int               step,
    float*                  y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) y[i] = start + i * step;
}

template<> void Arange<int, CPUContext>(
    const int               count,
    const int               start,
    const int               step,
    int*                    y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) y[i] = start + i * step;
}

/******************** ndarray.argreduce ********************/

template<> void Argmax<float, CPUContext>(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const int               top_k,
    const float*            x,
    int64_t*                indices,
    float*                  values) {
    vector<pair<float, int> > vec(axis_dim);
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < axis_dim; ++j) 
            vec[j] = std::make_pair(x[(i / inner_dim * axis_dim + j) *
                inner_dim + i % inner_dim], j);
        std::partial_sort(
            vec.begin(), vec.begin() + top_k, vec.end(),
                std::greater< pair<float, int> >());
        for (int j = 0; j < top_k; ++j) {
            TIndex y_idx = (i / inner_dim * top_k + j) *
                inner_dim + i % inner_dim;
            indices[y_idx] = vec[j].second;
            if (values) values[y_idx] = vec[j].first;
        }
    }
}

template<> void Argmin<float, CPUContext>(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const int               top_k,
    const float*            x,
    int64_t*                indices,
    float*                  values) {
    vector<pair<float, int> > vec(axis_dim);
#ifdef WITH_OMP
#pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < axis_dim; ++j) 
            vec[j] = std::make_pair(x[(i / inner_dim * axis_dim + j) *
                inner_dim + i % inner_dim], j);
        std::partial_sort(vec.begin(), vec.begin() + top_k, vec.end());
        for (int j = 0; j < top_k; ++j) {
            TIndex y_idx = (i / inner_dim * top_k + j) *
                inner_dim + i % inner_dim;
            indices[y_idx] = vec[j].second;
            if (values) values[y_idx] = vec[j].first;
        }
    }
}

/******************** ndarray.gather ********************/

template <> void CanonicalAxis<int, CPUContext>(
    const int               count,
    const int               dim,
    int*                    y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) if (y[i] < 0) y[i] += dim;
}

template <typename T>
void _Gather(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const T*                x,
    T*                      y) {
    TIndex x_offset, y_offset, x_idx_offset, y_idx_offset;
    for (int i = 0; i < y_slice_dim; ++i) {
        y_idx_offset = i;
        x_idx_offset = indices[y_idx_offset];
        for (int n = 0; n < outer_dim; ++n) {
            x_offset = (n * x_slice_dim + x_idx_offset) * inner_dim;
            y_offset = (n * y_slice_dim + y_idx_offset) * inner_dim;
            CPUContext::Copy<T, CPUContext, CPUContext>(
                inner_dim, y + y_offset, x + x_offset);
        }
    }
}

template <> void Gather<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const float*            x,
    float*                  y) {
    _Gather<float>(count, outer_dim, inner_dim,
        x_slice_dim, y_slice_dim, indices, x, y);
}

template <> void Gather<int, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const int*              x,
    int*                    y) {
    _Gather<int>(count, outer_dim, inner_dim,
        x_slice_dim, y_slice_dim, indices, x, y);
}

template <typename T>
void _GatherGrad(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const T*                dy,
    T*                      dx) {
    TIndex x_offset, y_offset, x_idx_offset, y_idx_offset;
    for (int i = 0; i < y_slice_dim; ++i) {
        y_idx_offset = i;
        x_idx_offset = indices[y_idx_offset];
        for (int n = 0; n < outer_dim; ++n) {
            x_offset = (n * x_slice_dim + x_idx_offset) * inner_dim;
            y_offset = (n * y_slice_dim + y_idx_offset) * inner_dim;
            math::Add<T, CPUContext>(inner_dim,
                dy + y_offset, dx + x_offset, dx + x_offset);
        }
    }
}

template <> void GatherGrad<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const float*            dy,
    float*                  dx) {
    _GatherGrad<float>(count, outer_dim, inner_dim,
        x_slice_dim, y_slice_dim, indices, dy, dx);
}

template <> void GatherGrad<int, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int*              indices,
    const int*              dy,
    int*                    dx) {
    _GatherGrad<int>(count, outer_dim, inner_dim,
        x_slice_dim, y_slice_dim, indices, dy, dx);
}

/******************** ndarray.concat ********************/

template <> void Concat<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const float*            x,
    float*                  y) {
    TIndex x_offset, y_offset;
    for (int n = 0; n < outer_dim; ++n) {
        x_offset = n * x_concat_dim * inner_dim;
        y_offset = (n * y_concat_dim + concat_offset) * inner_dim;
        CPUContext::Copy<float, CPUContext, CPUContext>(
            x_concat_dim * inner_dim, y + y_offset, x + x_offset);
    }
}

template <> void Concat<float16, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const float16*          x,
    float16*                y) {
    TIndex x_offset, y_offset;
    for (int n = 0; n < outer_dim; ++n) {
        x_offset = n * x_concat_dim * inner_dim;
        y_offset = (n * y_concat_dim + concat_offset) * inner_dim;
        CPUContext::Copy<float16, CPUContext, CPUContext>(
            x_concat_dim * inner_dim, y + y_offset, x + x_offset);
    }
}

template <> void ConcatGrad<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const float*            dy,
    float*                  dx) {
    TIndex x_offset, y_offset;
    for (int n = 0; n < outer_dim; ++n) {
        x_offset = n * x_concat_dim * inner_dim;
        y_offset = (n * y_concat_dim + concat_offset) * inner_dim;
        CPUContext::Copy<float, CPUContext, CPUContext>(
            x_concat_dim * inner_dim, dx + x_offset, dy + y_offset);
    }
}

template <> void ConcatGrad<float16, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_concat_dim,
    const int               y_concat_dim,
    const int               concat_offset,
    const float16*          dy,
    float16*                dx) {
    TIndex x_offset, y_offset;
    for (int n = 0; n < outer_dim; ++n) {
        x_offset = n * x_concat_dim * inner_dim;
        y_offset = (n * y_concat_dim + concat_offset) * inner_dim;
        CPUContext::Copy<float16, CPUContext, CPUContext>(
            x_concat_dim * inner_dim, dx + x_offset, dy + y_offset);
    }
}

/******************** ndarray.crop ********************/

template <typename T>
void _Crop1D(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const T*                x,
    T*                      y) {
    const int count_v2 = count / inner_dim;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count_v2))
#endif
    for (int idx = 0; idx < count_v2; ++idx) {
        const int ex_d = idx % ex_dim;
        const int o = idx / ex_dim;
        const T* x_ptr = x + (o * dim + ex_d + start) * inner_dim;
        T* y_ptr = y + (o * ex_dim + ex_d) * inner_dim;
        CPUContext::Copy<T, CPUContext, CPUContext>(
            inner_dim, y_ptr, x_ptr);
    }
}

template<> void Crop1D<int, CPUContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const int*              x,
    int*                    y) {
    _Crop1D<int>(count, dim, ex_dim, inner_dim, start, x, y);
}

template<> void Crop1D<float, CPUContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const float*            x,
    float*                  y) {
    _Crop1D<float>(count, dim, ex_dim, inner_dim, start, x, y);
}

template <typename T>
void _Crop1DGrad(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const int               end,
    const T*                dy,
    T*                      dx) {
    const int count_v2 = count / inner_dim;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count_v2))
#endif
    for (int idx = 0; idx < count_v2; ++idx) {
        const int d = idx % dim;
        const int o = idx / dim;
        T* dx_ptr = dx + (o * dim + d) * inner_dim;
        if (d < start || d >= end) {
            for (int i = 0; i < inner_dim; ++i) dx_ptr[i] = 0;
        } else {
            const T* dy_ptr = dy + (o * ex_dim + d - start) * inner_dim;
            CPUContext::Copy<T, CPUContext, CPUContext>(
                inner_dim, dx_ptr, dy_ptr);
        }
    }
}

template<> void Crop1DGrad<int, CPUContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const int               end,
    const int*              dy,
    int*                    dx) {
    _Crop1DGrad<int>(
        count, dim, ex_dim, inner_dim,
            start, end, dy, dx);
}

template<> void Crop1DGrad<float, CPUContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               start,
    const int               end,
    const float*            dy,
    float*                  dx) {
    _Crop1DGrad<float>(
        count, dim, ex_dim, inner_dim,
            start, end, dy, dx);
}

/******************** ndarray.pad ********************/

template <> void ConstPad1D<float, CPUContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float             value,
    const float*            x,
    float*                  y) {
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
            CPUContext::Copy<float, CPUContext, CPUContext>(
                inner_dim, y_ptr, x_ptr);
        }
    }
}

template <> void ReflectPad1D<float, CPUContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float*            x,
    float*                  y) {
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
            CPUContext::Copy<float, CPUContext, CPUContext>(
                inner_dim, y_ptr, x_ptr);
        }
    }
}

template <> void EdgePad1D<float, CPUContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float*            x,
    float*                  y) {
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
            CPUContext::Copy<float, CPUContext, CPUContext>(
                inner_dim, y_ptr, x_ptr);
        }
    }
}

template <> void ConstPad1DGrad<float, CPUContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float*            dy,
    float*                  dx) {
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
        CPUContext::Copy<float, CPUContext, CPUContext>(
            inner_dim, dx_ptr, dy_ptr);
    }
}

template <> void ReflectPad1DGrad<float, CPUContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float*            dy,
    float*                  dx) {
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

template <> void EdgePad1DGrad<float, CPUContext>(
    const int               count,
    const int               dim,
    const int               ex_dim,
    const int               inner_dim,
    const int               pad_l,
    const float*            dy,
    float*                  dx) {
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
            CPUContext::Copy<float, CPUContext, CPUContext>(
                inner_dim, dx_ptr, dy_ptr);
        }
    }
}

/******************** ndarray.one_hot ********************/

template <> void OneHot<float, CPUContext>(
    const int               count,
    const int               depth,
    const int               on_value,
    const float*            x,
    float*                  y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const int val = x[i];
        y[i * depth + val] = on_value;
    }
}

/******************** ndarray.reduce ********************/

template<> void Sum<float, CPUContext>(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const float*            x,
    float*                  y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        float sum_val = 0.0;
        for (int j = 0; j < axis_dim; ++j)
            sum_val += x[(i / inner_dim * axis_dim + j)
                          * inner_dim + i % inner_dim];
        y[i] = sum_val;
    }
}

template<> void SumGrad<float, CPUContext>(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const float             coeff,
    const float*            dy,
    float*                  dx) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < axis_dim; ++j)
            dx[(i / inner_dim * axis_dim + j)
                * inner_dim + i % inner_dim] = dy[i] * coeff;
    }
}

/******************** ndarray.repeat ********************/

template <> void Repeat<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               dim,
    const int               inner_dim,
    const int               repeats,
    const float*            x,
    float*                  y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < outer_dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            for (int k = 0; k < repeats; ++k) {
                CPUContext::Copy<float, CPUContext, CPUContext>(
                    inner_dim, y, x);
                y += inner_dim;
            }
            x += inner_dim;
        }
    }
}

template <> void RepeatGrad<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               dim,
    const int               inner_dim,
    const int               repeats,
    const float*            dy,
    float*                  dx,
    CPUContext*             ctx) {
    for (int i = 0; i < outer_dim; ++i) {
        for (int j = 0; j < dim; ++j) {
            CPUContext::Copy<float, CPUContext, CPUContext>(
                inner_dim, dx, dy);
            dy += inner_dim;
            for (int k = 1; k < repeats; ++k) {
                math::Axpy<float, CPUContext>(
                    inner_dim, 1.0, dy, dx, ctx);
                dy += inner_dim;
            }
            dx += inner_dim;
        }
    }
} 

/******************** ndarray.slice ********************/

template <> void Slice<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int               slice_offset,
    const float*            x,
    float*                  y) {
    TIndex x_offset, y_offset;
    for (int n = 0; n < outer_dim; ++n) {
        x_offset = (n * x_slice_dim + slice_offset) * inner_dim;
        y_offset = n * y_slice_dim * inner_dim;
        CPUContext::Copy<float, CPUContext, CPUContext>(
            y_slice_dim * inner_dim, y + y_offset, x + x_offset);
    }
}

template <> void SliceGrad<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               inner_dim,
    const int               x_slice_dim,
    const int               y_slice_dim,
    const int               slice_offset,
    const float*            dy,
    float*                  dx) {
    TIndex x_offset, y_offset;
    for (int n = 0; n < outer_dim; ++n) {
        x_offset = (n * x_slice_dim + slice_offset) * inner_dim;
        y_offset = n * y_slice_dim * inner_dim;
        CPUContext::Copy<float, CPUContext, CPUContext>(
            y_slice_dim * inner_dim, dx + x_offset, dy + y_offset);
    }
}

/******************** ndarray.tile ********************/

template <> void Tile<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               ex_inner_dim,
    const int               multiple,
    const float*            x,
    float*                  y) {
    for (int i = 0; i < outer_dim; ++i) {
        for (int t = 0; t < multiple; ++t) {
            CPUContext::Copy<float, CPUContext, CPUContext>(
                ex_inner_dim, y, x);
            y += ex_inner_dim;
        }
        x += ex_inner_dim;
    }
}

template <> void TileGrad<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               ex_inner_dim,
    const int               multiple,
    const float*            dy,
    float*                  dx,
    CPUContext*             ctx) {
    for (int i = 0; i < outer_dim; ++i) {
        CPUContext::Copy<float, CPUContext, CPUContext>(
            ex_inner_dim, dx, dy);
        dy += ex_inner_dim;
        for (int t = 1; t < multiple; ++t) {
            math::Axpy<float, CPUContext>(
                ex_inner_dim, 1.0, dy, dx, ctx);
            dy += ex_inner_dim;
        }
        dx += ex_inner_dim;
    }
}

/******************** ndarray.transpose ********************/

template <> void Transpose<float, CPUContext>(
    const int               count,
    const int               ndim,
    const int*              order,
    const int*              old_steps,
    const int*              new_steps,
    const float*            x,
    float*                  y) {
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

template <> void Transpose<float16, CPUContext>(
    const int               count,
    const int               ndim,
    const int*              order,
    const int*              old_steps,
    const int*              new_steps,
    const float16*          x,
    float16*                y) {
    CPU_FP16_NOT_SUPPORTED;
}

template <> void TransposeGrad<float, CPUContext>(
    const int               count,
    const int               ndim,
    const int*              order,
    const int*              old_steps,
    const int*              new_steps,
    const float*            dy,
    float*                  dx) {
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

template <> void TransposeGrad<float16, CPUContext>(
    const int               count,
    const int               ndim,
    const int*              order,
    const int*              old_steps,
    const int*              new_steps,
    const float16*          dy,
    float16*                dx) {
    CPU_FP16_NOT_SUPPORTED;
}

/******************** recurrent.lstm_cell ********************/

template <> void LSTMCell<float, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const float*            cx,
    float*                  xact,
    float*                  c,
    float*                  h) {
    float i, f, o, c_;
    int f_offset = C, o_offset = 2 * C, c_offset = 3 * C, x_offset = 4 * C;
    for (int n = 0; n < N; ++n) {
        for (int idx = 0; idx < C; ++idx) {
            xact[idx] = i = _sigmoid<float>(xact[idx]);
            xact[idx + f_offset] = f = _sigmoid<float>(xact[idx + f_offset]);
            xact[idx + o_offset] = o = _sigmoid<float>(xact[idx + o_offset]);
            xact[idx + c_offset] = c_ = tanh(xact[idx + c_offset]);
            c_ = c[idx] = f * cx[idx] + i * c_;
            h[idx] = o * tanh(c_);
        }
        cx += C; xact += x_offset; c += C; h += C;
    }
}

template <> void LSTMCellGrad<float, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const float*            cx,
    const float*            xact,
    const float*            c,
    const float*            dc,
    const float*            dh,
    float*                  dcx,
    float*                  dx) {
    float i, f, o, g, tanh_c, dcx_sum_term;
    int f_offset = C,
            o_offset = 2 * C,
                c_offset = 3 * C,
                    x_offset = 4 * C;
    for (int n = 0; n < N; ++n) {
        for (int idx = 0; idx < C; ++idx) {
            i = xact[idx];
            f = xact[idx + f_offset];
            o = xact[idx + o_offset];
            g = xact[idx + c_offset];
            //  BPTT compute the dc_{t-1} at the time of t
            //  dc_{t-1} =  dl / d(h_{t}) * d(h_{t}) / d(c_{t}) * d(c_{t}) / d(c_{t-1})
            //                  + d(c_{t+1}) / d(c_{t}) * d(c_{t}) / d(c_{t-1})
            //           =  (dl / d(h_{t}) * d(h_{t}) / d(c_{t}) + d(c_{t+1}) / d(c_{t}))
            //                  * d(c_{t}) / d(c_{t-1})
            tanh_c = tanh(c[idx]);
            dcx_sum_term = dh[idx] * o * (1 - tanh_c * tanh_c) + dc[idx];
            dcx[idx] = dcx_sum_term * f;
            dx[idx] = dcx_sum_term * g * i * (1 - i);
            dx[idx + f_offset] = dcx_sum_term * cx[idx] * f * (1 - f);
            dx[idx + o_offset] = dh[idx] * tanh_c * o * (1 - o);
            dx[idx + c_offset] = dcx_sum_term * i * (1 - g * g);
        }
        cx += C; xact += x_offset; c += C; dc += C; dh += C;
        dcx += C; dx += x_offset;
    }
}

/******************** update.adam_update ********************/

template <typename T>
void _AdamUpdate(
    const int               count,
    const T                 lr,
    const T                 beta1,
    const T                 beta2,
    const T                 eps,
    T*                      g,
    T*                      m,
    T*                      v) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        T gi = g[i];
        T mi = m[i] = m[i] * beta1 + gi * (1 - beta1);
        T vi = v[i] = v[i] * beta2 + gi * gi * (1 - beta2);
        g[i] = lr * mi / (std::sqrt(vi) + eps);
    }
}

template <> void AdamUpdate<float, CPUContext>(
    const int               count,
    const float             lr,
    const float             beta1,
    const float             beta2,
    const float             eps,
    float*                  g,
    float*                  m,
    float*                  v) {
    _AdamUpdate<float>(count, lr, beta1, beta2, eps, g, m, v);
}

template <> void AdamUpdate<float16, CPUContext>(
    const int               count,
    const float             lr,
    const float             beta1,
    const float             beta2,
    const float             eps,
    float16*                g,
    float16*                m,
    float16*                v) {
    CPU_FP16_NOT_SUPPORTED;
}

/******************** update.nesterov_update ********************/

template <typename T>
void _NesterovUpdate(
    const int               count,
    const T                 lr,
    const T                 momentum,
    T*                      g,
    T*                      h) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        T hi = h[i];
        T hi_new = h[i] = momentum * hi + lr * g[i];
        g[i] = (1 + momentum) * hi_new - momentum * hi;
    }
}

template <> void NesterovUpdate<float, CPUContext>(
    const int               count,
    const float             lr,
    const float             momentum,
    float*                  g,
    float*                  h) {
    _NesterovUpdate<float>(count, lr, momentum, g, h);
}

template <> void NesterovUpdate<float16, CPUContext>(
    const int               count,
    const float             lr,
    const float             momentum,
    float16*                g,
    float16*                h) {
    CPU_FP16_NOT_SUPPORTED;
}

/******************** update.rmsprop_update ********************/

template <typename T>
void _RMSPropUpdate(
    const int               count,
    const T                 lr,
    const T                 decay,
    const T                 eps,
    T*                      g,
    T*                      h) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        T gi = g[i];
        T hi = h[i] = decay * h[i] + (1 - decay) * gi * gi;
        g[i] = lr * g[i] / (std::sqrt(hi) + eps);
    }
}

template <> void RMSPropUpdate<float, CPUContext>(
    const int               count,
    const float             lr,
    const float             decay,
    const float             eps,
    float*                  g,
    float*                  h) {
    _RMSPropUpdate<float>(count, lr, decay, eps, g, h);
}

template <> void RMSPropUpdate<float16, CPUContext>(
    const int               count,
    const float             lr,
    const float             decay,
    const float             eps,
    float16*                g,
    float16*                h) {
    CPU_FP16_NOT_SUPPORTED;
}

/******************** update.sgd_update ********************/

template <typename T>
void _SGDUpdate(
    const int               count,
    const T                 lr,
    const T                 momentum,
    T*                      g,
    T*                      h) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        T hi = h[i];
        g[i] = h[i] = momentum * hi + lr * g[i];
    }
}

template <> void SGDUpdate<float, CPUContext>(
    const int               count,
    const float             lr,
    const float             momentum,
    float*                  g,
    float*                  h) {
    _SGDUpdate<float>(count, lr, momentum, g, h);
}

template <> void SGDUpdate<float16, CPUContext>(
    const int               count,
    const float             lr,
    const float             momentum,
    float16*                g,
    float16*                h) {
    CPU_FP16_NOT_SUPPORTED;
}

/******************** vision.bias_add ********************/

template<> void BiasAdd<float, CPUContext>(
    const int               count,
    const int               outer_dim,
    const int               dim,
    const int               inner_dim,
    const string&           data_format,
    const float*            bias,
    const float*            bias_multiplier,
    float*                  y,
    CPUContext*             ctx) {
    const int y_offset = dim * inner_dim;
    for (int n = 0; n < outer_dim; ++n) {
        if (data_format == "NCHW") {
            math::Gemm<float, CPUContext>(
                CblasNoTrans, CblasNoTrans,
                    dim, inner_dim, 1,
                        1.0, bias, bias_multiplier,
                            1.0, y, ctx);
        } else if (data_format == "NHWC") {
            math::Gemm<float, CPUContext>(
                CblasNoTrans, CblasNoTrans,
                    inner_dim, dim, 1,
                        1.0, bias_multiplier, bias,
                            1.0, y, ctx);
        } else LOG(FATAL) << "Unknown data format: " << data_format;
        y += y_offset;
    }
}

/******************** vision.bilinear_resize ********************/

template <typename T>
void _BilinearResize_NCHW(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                x,
    T*                      y) {
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
void _BilinearResize_NHWC(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                x,
    T*                      y) {
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

template <> void BilinearResize<float, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const string&           data_format,
    const float*            x,
    float*                  y) {
    const float scale_h = (float)H / out_h;
    const float scale_w = (float)W / out_w;
    if (data_format == "NCHW") {
        _BilinearResize_NCHW<float>(
            N, C, H, W, out_h, out_w,
                scale_h, scale_w, x, y);
    } else if (data_format == "NHWC"){
        _BilinearResize_NHWC<float>(
            N, C, H, W, out_h, out_w,
                scale_h, scale_w, x, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template <typename T>
void _BilinearResizeGrad_NCHW(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                dy,
    T*                      dx) {
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
                    dx[NCHT * W + left_x_idx] +=
                        static_cast<T>((1 - x_lerp) * dtop);
                    dx[NCHT * W + right_x_idx] +=
                        static_cast<T>(x_lerp * dtop);
                    dx[NCHB * W + left_x_idx] +=
                        static_cast<T>((1 - x_lerp) * dbottom);
                    dx[NCHB * W + right_x_idx] += 
                        static_cast<T>(x_lerp * dbottom);
                }
            }
        }
    }
}

template <typename T>
void _BilinearResizeGrad_NHWC(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                dy,
    T*                      dx) {
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
                    dx[(NHT * W + left_x_idx) * C + c] +=
                        static_cast<T>((1 - x_lerp) * dtop);
                    dx[(NHT * W + right_x_idx) * C + c] +=
                        static_cast<T>(x_lerp * dtop);
                    dx[(NHB * W + left_x_idx) * C + c] +=
                        static_cast<T>((1 - x_lerp) * dbottom);
                    dx[(NHB * W + right_x_idx) * C + c] += 
                        static_cast<T>(x_lerp * dbottom);
                }
            }
        }
    }
}

template <> void BilinearResizeGrad<float, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const string&           data_format,
    const float*            dy,
    float*                  dx) {
    const float scale_h = (float)H / out_h;
    const float scale_w = (float)W / out_w;
    math::Set<float, CPUContext>(N * C * H * W, 0, dx);
    if (data_format == "NCHW") {
        _BilinearResizeGrad_NCHW<float>(
            N, C, H, W, out_h, out_w,
                scale_h, scale_w, dy, dx);
    } else if (data_format == "NHWC"){
        _BilinearResizeGrad_NHWC<float>(
            N, C, H, W, out_h, out_w,
                scale_h, scale_w, dy, dx);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/******************** vision.conv ********************/

template<typename T>
void _Im2Col2d_NCHW(
    const int               C,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const T*                im,
    T*                      col) {
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
void _Im2Col2d_NHWC(
    const int               C,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const T*                im,
    T*                      col) {
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

template <> void Im2Col2d<float, CPUContext>(
    const int               C,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const string&           data_format,
    const float*            im,
    float*                  col) {
    if (data_format == "NCHW") {
        const int count = (C * col_h * col_w);
        _Im2Col2d_NCHW<float>(
            C, H, W, col_h, col_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w,
                    dilation_h, dilation_w, im, col);
    } else if (data_format == "NHWC") {
        const int count = (col_h * col_w * C);
        _Im2Col2d_NHWC<float>(
            C, H, W, col_h, col_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w,
                    dilation_h, dilation_w, im, col);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template<typename T>
void _Col2Im2d_NCHW(
    const int               C,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const T*                col,
    T*                      im) {
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
void _Col2Im2d_NHWC(
    const int               C,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const T*                col,
    T*                      im) {
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

template<> void Col2Im2d<float, CPUContext>(
    const int               C,
    const int               H,
    const int               W,
    const int               col_h,
    const int               col_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const int               dilation_h,
    const int               dilation_w,
    const string&           data_format,
    const float*            col,
    float*                  im) {
    if (data_format == "NCHW") {
        const int count = (C * H * W);
        _Col2Im2d_NCHW<float>(
            C, H, W, col_h, col_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w,
                    dilation_h, dilation_w, col, im);
    } else if (data_format == "NHWC") {
        const int count = (H * W * C);
        _Col2Im2d_NHWC<float>(
            C, H, W, col_h, col_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w,
                    dilation_h, dilation_w, col, im);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/******************** vision.nn_resize ********************/

template <typename T>
void _NNResize_NCHW(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                x,
    T*                      y) {
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
void _NNResize_NHWC(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                x,
    T*                      y) {
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

template <> void NNResize<float, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const string&           data_format,
    const float*            x,
    float*                  y) {
    const float scale_h = (float)H / out_h;
    const float scale_w = (float)W / out_w;
    if (data_format == "NCHW") {
        _NNResize_NCHW<float>(
            N, C, H, W, out_h, out_w,
                scale_h, scale_w, x, y);
    } else if (data_format == "NHWC"){
        _NNResize_NHWC<float>(
            N, C, H, W, out_h, out_w,
                scale_h, scale_w, x, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template <typename T>
void _NNResizeGrad_NCHW(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                dy,
    T*                      dx) {
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
void _NNResizeGrad_NHWC(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const float             scale_h,
    const float             scale_w,
    const T*                dy,
    T*                      dx) {
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

template <> void NNResizeGrad<float, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               out_h,
    const int               out_w,
    const string&           data_format,
    const float*            dy,
    float*                  dx) {
    const float scale_h = (float)H / out_h;
    const float scale_w = (float)W / out_w;
    math::Set<float, CPUContext>(N * C * H * W, 0, dx);
    if (data_format == "NCHW") {
        _NNResizeGrad_NCHW<float>(
            N, C, H, W, out_h, out_w,
                scale_h, scale_w, dy, dx);
    } else if (data_format == "NHWC"){
        _NNResizeGrad_NHWC<float>(
            N, C, H, W, out_h, out_w,
                scale_h, scale_w, dy, dx);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/******************** vision.pooling ********************/

template <typename T>
void _MAXPooling2d_NCHW(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const float*            x,
    int*                    mask,
    float*                  y) {
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
void _MAXPooling2d_NHWC(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const float*            x,
    int*                    mask,
    float*                  y) {
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

template<> void MAXPooling2d<float, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const string&           data_format,
    const float*            x,
    int*                    mask,
    float*                  y) {
    if (data_format == "NCHW") {
        _MAXPooling2d_NCHW<float>(
            N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, x, mask, y);
    } else if (data_format == "NHWC") {
        _MAXPooling2d_NHWC<float>(
            N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, x, mask, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template<typename T>
void _AVGPooling2d_NCHW(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const float*            x,
    float*                  y) {
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
void _AVGPooling2d_NHWC(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const float*            x,
    float*                  y) {
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

template<> void AVGPooling2d<float, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const string&           data_format,
    const float*            x,
    float* y) {
    if (data_format == "NCHW") {
        _AVGPooling2d_NCHW<float>(
            N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, x, y);
    } else if (data_format == "NHWC") {
        _AVGPooling2d_NHWC<float>(
            N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, x, y);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template <typename T>
void _MAXPooling2dGrad_NCHW(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const float*            dy,
    const int*              mask,
    float*                  dx) {
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
void _MAXPooling2dGrad_NHWC(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const float*            dy,
    const int*              mask,
    float*                  dx) {
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

template<> void MAXPooling2dGrad<float, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const string&           data_format,
    const float*            dy,
    const int*              mask,
    float*                  dx) {
   if (data_format == "NCHW") {
        _MAXPooling2dGrad_NCHW<float>(
            N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, dy, mask, dx);
    } else if (data_format == "NHWC") {
        _MAXPooling2dGrad_NHWC<float>(
            N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, dy, mask, dx);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

template <typename T>
void _AVGPooling2dGrad_NCHW(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const float*            dy,
    float*                  dx) {
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
void _AVGPooling2dGrad_NHWC(
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const float*            dy,
    float*                  dx) {
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

template<> void AVGPooling2dGrad<float, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               kernel_h,
    const int               kernel_w,
    const int               stride_h,
    const int               stride_w,
    const int               pad_h,
    const int               pad_w,
    const string&           data_format,
    const float*            dy,
    float*                  dx) {
    if (data_format == "NCHW") {
        _AVGPooling2dGrad_NCHW<float>(
            N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, dy, dx);
    } else if (data_format == "NHWC") {
        _AVGPooling2dGrad_NHWC<float>(
            N, C, H, W, pool_h, pool_w, kernel_h, kernel_w,
                stride_h, stride_w, pad_h, pad_w, dy, dx);
    } else LOG(FATAL) << "Unknown data format: " << data_format;
}

/******************** vision.roi_pooling ********************/

template<> void ROIPooling<float, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const float*            x,
    const float*            rois,
    int*                    mask,
    float*                  y) {
    const TIndex x_offset = H * W,
                     y_offset = pool_h * pool_w,
                         im_offset = C * H * W;
    math::Set<float, CPUContext>(count, -FLT_MAX, y);
    math::Set<int, CPUContext>(count, -1, mask);
    for (int n = 0; n < num_rois; ++n) {
        int im_idx = rois[0];
        int x1 = round(rois[1] * spatial_scale);
        int y1 = round(rois[2] * spatial_scale);
        int x2 = round(rois[3] * spatial_scale);
        int y2 = round(rois[4] * spatial_scale);
        int roi_height = std::max(y2 - y1 + 1, 1);
        int roi_width = std::max(x2 - x1 + 1, 1);
        const float unit_h = (float)roi_height / (float)pool_h;
        const float unit_w = (float)roi_width / (float)pool_w;
        const float* Idata = x + im_idx * im_offset;
        for (int c = 0; c < C; ++c) {
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
                    start_h = std::min(start_h, H);
                    start_w = std::min(start_w, W);
                    end_h = std::min(end_h, H);
                    end_w = std::min(end_w, W);
                    bool is_empty = (end_h == start_h) || (end_w == start_w);
                    const int pool_idx = ph * pool_w + pw;
                    if (is_empty) {
                        y[pool_idx] = 0;
                        mask[pool_idx] = -1;
                    }
                    for (int h = start_h; h < end_h; ++h) {
                        for (int w = start_w; w < end_w; ++w) {
                            const int idx = h * W + w;
                            if (Idata[idx] > y[pool_idx]) {
                                y[pool_idx] = Idata[idx];
                                mask[pool_idx] = idx;
                            }
                        }    //  end w
                    }    //  end h
                }    //  end pw
            }    //  end ph
            //  offset image channels
            x += x_offset;
            y += y_offset;
            mask += y_offset;
        }    //  end c
        //  offset roi region
        rois += 5;
    }    //  end n
}

template<> void ROIPoolingGrad<float, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const float*            dy,
    const float*            rois,
    const int*              mask,
    float*                  dx) {
    NOT_IMPLEMENTED;
}

/******************** vision.roi_align ********************/

template<> void ROIAlign<float, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const int               sampling_ratio,
    const float*            x,
    const float*            rois,
    float*                  y) {
    NOT_IMPLEMENTED;
}

template<> void ROIAlignGrad<float, CPUContext>(
    const int               count,
    const int               N,
    const int               C,
    const int               H,
    const int               W,
    const int               pool_h,
    const int               pool_w,
    const int               num_rois,
    const float             spatial_scale,
    const int               sampling_ratio,
    const float*            dy,
    const float*            rois,
    float*                  dx) {
    NOT_IMPLEMENTED;
}

}    // namespace kernel

}    // namespace dragon