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
                                           const string& format, 
                                           const float* bias, 
                                           const float* bias_multiplier, 
                                           float* y) {
    if (format == "NCHW") {
        const int y_offset = dim * inner_dim;
        for (int n = 0; n < outer_dim; ++n) {
            math::Gemm<float, CPUContext>(CblasNoTrans, CblasNoTrans, 
                                                   dim, inner_dim, 1, 
                                                                 1.0, 
                                               bias, bias_multiplier, 
                                                                 1.0, 
                                                                  y);
            y += y_offset;
        }
    } else {
        NOT_IMPLEMENTED;
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
        if (abs_val < 1.0 / sigma2) y[i] = 0.5 * val * val *sigma2;
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

/******************** misc.memory_data ********************/

template <> void MemoryData<float, float, CPUContext>(const int count, 
                                                      const int num, 
                                                      const int channels,
                                                      const int height, 
                                                      const int width,
                                                      const float* x, 
                                                      float* y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const int w = i % width;
        const int h = (i / width) % height;
        const int c = (i / width / height) % channels;
        const int n = i / width / height / channels;
        const int x_idx = ((n * height + h) * width + w) * channels + c;
        if (c == 0) y[i] = x[x_idx] - 102.9801;
        else if (c == 1) y[i] = x[x_idx] - 115.9465;
        else y[i] = x[x_idx] - 122.7717;
    }
}

template <> void MemoryData<uint8_t, float, CPUContext>(const int count, 
                                                        const int num, 
                                                        const int channels,
                                                        const int height, 
                                                        const int width,
                                                        const uint8_t* x, 
                                                        float* y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const int w = i % width;
        const int h = (i / width) % height;
        const int c = (i / width / height) % channels;
        const int n = i / width / height / channels;
        const int x_idx = ((n * height + h) * width + w) * channels + c;
        if (c == 0) y[i] = x[x_idx] - 102.9801;
        else if (c == 1) y[i] = x[x_idx] - 115.9465;
        else y[i] = x[x_idx] - 122.7717;
    }
}

template <> void MemoryData<float, float16, CPUContext>(const int count, 
                                                        const int num, 
                                                        const int channels,
                                                        const int height, 
                                                        const int width,
                                                        const float* x, 
                                                        float16* y) {
    LOG(FATAL) << "float16 is unsupported for CPUContext.";
}

template <> void MemoryData<uint8_t, float16, CPUContext>(const int count, 
                                                          const int num, 
                                                          const int channels,
                                                          const int height, 
                                                          const int width,
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

template <> void CanonicalAxis<float, CPUContext>(const int count, const int dim, float* y) {
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) if (y[i] < 0) y[i] += dim;
}

template <> void At<float, CPUContext>(const int count, 
                                       const int outer_dim, 
                                       const int inner_dim,
                                       const int x_slice_dim, 
                                       const int y_slice_dim, 
                                       const float* indices,
                                       const float* x, 
                                       float* y, 
                                       CPUContext* context) {
    TIndex x_offset, y_offset, x_idx_offset, y_idx_offset;
    for (int i = 0; i < y_slice_dim; ++i) {
        y_idx_offset = i;
        x_idx_offset = indices[y_idx_offset];
        for (int n = 0; n < outer_dim; ++n) {
            x_offset = (n * x_slice_dim + x_idx_offset) * inner_dim;
            y_offset = (n * y_slice_dim + y_idx_offset) * inner_dim;
            context->Copy<float, CPUContext, CPUContext>(inner_dim, 
                                                         y + y_offset, 
                                                         x + x_offset);
        }
    }
}

template <> void AtGrad<float, CPUContext>(const int count, 
                                           const int outer_dim,
                                           const int inner_dim,
                                           const int x_slice_dim, 
                                           const int y_slice_dim, 
                                           const float* indices,
                                           const float* dy, 
                                           float* dx, 
                                           CPUContext* context) {
    TIndex x_offset, y_offset, x_idx_offset, y_idx_offset;
    for (int i = 0; i < y_slice_dim; ++i) {
        y_idx_offset = i;
        x_idx_offset = indices[y_idx_offset];
        for (int n = 0; n < outer_dim; ++n) {
            x_offset = (n * x_slice_dim + x_idx_offset) * inner_dim;
            y_offset = (n * y_slice_dim + y_idx_offset) * inner_dim;
            math::Add<float, CPUContext>(inner_dim, 
                                         dy + y_offset, 
                                         dx + x_offset, 
                                         dx + x_offset);
        }
    }
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

template<> void Crop2D<float, CPUContext>(vector<TIndex> idxs,
                                          const vector<TIndex>& offsets, 
                                          const int cur_dim, 
                                          Tensor* x,
                                          Tensor* y,
                                          CPUContext* context) {
    //  run as Crop1D
    auto* Xdata = x->data<float, CPUContext>();
    auto* Ydata = y->mutable_data<float, CPUContext>();

    for (int i = 0; i < y->dim(cur_dim); ++i) {
        vector<TIndex> idx_off(cur_dim + 1, 0);
        for (int j = 0; j < cur_dim; j++) idx_off[j] = idxs[j] + offsets[j];
        idx_off[cur_dim] = offsets[cur_dim];
        context->Copy<float, CPUContext, CPUContext>(y->dim(cur_dim),
                                                     Ydata + y->offset(idxs),
                                                     Xdata + x->offset(idx_off));
     }
}

template<> void Crop2DGrad<float, CPUContext>(vector<TIndex> idxs,
                                              const vector<TIndex>& offsets, 
                                              const int cur_dim, 
                                              Tensor* dy,
                                              Tensor* dx,
                                              CPUContext* context) {
    //  run as Crop1D
    auto* dYdata = dy->data<float, CPUContext>();
    auto* dXdata = dx->mutable_data<float, CPUContext>();

    for (int i = 0; i < dy->dim(cur_dim); ++i) {
        vector<TIndex> idx_off(cur_dim + 1, 0);
        for (int j = 0; j < cur_dim; j++) idx_off[j] = idxs[j] + offsets[j];
        idx_off[cur_dim] = offsets[cur_dim];
        context->Copy<float, CPUContext, CPUContext>(dy->dim(cur_dim),
                                                     dXdata + dx->offset(idx_off),
                                                     dXdata + dy->offset(idxs));
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
        }  //  end ch
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
        }    //  end ch
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

/******************** vision.nn_resize ********************/

template <> void BilinearResize<float, CPUContext>(const int count, 
                                                   const int num, const int channels,
                                                   const int h_in, const int w_in, 
                                                   const int h_out, const int w_out,
                                                   const float* x, 
                                                   float* y) {
    const float h_scale = (float)h_in / h_out;
    const float w_scale = (float)w_in / w_out;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const int w = i % w_out;
        const int h = (i / w_out) % h_out;
        const int c = (i / w_out / h_out) % channels;
        const int n = i / w_out / h_out / channels;

        const float in_h = h * h_scale;
        const int top_y_idx = floorf(in_h);
        const int bottom_y_idx = (in_h < h_in - 1) ? ceilf(in_h) : h_in - 1;
        const float y_lerp = in_h - top_y_idx;

        const float in_w = w * w_scale;
        const int left_x_idx = floorf(in_w);
        const int right_x_idx = (in_w < w_in - 1) ? ceilf(in_w) : w_in - 1;
        const float x_lerp = in_w - left_x_idx;

        const float top_left(x[((n * channels + c) * h_in + top_y_idx) * w_in + left_x_idx]);
        const float top_right(x[((n * channels + c) * h_in + top_y_idx) * w_in + right_x_idx]);
        const float bottom_left(x[((n * channels + c) * h_in + bottom_y_idx) * w_in + left_x_idx]);
        const float bottom_right(x[((n * channels + c) * h_in + bottom_y_idx) * w_in + right_x_idx]);

        const float top = top_left + (top_right - top_left) * x_lerp;
        const float bottom = bottom_left + (bottom_right - bottom_left) * x_lerp;
        y[i] = top + (bottom - top) * y_lerp;
    }
}

template <> void BilinearResizeGrad<float, CPUContext>(const int count,
                                                       const int num, const int channels,
                                                       const int h_in, const int w_in, 
                                                       const int h_out, const int w_out,
                                                       const float* dy, 
                                                       float* dx) {
    const float h_scale = (float)h_out / h_in;
    const float w_scale = (float)w_out / w_in;

    for (int i = 0; i < count; i++) {
        const int w = i % w_in;
        const int h = (i / w_in) % h_in;
        const int c = (i / w_in / h_in) % channels;
        const int n = i / w_in / h_in / channels;

        const float original_h = h * h_scale;
        const int top_y_idx = floorf(original_h);
        const int bottom_y_idx = (original_h < h_out - 1) ? ceilf(original_h) : h_out - 1;
        const float y_lerp = original_h - top_y_idx;

        const float original_w = w * w_scale;
        const int left_x_idx = floorf(original_w);
        const int right_x_idx = (original_w < w_out - 1) ? ceilf(original_w) : w_out - 1;
        const float x_lerp = original_w - left_x_idx;

        const float dtop = (1 - y_lerp) * dy[i];
        *(dx + ((n * channels + c) * h_out + top_y_idx) * w_out + left_x_idx)
            += static_cast<float>((1 - x_lerp) * dtop);
        *(dx + ((n * channels + c) * h_out + top_y_idx) * w_out + right_x_idx)
            += static_cast<float>(x_lerp * dtop);

        const float dbottom = y_lerp * dy[i];
        *(dx + ((n * channels + c) * h_out + bottom_y_idx) * w_out + left_x_idx)
            += static_cast<float>((1 - x_lerp) * dbottom);
        *(dx + ((n * channels + c) * h_out + bottom_y_idx) * w_out + right_x_idx)
            += static_cast<float>(x_lerp * dbottom);
    }
}

/******************** vision.conv ********************/

template <> void Im2Col<float, CPUContext>(const int channels, 
                                           const int height, const int width,
                                           const int kernel_h, const int kernel_w, 
                                           const int stride_h, const int stride_w, 
                                           const int pad_h, const int pad_w,
                                           const int dilation_h, const int dilation_w, 
                                           const float* im,
                                           float* col) {
    const int col_h = (height + 2 * pad_h - (dilation_h*(kernel_h - 1) + 1)) / stride_h + 1;
    const int col_w = (width + 2 * pad_w - (dilation_w*(kernel_w - 1) + 1)) / stride_w + 1;
    const int input_spatial = height * width;

    //  for each element in kernel, create a row-col-map for a input feature map
    for (int channel = 0; channel < channels; ++channel, im += input_spatial) {
        for (int kh_off = 0; kh_off < kernel_h; ++kh_off) {
            for (int kw_off = 0; kw_off < kernel_w; ++kw_off) {
                int input_row = -pad_h + kh_off * dilation_h;
                //  scan all output pixels and find the corresponding input pixels
                for (int output_row = 0; output_row < col_h; ++output_row ) {
                    //  set '0' for all output pixels out of the input map
                    if (!judge(input_row, height)) {
                        for (int output_col = 0; output_col < col_w; ++output_col) *(col++) = 0;
                    } else {    //  find the corresponding input pixels
                        int input_col = -pad_w + kw_off * dilation_w;
                        for (int output_col = 0; output_col < col_w; ++output_col) {
                            if (!judge(input_col, width)) *(col++) = 0;
                            else *(col++) = im[input_row * width + input_col];
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }    //  end output_row
            }    //  end kw_off
        }    //  end kh_off
    }    //  end channel
}

template<> void Col2Im<float, CPUContext>(const int channels, 
                                          const int height, const int width,
                                          const int kernel_h, const int kernel_w, 
                                          const int stride_h, const int stride_w,
                                          const int pad_h, const int pad_w,
                                          const int dilation_h, const int dilation_w, 
                                          const float* col,
                                          float* im) {
    //  must memset before use '+='
    math::Set<float, CPUContext>(channels * height * width, 0, im);
    const int col_h = (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
    const int col_w = (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
    const int input_spatial = height * width;

    //  for each element in kernel, create a row-col-map for a input feature map
    for (int channel = 0; channel < channels; ++channel, im += input_spatial) {
        for (int kh_off = 0; kh_off < kernel_h; ++kh_off) {
            for (int kw_off = 0; kw_off < kernel_w; ++kw_off) {
                int input_row = -pad_h + kh_off * dilation_h;
                //  scan all output pixels and find the corresponding input pixels
                for (int output_row = 0; output_row < col_h; ++output_row) {
                    //  skip the num of col_w pixels
                    if (!judge(input_row, height)) {
                        col += col_w;
                    } else {    //  find the corresponding input pixels
                        int input_col = -pad_w + kw_off * dilation_w;
                        for (int output_col = 0; output_col < col_w; output_col++) {
                            if (judge(input_col, width)) im[input_row * width + input_col] += *col;
                            ++col;
                            input_col += stride_w;
                        }
                    }
                    input_row += stride_h;
                }    //  end output_row
            }    //  end kw_off
        }    //  end kh_off
    }    //  end channel
}

/******************** vision.nn_resize ********************/

template <> void NNResize<float, CPUContext>(const int count, 
                                             const int num, const int channels,
                                             const int h_in, const int w_in, 
                                             const int h_out, const int w_out,
                                             const float* x, 
                                             float* y) {
    const float h_scale = (float)h_in / h_out;
    const float w_scale = (float)w_in / w_out;
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        const int w = i % w_out;
        const int h = (i / w_out) % h_out;
        const int in_h = std::min(int(floorf(h * h_scale)), h_in - 1);
        const int in_w = std::min(int(floorf(w * w_scale)), w_in - 1);
        const int c = (i / w_out / h_out) % channels;
        const int n = i / w_out / h_out / channels;
        const int x_idx = ((n * channels + c) * h_in + in_h) * w_in + in_w;
        y[i] = x[x_idx];
    }
}

template <> void NNResizeGrad<float, CPUContext>(const int count, 
                                                 const int num, const int channels,
                                                 const int h_in, const int w_in, 
                                                 const int h_out, const int w_out,
                                                 const float* dy, 
                                                 float* dx) {
    const float h_scale = (float)h_out / h_in;
    const float w_scale = (float)w_out / w_in;
    for (int n = 0; n < num; n++) {
        for (int c = 0; c < channels; ++c) {
            for (int h = 0; h < h_in; ++h) {
                const int out_h = std::min(int(floorf(h * h_scale)), (h_out - 1));
                for (int w = 0; w < w_in; ++w) {
                    const int out_w = std::min(int(floorf(w * w_scale)), (w_out - 1));
                    const int y_idx = ((n * channels + c) * h_in + h) * w_in + w;
                    const int x_idx = ((n * channels + c) * h_out + out_h) * w_out + out_w;
                    dx[x_idx] += dy[y_idx];
                }
            }
        }
    }
}

/******************** vision.pooling ********************/

template<> void MAXPooling<float, CPUContext>(const int count, 
                                              const int num, const int channels,
                                              const int height, const int width, 
                                              const int pool_height, const int pool_width,
                                              const int kernel_h, const int kernel_w, 
                                              const int stride_h, const int stride_w,
                                              const int pad_h, const int pad_w,
                                              const float* x, 
                                              int* mask, 
                                              float* y) {
    int x_offset = height * width;
    int y_offset = pool_height * pool_width;
    for (int n = 0; n < num; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int ph = 0; ph < pool_height; ++ph) {
                for (int pw = 0; pw < pool_width; ++pw) {
                    int start_h = ph * stride_h - pad_h;
                    int start_w = pw * stride_w - pad_w;
                    int end_h = std::min(start_h + kernel_h, height);
                    int end_w = std::min(start_w + kernel_w, width);
                    start_h = std::max(start_h, 0);
                    start_w = std::max(start_w, 0);
                    const int pool_idx = ph * pool_width + pw;
                    float max_val = -FLT_MAX;
                    int max_idx = -1;
                    for (int h = start_h; h < end_h; ++h) {
                        for (int w = start_w; w < end_w; ++w) {
                            const int idx = h * width + w;
                            if (x[idx]>max_val) {
                                max_val = x[idx];
                                max_idx = idx;
                            }
                        }    //  end w
                    }    //  end h
                    y[pool_idx] = max_val;
                    mask[pool_idx] = max_idx;
                }    //  end pw
            }    //  end ph
            //  offset a channel
            x += x_offset;
            y += y_offset;
            mask += y_offset;
        }    //  end c
    }    //  end n
}

template<> void AVEPooling<float, CPUContext>(const int count, 
                                              const int num, const int channels,
                                              const int height, const int width, 
                                              const int pool_height, const int pool_width,
                                              const int kernel_h, const int kernel_w, 
                                              const int stride_h, const int stride_w, 
                                              const int pad_h, const int pad_w,
                                              const float* x, 
                                              float* y) {
    int x_offset = height * width;
    int y_offset = pool_height * pool_width;
    math::Set<float, CPUContext>(count, 0, y);
    for (int n = 0; n < num; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int ph = 0; ph < pool_height; ++ph) {
                for (int pw = 0; pw < pool_width; ++pw) {
                    int start_h = ph * stride_h - pad_h;
                    int start_w = pw * stride_w - pad_w;
                    int end_h = std::min(start_h + kernel_h, height + pad_h);
                    int end_w = std::min(start_w + kernel_w, width + pad_w);
                    int pool_size = (end_h - start_h) * (end_w - start_w);
                    end_h = std::min(end_h, height);
                    end_w = std::min(end_w, width);
                    start_h = std::max(start_h, 0);
                    start_w = std::max(start_w, 0);
                    const int pool_idx = ph * pool_width + pw;
                    for (int h = start_h; h < end_h; ++h) {
                        for (int w = start_w; w < end_w; ++w) {
                            const int idx = h * width + w;
                            y[pool_idx] += x[idx];
                        }
                    }
                    y[pool_idx] /= pool_size;
                }    //end pw
            }    //end ph
            x += x_offset;
            y += y_offset;
        }    //end c
    }    //end n
}

template<> void MAXPoolingGrad<float, CPUContext>(const int count, 
                                                  const int num, const int channels,
                                                  const int height, const int width, 
                                                  const int pool_height, const int pool_width,
                                                  const int kernel_h, const int kernel_w, 
                                                  const int stride_h, const int stride_w, 
                                                  const int pad_h, const int pad_w,
                                                  const float* dy, 
                                                  const int* mask, 
                                                  float* dx) {
    int x_offset = height * width;
    int y_offset = pool_height * pool_width;
    math::Set<float, CPUContext>(count, 0, dx);
    for (int n = 0; n < num; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int ph = 0; ph < pool_height; ++ph) {
                    for (int pw = 0; pw < pool_width; ++pw) {
                        const int pool_idx = ph * pool_width + pw;
                        const int idx = mask[pool_idx];
                        dx[idx] += dy[pool_idx];
                    }    //  end pw
            }    //  end ph
            dx += x_offset;
            dy += y_offset;
            mask += y_offset;
        }    //  end c
    }    //  end n
}

template<> void AVEPoolingGrad<float, CPUContext>(const int count, 
                                                  const int num, const int channels,
                                                  const int height, const int width, 
                                                  const int pool_height, const int pool_width,
                                                  const int kernel_h, const int kernel_w, 
                                                  const int stride_h, const int stride_w, 
                                                  const int pad_h, const int pad_w,
                                                  const float* dy, 
                                                  float* dx) {
    int x_offset = height * width;
    int y_offset = pool_height * pool_width;
    math::Set<float, CPUContext>(count, 0, dx);
    for (int n = 0; n < num; ++n) {
        for (int c = 0; c < channels; ++c) {
            for (int ph = 0; ph < pool_height; ++ph) {
                for (int pw = 0; pw < pool_width; ++pw) {
                    int start_h = ph * stride_h - pad_h;
                    int start_w = pw * stride_w - pad_w;
                    int end_h = std::min(start_h + kernel_h, height + pad_h);
                    int end_w = std::min(start_w + kernel_w, width + pad_w);
                    int pool_size = (end_h - start_h)*(end_w - start_w);
                    end_h = std::min(end_h, height);
                    end_w = std::min(end_w, width);
                    start_h = std::max(start_h, 0);
                    start_w = std::max(start_w, 0);
                    const int pool_idx = ph * pool_width + pw;
                    for (int h = start_h; h < end_h; ++h) {
                        for (int w = start_w; w < end_w; ++w) {
                            const int idx = h * width + w;
                            dx[idx] += (dy[pool_idx] / pool_size);
                        }
                    }
                }    //  end pw
            }    //  end ph
            dx += x_offset;
            dy += y_offset;
        }    //  end c
    }    //  end n
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
        Rdata += roi->offset(1);
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
                                            Tensor* mask_h, Tensor* mask_w,
                                            Tensor* y) {
    NOT_IMPLEMENTED;
}

template<> void ROIAlignGrad<float, CPUContext>(const float spatial_scale, 
                                                const int pool_h, const int pool_w,
                                                Tensor* dy,
                                                Tensor* roi,
                                                Tensor* mask_h, Tensor* mask_w,
                                                Tensor* dx) {
    NOT_IMPLEMENTED;
}

}    // namespace kernel

}    // namespace dragon