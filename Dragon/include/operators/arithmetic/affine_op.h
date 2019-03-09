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

#ifndef DRAGON_OPERATORS_ARITHMETIC_AFFINE_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_AFFINE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class AffineOp final : public Operator<Context> {
 public:
    AffineOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int64_t>("axis", 1)),
          num_axes(OperatorBase::Arg<int64_t>("num_axes", 1)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    int64_t axis, num_axes;
    int64_t outer_dim, scale_dim, inner_dim;
};

template <class Context>
class AffineGradientOp final : public Operator<Context> {
 public:
    AffineGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int64_t>("axis", 1)),
          num_axes(OperatorBase::Arg<int64_t>("num_axes", 1)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void BiasRunWithType();
    template <typename T> void ScaleRunWithType();
    template <typename T> void ComputeScaleGradient(T* dYxX, T* dA);
    template <typename T> void RunWithType();

 protected:
    int64_t axis, num_axes;
    int64_t outer_dim, inner_dim, scale_dim, sum_dim, dim;
};

#ifdef WITH_CUDNN

#if CUDNN_VERSION_MIN(6, 0, 0)

template <class Context>
class CuDNNAffineOpBase : public Operator<Context> {
 public:
    CuDNNAffineOpBase(const OperatorDef& def, Workspace* ws)
         : Operator<Context>(def, ws),
           axis(OperatorBase::Arg<int64_t>("axis", 1)),
           num_axes(OperatorBase::Arg<int64_t>("num_axes", 1)) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&param_desc));
        CUDNN_CHECK(cudnnCreateOpTensorDescriptor(&mul_desc));
        CUDNN_CHECK(cudnnCreateOpTensorDescriptor(&add_desc));
        CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&reduce_desc));
    }
    USE_OPERATOR_FUNCTIONS;

    virtual ~CuDNNAffineOpBase() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(input_desc));
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(param_desc));
        CUDNN_CHECK(cudnnDestroyOpTensorDescriptor(mul_desc));
        CUDNN_CHECK(cudnnDestroyReduceTensorDescriptor(reduce_desc));
    }

    template <typename T>
    void ResetDesc(const Tensor& X);

    int64_t axis, num_axes;
    cudnnTensorDescriptor_t input_desc, param_desc;
    cudnnOpTensorDescriptor_t mul_desc, add_desc;
    cudnnReduceTensorDescriptor_t reduce_desc;
};

#define USE_CUDNN_AFFINE_FUCNTIONS \
    USE_OPERATOR_FUNCTIONS; \
    using CuDNNAffineOpBase<Context>::axis; \
    using CuDNNAffineOpBase<Context>::num_axes; \
    using CuDNNAffineOpBase<Context>::input_desc; \
    using CuDNNAffineOpBase<Context>::param_desc; \
    using CuDNNAffineOpBase<Context>::mul_desc; \
    using CuDNNAffineOpBase<Context>::add_desc; \
    using CuDNNAffineOpBase<Context>::reduce_desc

template <class Context>
class CuDNNAffineOp final : public CuDNNAffineOpBase<Context> {
 public:
    CuDNNAffineOp(const OperatorDef& def, Workspace* ws)
         : CuDNNAffineOpBase<Context>(def, ws) {}

    void RunOnDevice() override;
    template <typename DT, typename CT> void RunWithType();

 protected:
    USE_CUDNN_AFFINE_FUCNTIONS;
};

template <class Context>
class CuDNNAffineGradientOp final
    : public CuDNNAffineOpBase<Context> {
public:
    CuDNNAffineGradientOp(
        const OperatorDef&          def,
        Workspace*                  ws)
        : CuDNNAffineOpBase<Context>(def, ws) {}

    void RunOnDevice() override;

    template <typename DT, typename CT>
    void ComputeScaleGradient(DT* dYxX, DT* dA);
    template <typename T> void ComputeScaleGradient_v2(T* dYxX, T* dA);
    template <typename DT, typename CT> void RunWithType();

 protected:
    USE_CUDNN_AFFINE_FUCNTIONS;
    int64_t outer_dim, inner_dim, scale_dim, dim, sum_dim;
};

#endif

#endif  // WITH_CUDNN

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARITHMETIC_AFFINE_OP_H_