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
          axis_(OpArg<int64_t>("axis", 1)),
          num_axes_(OpArg<int64_t>("num_axes", 1)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();

 protected:
    int64_t outer_dim_, inner_dim_;
    int64_t axis_, num_axes_, scale_dim_;
};

template <class Context>
class AffineGradientOp final : public Operator<Context> {
 public:
    AffineGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis_(OpArg<int64_t>("axis", 1)),
          num_axes_(OpArg<int64_t>("num_axes", 1)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void Reduce(T* x, T* y);
    template <typename T> void RunImpl();

 protected:
    int64_t axis_, num_axes_;
    int64_t outer_dim_, inner_dim_;
    int64_t scale_dim_, reduce_dim_, dim_;
};

#ifdef WITH_CUDNN

#if CUDNN_VERSION_MIN(6, 0, 0)

template <class Context>
class CuDNNAffineOpBase : public Operator<Context> {
 public:
    CuDNNAffineOpBase(const OperatorDef& def, Workspace* ws)
         : Operator<Context>(def, ws),
           axis_(OpArg<int64_t>("axis", 1)),
           num_axes_(OpArg<int64_t>("num_axes", 1)) {
        CuDNNCreateTensorDesc(&input_desc_);
        CuDNNCreateTensorDesc(&param_desc_);
        CUDNN_CHECK(cudnnCreateOpTensorDescriptor(&mul_op_));
        CUDNN_CHECK(cudnnCreateOpTensorDescriptor(&add_op_));
        CUDNN_CHECK(cudnnCreateReduceTensorDescriptor(&reduce_desc_));
    }
    USE_OPERATOR_FUNCTIONS;

    virtual ~CuDNNAffineOpBase() {
        CuDNNDestroyTensorDesc(&input_desc_);
        CuDNNDestroyTensorDesc(&param_desc_);
        CUDNN_CHECK(cudnnDestroyOpTensorDescriptor(mul_op_));
        CUDNN_CHECK(cudnnDestroyOpTensorDescriptor(add_op_));
        CUDNN_CHECK(cudnnDestroyReduceTensorDescriptor(reduce_desc_));
    }

    template <typename T>
    void ResetDesc(const Tensor& X);

    int64_t axis_, num_axes_;
    cudnnTensorDescriptor_t input_desc_, param_desc_;
    cudnnOpTensorDescriptor_t mul_op_, add_op_;
    cudnnReduceTensorDescriptor_t reduce_desc_;
};

#define USE_CUDNN_AFFINE_FUCNTIONS \
    USE_OPERATOR_FUNCTIONS; \
    using CuDNNAffineOpBase<Context>::axis_; \
    using CuDNNAffineOpBase<Context>::num_axes_; \
    using CuDNNAffineOpBase<Context>::input_desc_; \
    using CuDNNAffineOpBase<Context>::param_desc_; \
    using CuDNNAffineOpBase<Context>::mul_op_; \
    using CuDNNAffineOpBase<Context>::add_op_; \
    using CuDNNAffineOpBase<Context>::reduce_desc_

template <class Context>
class CuDNNAffineOp final : public CuDNNAffineOpBase<Context> {
 public:
    CuDNNAffineOp(const OperatorDef& def, Workspace* ws)
         : CuDNNAffineOpBase<Context>(def, ws) {}

    void RunOnDevice() override;
    template <typename DT, typename CT> void RunImpl();

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
    void CuDNNReduce(DT* x, DT* y);
    template <typename T> void Reduce(T* x, T* y);
    template <typename DT, typename CT> void RunImpl();

 protected:
    USE_CUDNN_AFFINE_FUCNTIONS;
    int64_t outer_dim_, inner_dim_;
    int64_t scale_dim_, dim_, reduce_dim_;
};

#endif

#endif  // WITH_CUDNN

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARITHMETIC_AFFINE_OP_H_