// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_OPERATORS_ARITHMETIC_AFFINE_OP_H_
#define DRAGON_OPERATORS_ARITHMETIC_AFFINE_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class AffineOp final : public Operator<Context> {
 public:
    AffineOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int>("axis", 1)),
          num_axes(OperatorBase::Arg<int>("num_axes", 1)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    TIndex axis, start_axis, num_axes;
    TIndex outer_dim, scale_dim, inner_dim;
};

template <class Context>
class AffineGradientOp final : public Operator<Context> {
 public:
    AffineGradientOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          axis(OperatorBase::Arg<int>("axis", 1)),
          num_axes(OperatorBase::Arg<int>("num_axes", -1)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void BiasRunWithType();
    template <typename T> void ScaleRunWithType();
    template <typename T> void RunWithType();

 protected:
    TIndex axis, start_axis, num_axes;
    TIndex outer_dim, inner_dim, scale_dim, sum_dim, dim;
    Tensor sum_result;
};

#ifdef WITH_CUDNN

#include "utils/cudnn_device.h"

template <class Context>
class CuDNNAffineOpBase : public Operator<Context> {
 public:
    CuDNNAffineOpBase(const OperatorDef& def, Workspace* ws)
         : Operator<Context>(def, ws),
           axis(OperatorBase::Arg<int>("axis", 1)),
           num_axes(OperatorBase::Arg<int>("num_axes", -1)) {
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
    void ResetDesc(const Tensor& X) {
        //  determine the range of affine
        start_axis = axis;
        if (start_axis < 0) start_axis += (int)X.ndim();
        if (num_axes == -1) num_axes = (int)X.ndim() - start_axis;
        else if (num_axes == 0) num_axes = 1;
        end_axis = start_axis + num_axes;
        CHECK_LT(start_axis, (int)X.ndim());
        CHECK_LE(start_axis + num_axes, (int)X.ndim());
        //  determine the input desc
        vector<TIndex> input_dims = X.dims();
        //  cudnn requires ndimensions range from [4, 5]
        if (input_dims.size() < 4) input_dims.resize(4, 1);
        else if (input_dims.size() > 5) 
            LOG(FATAL) << "CuDNN Affine the dimensions up to 5.";
        cudnnSetTensorDesc<T>(&input_desc, input_dims);
        //  determine the scale desc
        vector<TIndex> param_dims(input_dims.size(), 1);
        for (int i = start_axis; i < end_axis; i++)
            param_dims[i] = input_dims[i];
        cudnnSetTensorDesc<T>(&param_desc, param_dims);
    }

    TIndex axis, start_axis, end_axis, num_axes;

    cudnnTensorDescriptor_t input_desc, param_desc;
    cudnnOpTensorDescriptor_t mul_desc, add_desc;
    cudnnReduceTensorDescriptor_t reduce_desc;
};

#define USE_CUDNN_AFFINE_FUCNTIONS \
    USE_OPERATOR_FUNCTIONS; \
    using CuDNNAffineOpBase<Context>::start_axis; \
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
    template <typename T> void RunWithType();

 protected:
     USE_CUDNN_AFFINE_FUCNTIONS;
};

template <class Context>
class CuDNNAffineGradientOp final : public CuDNNAffineOpBase<Context> {
public:
    CuDNNAffineGradientOp(const OperatorDef& def, Workspace* ws)
        : CuDNNAffineOpBase<Context>(def, ws) {}

    void RunOnDevice() override;
    template <typename T> void ComputeScaleGradient(T* dYxX, T* dA);
    template <typename T> void ComputeScaleGradient_v2(T* dYxX, T* dA);
    template <typename T> void ComputeBiasGradient(const T* dY, T* dB);
    template <typename T> void ComputeBiasGradient_v2(const T* dY, T* dB);
    template <typename T> void RunWithType();

protected:
    USE_CUDNN_AFFINE_FUCNTIONS;
    TIndex outer_dim, inner_dim, scale_dim, sum_dim, dim;
    Tensor sum_result;
};

#endif    // WITH_CUDNN

}    // namespace dragon

#endif    // DRAGON_OPERATORS_ARITHMETIC_AFFINE_OP_H_