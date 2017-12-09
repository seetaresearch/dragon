// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_NORM_BATCH_NORM_OP_H_
#define DRAGON_OPERATORS_NORM_BATCH_NORM_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class BatchNormOp : public Operator<Context> {
 public:
    BatchNormOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", -1)),
          momentum(OperatorBase::GetSingleArg<float>("momentum", float(0.9))),
          eps(OperatorBase::GetSingleArg<float>("eps", float(1e-3))),
          use_stats(OperatorBase::GetSingleArg<int>("use_stats", -1)),
          mode(OperatorBase::GetSingleArg<string>("mode", "DEFAULT")) {
        if (axis != -1) 
            CHECK_EQ(axis, 1) 
                << "\nThe axis can only be set to 1.";
    }

    void Setup();

    void RunOnDevice() override;
    template <typename T> void TrainingRunWithType();
    template <typename T> void InferenceRunWithType();

 protected:
    float momentum, eps;
    Tensor mean, num_by_chans;
    Tensor* multiplier, *num_multiplier, *spatial_multiplier;
    Tensor* stddev, *var;
    TIndex axis, N, C, S, NC, NS;
    string data_format, mode;
    int use_stats;
    bool use_global_stats, is_recomputing;
};

template <class Context>
class BatchNormGradientOp final : public Operator<Context> {
 public:
    BatchNormGradientOp(const OperatorDef& op_def, Workspace *ws)
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", -1)),
          use_stats(OperatorBase::GetSingleArg<int>("use_stats", -1)) {
        if (axis != -1)
            CHECK_EQ(axis, 1)
                << "\nThe axis can only be set to 1.";
    }

    void Setup();

    void RunOnDevice() override;
    template <typename T> void TrainingRunWithType();
    template <typename T> void InferenceRunWithType();

 protected:
    Tensor num_by_chans;
    Tensor* multiplier, *num_multiplier, *spatial_multiplier;
    Tensor* stddev, *var;
    TIndex axis, N, C, S, NC, NS;
    string data_format;
    int use_stats;
    bool use_global_stats;
};

template <class Context>
class FusedBatchNormOp : public Operator<Context> {
 public:
    FusedBatchNormOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", -1)),
          momentum(OperatorBase::GetSingleArg<float>("momentum", float(0.9))),
          eps(OperatorBase::GetSingleArg<float>("eps", float(1e-3))),
          use_stats(OperatorBase::GetSingleArg<int>("use_stats", -1)) {}

    void Setup() { NOT_IMPLEMENTED; }

    void RunOnDevice() override { NOT_IMPLEMENTED; }
    template <typename T> void RunWithType() { NOT_IMPLEMENTED; }

 protected:
    float momentum, eps;
    TIndex axis, N, C, S, NC, NS;
    string data_format;
    int use_stats;
    bool use_global_stats, is_recomputing;
};

template <class Context>
class FusedBatchNormGradientOp : public Operator<Context> {
 public:
    FusedBatchNormGradientOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          axis(OperatorBase::GetSingleArg<int>("axis", -1)),
          eps(OperatorBase::GetSingleArg<float>("eps", float(1e-3))),
          use_stats(OperatorBase::GetSingleArg<int>("use_stats", -1)) { }

    void Setup() { NOT_IMPLEMENTED; }

    void ShareGradient() override;

    void RunOnDevice() override { NOT_IMPLEMENTED; }
    template <typename T> void RunWithType() { NOT_IMPLEMENTED; }

 protected:
    float eps;
    TIndex axis, N, C, S, NC, NS;
    string data_format;
    int use_stats;
    bool use_global_stats;
};

#ifdef WITH_CUDNN

#if CUDNN_VERSION_MIN(5, 0, 0)

#include "utils/cudnn_device.h"

template <class Context>
class CuDNNBatchNormOp final : public FusedBatchNormOp<Context> {
 public:
    CuDNNBatchNormOp(const OperatorDef& op_def, Workspace* ws)
        : FusedBatchNormOp<Context>(op_def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_desc));
        this->eps = std::max(this->eps, float(CUDNN_BN_MIN_EPSILON));
    }

    void Setup();

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    cudnnTensorDescriptor_t input_desc, output_desc, bn_desc;
    cudnnBatchNormMode_t bn_mode;
    TIndex N, C;
    string data_format;
    Tensor* mean, *var;
};

template <class Context>
class CuDNNBatchNormGradientOp final : public FusedBatchNormGradientOp<Context> {
 public:
    CuDNNBatchNormGradientOp(const OperatorDef& op_def, Workspace* ws)
        : FusedBatchNormGradientOp<Context>(op_def, ws) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&input_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&output_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bn_desc));
        this->eps = std::max(this->eps, float(CUDNN_BN_MIN_EPSILON));
    }

    void Setup();

    void RunOnDevice() override;
    template <typename T> void TrainingRunWithType();
    template <typename T> void InferenceRunWithType();

 protected:
    cudnnTensorDescriptor_t input_desc, output_desc, bn_desc;
    cudnnBatchNormMode_t bn_mode;
    TIndex N, C, S, NC, NS;
    string data_format;
    Tensor num_by_chans;
    Tensor* multiplier, *num_multiplier, *spatial_multiplier;
    Tensor* mean, *var, *stddev;
};

#endif

#endif  // WITH_CUDNN

}    // namespace dragon 

#endif    // DRAGON_OPERATORS_NORM_BATCH_NORM_OP_H_