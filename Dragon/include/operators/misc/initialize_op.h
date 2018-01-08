// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_MISC_INITIALIZE_OP_H_
#define DRAGON_OPERATORS_MISC_INITIALIZE_OP_H_

#include "core/operator.h"
#include "utils/filler.h"

namespace dragon {

template <class Context>
class InitializeOp: public Operator<Context> {
 public:
    InitializeOp(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          dims_desc(OperatorBase::GetRepeatedArg<string>("dims")), 
          shape_desc(OperatorBase::GetSingleArg<string>("shape", "")) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    vector<string> dims_desc;
    string shape_desc;
    TensorFiller filler;
};

template <class Context>
class FillOp final : public InitializeOp<Context> {
public:
    FillOp(const OperatorDef& op_def, Workspace* ws) 
        : InitializeOp<Context>(op_def, ws) {
        this->filler.set_type("constant");
        this->filler.set_value(OperatorBase::GetSingleArg<float>("value", 0.0));
    }
};

template <class Context>
class RandomUniformOp final : public InitializeOp<Context> {
public:
    RandomUniformOp(const OperatorDef& op_def, Workspace* ws) 
        : InitializeOp<Context>(op_def, ws) {
        this->filler.set_type("uniform");
        this->filler.set_low(OperatorBase::GetSingleArg<float>("low", -1.0));
        this->filler.set_high(OperatorBase::GetSingleArg<float>("high", 1.0));
    }
};

template <class Context>
class RandomNormalOp final : public InitializeOp<Context> {
public:
    RandomNormalOp(const OperatorDef& op_def, Workspace* ws) 
        : InitializeOp<Context>(op_def, ws) {
        this->filler.set_type("normal");
        this->filler.set_mean(OperatorBase::GetSingleArg<float>("mean", 0.0));
        this->filler.set_std(OperatorBase::GetSingleArg<float>("std", 1.0));
    }
};

template <class Context>
class TruncatedNormalOp final : public InitializeOp<Context> {
public:
    TruncatedNormalOp(const OperatorDef& op_def, Workspace* ws) 
        : InitializeOp<Context>(op_def, ws) {
        this->filler.set_type("truncated_normal");
        float mu = OperatorBase::GetSingleArg<float>("mean", 0.0);
        float sigma = OperatorBase::GetSingleArg<float>("std", 1.0);
        this->filler.set_mean(mu);
        this->filler.set_std(sigma);
        this->filler.set_low(mu - 2 * sigma);
        this->filler.set_high(mu + 2 * sigma);
    }
};

template <class Context>
class GlorotUniformOp final : public InitializeOp<Context> {
public:
    GlorotUniformOp(const OperatorDef& op_def, Workspace* ws)
        : InitializeOp<Context>(op_def, ws) {
        string mode = OperatorBase::GetSingleArg<string>("mode", "fan_in");
        float scale = OperatorBase::GetSingleArg<float>("scale", 3.0);

        this->filler.set_type("xavier");
        if (mode == "fan_avg") {
            this->filler.set_variance_norm(TensorFiller_VarianceNorm_FAN_AVG);
        } else if (mode == "fan_out") {
            this->filler.set_variance_norm(TensorFiller_VarianceNorm_FAN_OUT);
        } else {
            this->filler.set_variance_norm(TensorFiller_VarianceNorm_FAN_IN);
        }
        this->filler.set_scale(scale);
    }
};

template <class Context>
class GlorotNormalOp final : public InitializeOp<Context> {
public:
    GlorotNormalOp(const OperatorDef& op_def, Workspace* ws)
        : InitializeOp<Context>(op_def, ws) {
        string mode = OperatorBase::GetSingleArg<string>("mode", "fan_in");
        float scale = OperatorBase::GetSingleArg<float>("scale", 2.0);

        this->filler.set_type("msra");
        if (mode == "fan_avg") {
            this->filler.set_variance_norm(TensorFiller_VarianceNorm_FAN_AVG);
        } else if (mode == "fan_out") {
            this->filler.set_variance_norm(TensorFiller_VarianceNorm_FAN_OUT);
        } else {
            this->filler.set_variance_norm(TensorFiller_VarianceNorm_FAN_IN);
        }
        this->filler.set_scale(scale);
    }
};

}    // namespace

#endif    // DRAGON_OPERATORS_MISC_INITIALIZE_OP_H_
