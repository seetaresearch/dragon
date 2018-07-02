// ------------------------------------------------------------
// Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// -------------------------------------------------------------

#ifndef DRAGON_OPERATORS_MISC_INITIALIZE_OP_H_
#define DRAGON_OPERATORS_MISC_INITIALIZE_OP_H_

#include "core/operator.h"
#include "utils/filler.h"

namespace dragon {

template <class Context>
class InitializeOp : public Operator<Context> {
 public:
    InitializeOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          shape_desc(OperatorBase::Arg<string>("shape", "")) {
        GET_ARGUMENTS_WITH_DESC(int, dims);
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    DECLARE_ARGUMENTS_WITH_DESC(int, dims);
    string shape_desc;
    TensorFiller filler;
};

template <class Context>
class FillOp final : public InitializeOp<Context> {
 public:
    FillOp(const OperatorDef& def, Workspace* ws)
        : InitializeOp<Context>(def, ws) {
        this->filler.set_type("constant");
        this->filler.set_value(OperatorBase::Arg<float>("value", 0.0));
    }
    USE_OPERATOR_FUNCTIONS;
};

template <class Context>
class RandomUniformOp final : public InitializeOp<Context> {
public:
    RandomUniformOp(const OperatorDef& def, Workspace* ws)
        : InitializeOp<Context>(def, ws) {
        this->filler.set_type("uniform");
        this->filler.set_low(OperatorBase::Arg<float>("low", -1.0));
        this->filler.set_high(OperatorBase::Arg<float>("high", 1.0));
    }
    USE_OPERATOR_FUNCTIONS;
};

template <class Context>
class RandomNormalOp final : public InitializeOp<Context> {
public:
    RandomNormalOp(const OperatorDef& def, Workspace* ws)
        : InitializeOp<Context>(def, ws) {
        this->filler.set_type("normal");
        this->filler.set_mean(OperatorBase::Arg<float>("mean", 0.0));
        this->filler.set_std(OperatorBase::Arg<float>("std", 1.0));
    }
    USE_OPERATOR_FUNCTIONS;
};

template <class Context>
class TruncatedNormalOp final : public InitializeOp<Context> {
public:
    TruncatedNormalOp(const OperatorDef& def, Workspace* ws)
        : InitializeOp<Context>(def, ws) {
        this->filler.set_type("truncated_normal");
        float mu = OperatorBase::Arg<float>("mean", 0.0);
        float sigma = OperatorBase::Arg<float>("std", 1.0);
        this->filler.set_mean(mu);
        this->filler.set_std(sigma);
        this->filler.set_low(mu - 2 * sigma);
        this->filler.set_high(mu + 2 * sigma);
    }
    USE_OPERATOR_FUNCTIONS;
};

template <class Context>
class GlorotUniformOp final : public InitializeOp<Context> {
public:
    GlorotUniformOp(const OperatorDef& def, Workspace* ws)
        : InitializeOp<Context>(def, ws) {
        string mode = OperatorBase::Arg<string>("mode", "fan_in");
        float scale = OperatorBase::Arg<float>("scale", 3.0);

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
    USE_OPERATOR_FUNCTIONS;
};

template <class Context>
class GlorotNormalOp final : public InitializeOp<Context> {
public:
    GlorotNormalOp(const OperatorDef& def, Workspace* ws)
        : InitializeOp<Context>(def, ws) {
        string mode = OperatorBase::Arg<string>("mode", "fan_in");
        float scale = OperatorBase::Arg<float>("scale", 2.0);

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
    USE_OPERATOR_FUNCTIONS;
};

DEFINE_ARGUMENTS_WITH_DESC(int, InitializeOp, dims);

}    // namespace

#endif    // DRAGON_OPERATORS_MISC_INITIALIZE_OP_H_