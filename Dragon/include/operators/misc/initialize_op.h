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
          shape_desc(OperatorBase::Arg<string>("shape", "")),
          dtype(OperatorBase::Arg<string>("dtype", "float32")) {
        GET_ARGUMENTS_WITH_DESC(int64_t, dims);
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    string shape_desc, dtype;
    TensorFillerProto filler_proto;
    DECLARE_ARGUMENTS_WITH_DESC(int64_t, dims);
};

template <class Context>
class FillOp final : public Operator<Context> {
 public:
    FillOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          shape_desc(OperatorBase::Arg<string>("shape", "")),
          dtype(OperatorBase::Arg<string>("dtype", "float32")),
          value(OperatorBase::Arg<float>("value", 0.f)) {
        GET_ARGUMENTS_WITH_DESC(int64_t, dims);
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    float value;
    string shape_desc, dtype;
    DECLARE_ARGUMENTS_WITH_DESC(int64_t, dims);
};

template <class Context>
class GivenTensorFillOp final : public Operator<Context> {
 public:
    GivenTensorFillOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          shape(OperatorBase::Args<int64_t>("shape")),
          dtype(OperatorBase::Arg<string>("dtype", "float32")) {
        GET_ARGUMENTS_WITH_DESC(int64_t, dims);
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunWithType();

    template<typename T>
    struct TypeIdentity { typedef T type; };

    template <typename T>
    void ExtractValues() { ExtractValuesImpl(TypeIdentity<T>()); }

    template <typename T> void ExtractValuesImpl(TypeIdentity<T>) {
        auto source_values = OperatorBase::Args<T>("values");
        auto num_values = (int64_t)source_values.size();
        values.Reshape(vector<int64_t>({ num_values }));
        auto* Vdata = values.template mutable_data<T, CPUContext>();
        memcpy(Vdata, source_values.data(), num_values * sizeof(T));
    }

    void ExtractValuesImpl(TypeIdentity<float16>) {
        auto source_values = OperatorBase::Args<float>("values");
        auto num_values = (int64_t)source_values.size();
        values.Reshape(vector<int64_t>({ num_values }));
        auto* Vdata = values.template mutable_data<float16, CPUContext>();
        memcpy(Vdata, source_values.data(), num_values * sizeof(float16));
    }

 protected:
    string dtype;
    vector<int64_t> shape;
    Tensor values;
    DECLARE_ARGUMENTS_WITH_DESC(int64_t, dims);
};

template <class Context>
class RandomUniformOp final : public InitializeOp<Context> {
public:
    RandomUniformOp(const OperatorDef& def, Workspace* ws)
        : InitializeOp<Context>(def, ws) {
        this->filler_proto.set_type("uniform");
        this->filler_proto.set_low(OperatorBase::Arg<float>("low", -1.f));
        this->filler_proto.set_high(OperatorBase::Arg<float>("high", 1.f));
    }
    USE_OPERATOR_FUNCTIONS;
};

template <class Context>
class RandomNormalOp final : public InitializeOp<Context> {
public:
    RandomNormalOp(const OperatorDef& def, Workspace* ws)
        : InitializeOp<Context>(def, ws) {
        this->filler_proto.set_type("normal");
        this->filler_proto.set_mean(OperatorBase::Arg<float>("mean", 0.f));
        this->filler_proto.set_std(OperatorBase::Arg<float>("std", 1.f));
    }
    USE_OPERATOR_FUNCTIONS;
};

template <class Context>
class TruncatedNormalOp final : public InitializeOp<Context> {
 public:
    TruncatedNormalOp(const OperatorDef& def, Workspace* ws)
        : InitializeOp<Context>(def, ws) {
        this->filler_proto.set_type("truncated_normal");
        float mu = OperatorBase::Arg<float>("mean", 0.f);
        float sigma = OperatorBase::Arg<float>("std", 1.f);
        this->filler_proto.set_mean(mu);
        this->filler_proto.set_std(sigma);
        this->filler_proto.set_low(mu - 2 * sigma);
        this->filler_proto.set_high(mu + 2 * sigma);
    }
    USE_OPERATOR_FUNCTIONS;
};

template <class Context>
class GlorotUniformOp final : public InitializeOp<Context> {
public:
    GlorotUniformOp(const OperatorDef& def, Workspace* ws)
        : InitializeOp<Context>(def, ws) {
        string mode = OperatorBase::Arg<string>("mode", "fan_in");
        float scale = OperatorBase::Arg<float>("scale", 3.f);
        this->filler_proto.set_type("xavier");
        if (mode == "fan_avg") {
            this->filler_proto.set_variance_norm(
                TensorFillerProto_VarianceNorm_FAN_AVG);
        } else if (mode == "fan_out") {
            this->filler_proto.set_variance_norm(
                TensorFillerProto_VarianceNorm_FAN_OUT);
        } else {
            this->filler_proto.set_variance_norm(
                TensorFillerProto_VarianceNorm_FAN_IN);
        }
        this->filler_proto.set_scale(scale);
    }
    USE_OPERATOR_FUNCTIONS;
};

template <class Context>
class GlorotNormalOp final : public InitializeOp<Context> {
public:
    GlorotNormalOp(const OperatorDef& def, Workspace* ws)
        : InitializeOp<Context>(def, ws) {
        string mode = OperatorBase::Arg<string>("mode", "fan_in");
        float scale = OperatorBase::Arg<float>("scale", 2.f);
        this->filler_proto.set_type("msra");
        if (mode == "fan_avg") {
            this->filler_proto.set_variance_norm(
                TensorFillerProto_VarianceNorm_FAN_AVG);
        } else if (mode == "fan_out") {
            this->filler_proto.set_variance_norm(
                TensorFillerProto_VarianceNorm_FAN_OUT);
        } else {
            this->filler_proto.set_variance_norm(
                TensorFillerProto_VarianceNorm_FAN_IN);
        }
        this->filler_proto.set_scale(scale);
    }
    USE_OPERATOR_FUNCTIONS;
};

DEFINE_ARGUMENTS_WITH_DESC(int64_t, InitializeOp, dims);
DEFINE_ARGUMENTS_WITH_DESC(int64_t, FillOp, dims);
DEFINE_ARGUMENTS_WITH_DESC(int64_t, GivenTensorFillOp, dims);

}  // namespace dragon

#endif  // DRAGON_OPERATORS_MISC_INITIALIZE_OP_H_