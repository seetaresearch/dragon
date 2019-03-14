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

#ifndef DRAGON_OPERATORS_ARRAY_DIMENSION_OP_H_
#define DRAGON_OPERATORS_ARRAY_DIMENSION_OP_H_

#include "core/operator.h"

namespace dragon {

/*********************************************
*                                            *
*                   Base                     *
*                                            *
**********************************************/

template <class Context>
class DimOpBase : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(DimOpBase);

    void MemorySwitch() override {
        /* Disable the Memory Activation */
    }
};

template <class Context>
class DimGradientOpBase : public Operator<Context> {
 public:
    USE_SIMPLE_CTOR_DTOR(DimGradientOpBase);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override {
        // Simply copy the dY to dX
        Output(0)->ReshapeLike(Input(0));
        Output(0)->template CopyFrom<Context>(Input(-1), ctx());
    }
};

#define DEFINE_DIMENSION_GRADIENT_OP(name) \
    template <class Context> \
    class name##GradientOp final : public DimGradientOpBase<Context> { \
     public: \
      name##GradientOp(const OperatorDef& def, Workspace* ws) \
        : DimGradientOpBase<Context>(def, ws) {} \
    };

/*********************************************
*                                            *
*                   Reshape                  *
*                                            *
**********************************************/

template <class Context>
class ReshapeOp final : public DimOpBase<Context> {
 public:
    ReshapeOp(const OperatorDef& def, Workspace* ws)
        : DimOpBase<Context>(def, ws),
          shape_like_desc(OperatorBase::Arg<string>("shape_like", "")) {
        GET_ARGUMENTS_WITH_DESC(int64_t, dims);
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;

 protected:
    string shape_like_desc;
    vector<int64_t> require_shape, new_shape;
    DECLARE_ARGUMENTS_WITH_DESC(int64_t, dims);
};

DEFINE_DIMENSION_GRADIENT_OP(Reshape);
DEFINE_ARGUMENTS_WITH_DESC(int64_t, ReshapeOp, dims);

/*********************************************
*                                            *
*                   Flatten                  *
*                                            *
**********************************************/

template <class Context>
class FlattenOp final : public DimOpBase<Context> {
 public:
    FlattenOp(const OperatorDef& def, Workspace* ws)
        : DimOpBase<Context>(def, ws),
          axis(OperatorBase::Arg<int64_t>("axis", 0)),
          num_axes(OperatorBase::Arg<int64_t>("num_axes", -1)),
          keep_axes(OperatorBase::Arg<int64_t>("keep_axes", INT_MAX)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;

 protected:
    int64_t axis, num_axes, keep_axes;
};

DEFINE_DIMENSION_GRADIENT_OP(Flatten);

/*********************************************
*                                            *
*                Expand Dims                 *
*                                            *
**********************************************/

template <class Context>
class ExpandDimsOp final : public DimOpBase<Context> {
 public:
    ExpandDimsOp(const OperatorDef& def, Workspace* ws)
        : DimOpBase<Context>(def, ws),
          axis(OperatorBase::Arg<int64_t>("axis", 0)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;

 protected:
    int64_t axis;
};

DEFINE_DIMENSION_GRADIENT_OP(ExpandDims);

/*********************************************
*                                            *
*                  Squeeze                   *
*                                            *
**********************************************/

template <class Context>
class SqueezeOp final : public DimOpBase<Context> {
public:
    SqueezeOp(const OperatorDef& def, Workspace* ws)
        : DimOpBase<Context>(def, ws),
        axis(OperatorBase::Arg<int64_t>("axis", INT_MAX)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;

 protected:
    int64_t axis;
};

DEFINE_DIMENSION_GRADIENT_OP(Squeeze);

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARRAY_RESHAPE_OP_H_