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

/*                  Base                  */

template <class Context>
class DimOpBase : public Operator<Context> {
 public:
    SIMPLE_CTOR_DTOR(DimOpBase);

    void MemorySwitch() override {
        /* Disable the Memory Activation */
    }
};

template <class Context>
class DimGradientOpBase : public Operator<Context> {
 public:
    SIMPLE_CTOR_DTOR(DimGradientOpBase);
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override {
        // Simply copy the dY to dX
        Y(0)->ReshapeLike(X(0));
        Y(0)->CopyFrom(X(-1), ctx());
    }
};

#define DEFINE_DIMENSION_GRADIENT_OP(name) \
    template <class Context> \
    class name##GradientOp final : \
        public DimGradientOpBase<Context> { \
     public: \
      name##GradientOp( \
        const OperatorDef&      def, \
        Workspace*              ws) \
        : DimGradientOpBase<Context>(def, ws) {} \
    };

/*                  Reshape                  */

template <class Context>
class ReshapeOp final : public DimOpBase<Context> {
 public:
    ReshapeOp(const OperatorDef& def, Workspace* ws)
        : DimOpBase<Context>(def, ws),
          shape_desc_(OpArg<string>("shape_like", "")) {
        GET_ARGS_WITH_DESC(int64_t, dims);
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;

 protected:
    string shape_desc_;
    vec64_t req_shape_, new_shape_;
    DECLARE_ARGS_WITH_DESC(int64_t, dims);
};

DEFINE_DIMENSION_GRADIENT_OP(Reshape);
DEFINE_ARGS_WITH_DESC(int64_t, ReshapeOp, dims);

/*                  Flatten                  */

template <class Context>
class FlattenOp final : public DimOpBase<Context> {
 public:
    FlattenOp(const OperatorDef& def, Workspace* ws)
        : DimOpBase<Context>(def, ws),
          axis_(OpArg<int64_t>("axis", 0)),
          num_axes_(OpArg<int64_t>("num_axes", -1)),
          keep_axes_(OpArg<int64_t>("keep_axes", INT_MAX)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;

 protected:
    int64_t axis_, num_axes_, keep_axes_;
};

DEFINE_DIMENSION_GRADIENT_OP(Flatten);

/*                  ExpandDims                  */

template <class Context>
class ExpandDimsOp final : public DimOpBase<Context> {
 public:
    ExpandDimsOp(const OperatorDef& def, Workspace* ws)
        : DimOpBase<Context>(def, ws),
          axis_(OpArg<int64_t>("axis", 0)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;

 protected:
    int64_t axis_;
};

DEFINE_DIMENSION_GRADIENT_OP(ExpandDims);

/*                  Squeeze                  */

template <class Context>
class SqueezeOp final : public DimOpBase<Context> {
public:
    SqueezeOp(const OperatorDef& def, Workspace* ws)
        : DimOpBase<Context>(def, ws),
        axis_(OpArg<int64_t>("axis", INT_MAX)) {}
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;

 protected:
    int64_t axis_;
};

DEFINE_DIMENSION_GRADIENT_OP(Squeeze);

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARRAY_RESHAPE_OP_H_