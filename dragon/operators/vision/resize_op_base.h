/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_VISION_RESIZE_OP_BASE_H_
#define DRAGON_OPERATORS_VISION_RESIZE_OP_BASE_H_

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class ResizeOpBase : public Operator<Context> {
 public:
  ResizeOpBase(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        mode_(str::upper(OP_SINGLE_ARG(string, "mode", "NEAREST"))),
        align_corners_(OP_SINGLE_ARG(int64_t, "align_corners", 0)) {
    INITIALIZE_OP_REPEATED_ARG(float, scales);
    INITIALIZE_OP_REPEATED_ARG(int64_t, sizes);
  }
  USE_OPERATOR_FUNCTIONS;

 protected:
  void ComputeOutShape() {
    auto& X = Input(0);
    CHECK(X.ndim() >= 3) << "\nExcept 3 or more dimensions.";

    int axis = 1;
    int num_axes = X.ndim() - 2;
    int num_sizes;
    sizes(0, &num_sizes);
    int num_scales;
    scales(0, &num_scales);

    in_dims_ = X.dims();
    if (data_format() == "NCHW") {
      axis = 2;
    } else if (data_format() == "NHWC") {
      in_dims_.insert(in_dims_.begin() + 1, in_dims_.back());
      in_dims_.pop_back(); // Store dimensions in NCHW order.
    } else {
      LOG(FATAL) << "Unknown data format: " << data_format();
    }

    out_shape_ = X.dims();
    out_dims_.resize((size_t)num_axes);

    if (num_sizes > 0) {
      if (num_sizes == 1) {
        for (int i = 0; i < num_axes; ++i)
          out_dims_[i] = out_shape_[axis + i] = sizes(0);
      } else if (num_sizes == num_axes) {
        for (int i = 0; i < num_axes; ++i)
          out_dims_[i] = out_shape_[axis + i] = sizes(i);
      } else {
        CHECK_EQ(num_sizes, X.ndim())
            << "\nExcepted 1/" << num_axes << "/" << X.ndim() << " values "
            << "for <sizes>, got " << num_sizes << ".";
        for (int i = 0; i < num_axes; ++i)
          out_dims_[i] = out_shape_[axis + i] = sizes(axis + i);
      }
    } else if (num_scales > 0) {
      if (num_scales == 1) {
        for (int i = 0; i < num_axes; ++i) {
          out_shape_[axis + i] *= scales(0);
          out_dims_[i] = out_shape_[axis + i];
        }
      } else if (num_scales == num_axes) {
        for (int i = 0; i < num_axes; ++i) {
          out_shape_[axis + i] *= scales(i);
          out_dims_[i] = out_shape_[axis + i];
        }
      } else {
        CHECK_EQ(num_scales, X.ndim())
            << "\nExcepted 1/" << num_axes << "/" << X.ndim() << " values "
            << "for <scales>, got " << num_scales << ".";
        for (int i = 0; i < num_axes; ++i) {
          out_shape_[axis + i] *= scales(axis + i);
          out_dims_[i] = out_shape_[axis + i];
        }
      }
    } else {
      LOG(FATAL) << "Specify either <sizes> or <scales>.";
    }
  }

  string mode_;
  int64_t align_corners_;
  vec64_t in_dims_, out_dims_, out_shape_;
  DECLARE_OP_REPEATED_ARG(float, scales);
  DECLARE_OP_REPEATED_ARG(int64_t, sizes);
};

#define USE_RESIZE_FUNCTIONS                    \
  using ResizeOpBase<Context>::ComputeOutShape; \
  using ResizeOpBase<Context>::scales;          \
  using ResizeOpBase<Context>::sizes;           \
  using ResizeOpBase<Context>::in_dims_;        \
  using ResizeOpBase<Context>::out_dims_;       \
  using ResizeOpBase<Context>::out_shape_;      \
  using ResizeOpBase<Context>::align_corners_;  \
  using ResizeOpBase<Context>::mode_

DEFINE_OP_REPEATED_ARG(float, ResizeOpBase, scales);
DEFINE_OP_REPEATED_ARG(int64_t, ResizeOpBase, sizes);

} // namespace dragon

#endif // RAGON_OPERATORS_VISION_RESIZE_OP_BASE_H_
