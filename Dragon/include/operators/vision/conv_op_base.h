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

#ifndef DRAGON_OPERATORS_VISION_CONV_OP_BASE_H_
#define DRAGON_OPERATORS_VISION_CONV_OP_BASE_H_

#include "core/operator.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context>
class ConvOpBase : public Operator<Context> {
 public:
    ConvOpBase(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          data_format(OperatorBase::Arg<string>("data_format", "NCHW")),
          padding(OperatorBase::Arg<string>("padding", "VALID")),
          num_output(OperatorBase::Arg<int64_t>("num_output", 0)),
          group(OperatorBase::Arg<int64_t>("group", 1)) {
        if (data_format == "NCHW") spatial_axis = 2;
        else if (data_format == "NHWC") spatial_axis = 1;
        else LOG(FATAL) << "Unknown data format: " << data_format;
        num_spatial_axes = -1;  // Unknown
        output_shape_spec_value = OperatorBase::Args<int64_t>("output_shape");
        output_shape_spec_desc = OperatorBase::Args<string>("output_shape_desc");
        GET_ARGUMENTS_WITH_DESC(int64_t, output_padding);
    }
    USE_OPERATOR_FUNCTIONS;

 public:
    vector<int64_t> kernel_shape, stride, pad_l, pad_r, dilation;
    string data_format, padding;
    vector<int64_t> input_shape, output_shape, bottom_shape, top_shape;
    vector<int64_t> weight_shape, bias_shape;
    int64_t num_output, group;
    int64_t spatial_axis, num_spatial_axes;
    int64_t channels, out_spatial_dim;
    int64_t conv_in_channels, conv_out_channels;
    int64_t conv_out_spatial_dim, kernel_dim, col_dim;
    int64_t col_offset, output_offset, weight_offset, x_offset, y_offset;
    bool is_1x1;
    DECLARE_ARGUMENTS_WITH_DESC(int64_t, output_padding);  // Adjs
    DECLARE_ARGUMENTS_WITH_DESC(int64_t, output_shape_spec);

    void Setup();
    void Reshape();
    void GradientReshape();
    virtual void ComputeOutputShape();
    virtual bool ReverseDimensions() = 0;
    virtual bool HasBias() { NOT_IMPLEMENTED; return true; }

    template <typename T> void Wx(const T* x,
        const T* weights, T* y, bool skip_im2col = false);

    template <typename T> void Pb(const T* bias, T* y);

    template <typename T> void Dx(const T* dy, const T* weights, T* dx);

    template <typename T> void Dw(const T* dy, const T* x, T* dw);

    template <typename T> void Db(const T* dy, T* db);

 private:
    template <typename T> void Im2Col(const T* im, T* col) {
        if (Input(0).ndim() == 4) {
             kernel::Im2Col2d<T, Context>(conv_in_channels,
                            input_shape[0], input_shape[1],
                          output_shape[0], output_shape[1],
                          kernel_shape[0], kernel_shape[1],
                                      stride[0], stride[1],
                                        pad_l[0], pad_l[1],
                                  dilation[0], dilation[1],
                                               data_format,
                                                        im,
                                                       col,
                                                   ctx());
        } else LOG(FATAL) << "ConvNd has not been implemented yet";
    }

    template <typename T> void Col2Im(const T* col, T* im) {
        if (Input(0).ndim() == 4) {
             kernel::Col2Im2d<T, Context>(conv_in_channels,
                            input_shape[0], input_shape[1],
                          output_shape[0], output_shape[1],
                          kernel_shape[0], kernel_shape[1],
                                      stride[0], stride[1],
                                        pad_l[0], pad_l[1],
                                  dilation[0], dilation[1],
                                               data_format,
                                                       col,
                                                        im,
                                                   ctx());
        } else LOG(FATAL) << "ConvNd has not been implemented yet";
    }
};

DEFINE_ARGUMENTS_WITH_DESC(int64_t, ConvOpBase, output_padding);
DEFINE_ARGUMENTS_WITH_DESC(int64_t, ConvOpBase, output_shape_spec);

#define USE_CONVOLUTION_FUNCTIONS \
    using ConvOpBase<Context>::Setup; \
    using ConvOpBase<Context>::Reshape; \
    using ConvOpBase<Context>::GradientReshape; \
    using ConvOpBase<Context>::ComputeOutputShape; \
    using ConvOpBase<Context>::ReverseDimensions; \
    using ConvOpBase<Context>::HasBias; \
    using ConvOpBase<Context>::Wx; \
    using ConvOpBase<Context>::Pb; \
    using ConvOpBase<Context>::Dx; \
    using ConvOpBase<Context>::Dw; \
    using ConvOpBase<Context>::Db; \
    using ConvOpBase<Context>::kernel_shape; \
    using ConvOpBase<Context>::stride; \
    using ConvOpBase<Context>::pad_l; \
    using ConvOpBase<Context>::pad_r; \
    using ConvOpBase<Context>::dilation; \
    using ConvOpBase<Context>::group; \
    using ConvOpBase<Context>::channels; \
    using ConvOpBase<Context>::num_output; \
    using ConvOpBase<Context>::data_format; \
    using ConvOpBase<Context>::x_offset; \
    using ConvOpBase<Context>::y_offset; \
    using ConvOpBase<Context>::weight_offset; \
    using ConvOpBase<Context>::weight_shape; \
    using ConvOpBase<Context>::bias_shape; \
    using ConvOpBase<Context>::input_shape; \
    using ConvOpBase<Context>::output_shape

}  // namespace dragon

#endif  // DRAGON_OPERATORS_VISION_CONV_OP_BASE_H_