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

#ifndef DRAGON_OPERATORS_MISC_IMAGE_DATA_OP_H_
#define DRAGON_OPERATORS_MISC_IMAGE_DATA_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class ImageDataOp final : public Operator<Context> {
 public:
    ImageDataOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          dtype(OperatorBase::Arg<string>("dtype", "float32")),
          mean_values(OperatorBase::Args<float>("mean_values")),
          std_values(OperatorBase::Args<float>("std_values")),
          data_format(OperatorBase::Arg<string>("data_format", "NCHW")) {
        if (mean_values.size() > 0) {
            CHECK_EQ((int)mean_values.size(), 3)
                << "The mean values should be a list with length 3.";
            mean.Reshape({ 3 });
            for (int i = 0; i < 3; i++)
                mean.mutable_data<float, CPUContext>()[i] = mean_values[i];
        }
        if (std_values.size() > 0) {
            CHECK_EQ((int)std_values.size(), 3)
                << "The std values should be a list with length 3.";
            std.Reshape({ 3 });
            for (int i = 0; i < 3; i++)
                std.mutable_data<float, CPUContext>()[i] = std_values[i];
        }
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;
    template <typename Tx, typename Ty> void RunWithType();

 protected:
    string dtype, data_format;
    vector<float> mean_values, std_values;
    int64_t n, c, h, w;
    Tensor mean, std;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_MISC_IMAGE_DATA_OP_H_