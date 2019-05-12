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
          mean_vec_(OpArgs<float>("mean_values")),
          std_vec_(OpArgs<float>("std_values")) {
        if (mean_vec_.size() > 0) {
            CHECK_EQ((int)mean_vec_.size(), 3);
            auto* mean = mean_.Reshape({ 3 })
                ->mutable_data<float, CPUContext>();
            for (int i = 0; i < 3; ++i) mean[i] = mean_vec_[i];
        }
        if (std_vec_.size() > 0) {
            CHECK_EQ((int)std_vec_.size(), 3);
            auto* std = std_.Reshape({ 3 })
                ->mutable_data<float, CPUContext>();
            for (int i = 0; i < 3; ++i) std[i] = std_vec_[i];
        }
    }
    USE_OPERATOR_FUNCTIONS;

    void RunOnDevice() override;

    template <typename Tx, typename Ty>
    void RunImpl();

 protected:
    Tensor mean_, std_;
    int64_t n_, c_, h_, w_;
    vector<float> mean_vec_, std_vec_;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_MISC_IMAGE_DATA_OP_H_