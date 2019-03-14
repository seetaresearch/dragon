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

#ifndef DRAGON_OPERATORS_ARRAY_MULTINOMIAL_OP_H_
#define DRAGON_OPERATORS_ARRAY_MULTINOMIAL_OP_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class MultinomialOp final : public Operator<Context> {
 public:
    MultinomialOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          normalize(OperatorBase::Arg<int64_t>("normalize", 0)),
          num_samples(OperatorBase::Arg<int64_t>("num_samples", 1)) {}
    USE_OPERATOR_FUNCTIONS;

    void SoftmaxRun();

    void RunOnDevice() override;
    template <typename T> void RunWithType();

 protected:
    Tensor* prob;
    int64_t normalize, num_samples, outer_dim, axis;
    unique_ptr<OperatorBase> softmax_op;
};

}  // namespace dragon

#endif  // DRAGON_OPERATORS_ARRAY_MULTINOMIAL_OP_H_