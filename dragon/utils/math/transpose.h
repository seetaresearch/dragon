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

#ifndef DRAGON_UTILS_MATH_TRANSPOSE_H_
#define DRAGON_UTILS_MATH_TRANSPOSE_H_

#include "dragon/core/context.h"

namespace dragon {

namespace math {

template <typename T, class Context>
DRAGON_API void Transpose(
    const int num_dims,
    const int64_t* dims,
    const int64_t* axes,
    const T* x,
    T* y,
    Context* ctx);

} // namespace math

} // namespace dragon

#endif // DRAGON_UTILS_MATH_TRANSPOSE_H_
