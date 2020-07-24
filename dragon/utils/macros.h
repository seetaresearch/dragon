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

#ifndef DRAGON_UTILS_MARCROS_H_
#define DRAGON_UTILS_MARCROS_H_

#ifndef DRAGON_API
#define DRAGON_API
#endif

// Avoid using of "thread_local" for VS2013 or older Xcode
#if defined(__clang__) || defined(__GNUC__)
#define TLS_OBJECT __thread
#else
#define TLS_OBJECT __declspec(thread)
#endif

#define CONCATENATE_IMPL(s1, s2) s1##s2
#define CONCATENATE(s1, s2) CONCATENATE_IMPL(s1, s2)
#define ANONYMOUS_VARIABLE(str) CONCATENATE(str, __LINE__)
#define NOT_IMPLEMENTED \
  LOG(FATAL) << "This module has not been implemented yet."

#endif // DRAGON_UTILS_MARCROS_H_
