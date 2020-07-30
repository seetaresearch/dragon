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

// Disable the copy and assignment operator for a class
#define DISABLE_COPY_AND_ASSIGN(classname) \
  classname(const classname&) = delete;    \
  classname& operator=(const classname&) = delete

// Concatenate two strings
#define CONCATENATE_IMPL(s1, s2) s1##s2
#define CONCATENATE(s1, s2) CONCATENATE_IMPL(s1, s2)

// Return a anonymous variable name using line number
#define ANONYMOUS_VARIABLE(str) CONCATENATE(str, __LINE__)

// Throw a fatal logging for not implemented function
#define NOT_IMPLEMENTED LOG(FATAL) << "This function is not implemented."

#endif // DRAGON_UTILS_MARCROS_H_
