// ------------------------------------------------------------
// Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
//
// Licensed under the BSD 2-Clause License.
// You should have received a copy of the BSD 2-Clause License
// along with the software. If not, See,
//
//      <https://opensource.org/licenses/BSD-2-Clause>
//
// ------------------------------------------------------------

#ifndef DRAGON_CORE_TYPES_H_
#define DRAGON_CORE_TYPES_H_

namespace dragon {

#ifdef _MSC_VER
  
typedef struct __declspec(align(2)) {
    unsigned short x;
} float16;

typedef struct __declspec(align(4)) {
    unsigned int x;
} float32;

#else 

typedef struct {
    unsigned short x;
} __attribute__((aligned(2))) float16;

typedef struct {
    unsigned int x;
} __attribute__((aligned(4))) float32;

#endif

}    // namespace dragon

#endif  // DRAGON_CORE_TYPES_H_