// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

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