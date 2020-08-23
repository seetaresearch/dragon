#ifndef DRAGON_UTILS_DEVICE_COMMON_NCCL_H_
#define DRAGON_UTILS_DEVICE_COMMON_NCCL_H_

#ifdef USE_NCCL

#include <nccl.h>

#define NCCL_VERSION_MIN(major, minor, patch) \
  (NCCL_VERSION_CODE >= NCCL_VERSION(major, minor, patch))

#define NCCL_CHECK(condition)                                            \
  do {                                                                   \
    ncclResult_t status = condition;                                     \
    CHECK_EQ(status, ncclSuccess) << "\n" << ncclGetErrorString(status); \
  } while (0)

#endif // USE_NCCL

#endif // DRAGON_UTILS_DEVICE_COMMON_NCCL_H_
