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

#ifndef DRAGON_CORE_CONTEXT_H_
#define DRAGON_CORE_CONTEXT_H_

#include "dragon/core/common.h"

namespace dragon {

class Workspace;

/*!
 * \brief The cpu device context.
 */
class DRAGON_API CPUContext {
 public:
  /*! \brief Constructor */
  CPUContext() : random_seed_(3) {}

  /*! \brief Constructor with the random seed */
  explicit CPUContext(unsigned int random_seed) : random_seed_(random_seed) {}

  /*! \brief Constructor with the device option */
  explicit CPUContext(const DeviceOption& option)
      : random_seed_(
            option.has_random_seed() ? option.random_seed()
                                     : DEFAULT_RNG_SEED) {}

  /*! \brief Destructor */
  virtual ~CPUContext() {}

  /*! \brief Allocate a block of memory */
  static void* New(size_t size) {
    void* data = malloc(size);
    CHECK(data) << "\nAllocate memory with " << size << " bytes failed.";
    return data;
  }

  /*! \brief Set a memory block to the given value */
  static void Memset(size_t n, void* ptr, int value = 0) {
    memset(ptr, value, n);
  }

  /*! \brief Set a memory block to the given value asynchronously */
  void MemsetAsync(size_t n, void* ptr, int value) {
    memset(ptr, value, n);
  }

  /*! \brief Copy a memory block to the destination */
  template <class DestContext, class SrcContext>
  static void Memcpy(size_t n, void* dest, const void* src) {
    memcpy(dest, src, n);
  }

  /*! \brief Copy a memory block to the destination asynchronously */
  template <class DestContext, class SrcContext>
  void MemcpyAsync(size_t n, void* dest, const void* src) {
    memcpy(dest, src, n);
  }

  /*! \brief Deallocate a memory block */
  static void Delete(void* ptr) {
    free(ptr);
  }

  /*! \brief Switch to the device and select given stream in current thread */
  void SwitchToDevice(int stream_id = 0) {}

  /*! \brief Copy a typed memory block to the destination */
  template <typename T, class DestContext, class SrcContext>
  static void Copy(int n, T* dest, const T* src) {
    if (dest == src) return;
    if (std::is_fundamental<T>::value) {
      Memcpy<DestContext, SrcContext>(
          n * sizeof(T), (void*)dest, (const void*)src);
    } else {
      for (int i = 0; i < n; ++i) {
        dest[i] = src[i];
      }
    }
  }

  /*! \brief Wait for the dispatched computation to complete */
  void FinishDeviceComputation() {}

  /*! \brief Return the current workspace */
  Workspace* workspace();

  /*! \brief Return the device index */
  int device() const {
    return 0;
  }

  /*! \brief Return the stream index */
  int stream() const {
    return 0;
  }

  /*! \brief Return the random generator */
  std::mt19937* rand_generator() {
    if (!rand_generator_.get()) {
      rand_generator_.reset(new std::mt19937(random_seed_));
    }
    return rand_generator_.get();
  }

  /*! \brief Set the stream index */
  void set_stream(int stream) {}

 private:
  /*! \brief The random seed */
  unsigned int random_seed_;

  /*! \brief The random generator */
  unique_ptr<std::mt19937> rand_generator_;
};

#define CPU_FP16_NOT_SUPPORTED \
  LOG(FATAL) << "FP16 is unsupported for CPUContext.";

} // namespace dragon

#endif // DRAGON_CORE_CONTEXT_H_
