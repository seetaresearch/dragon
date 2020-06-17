/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *    <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_CORE_CONTEXT_H_
#define DRAGON_CORE_CONTEXT_H_

#include "dragon/core/common.h"

namespace dragon {

class DRAGON_API CPUContext {
 public:
  /*! \brief Default Constructor */
  explicit CPUContext() : random_seed_(3) {}

  /*! \brief Constructor with the specified random seed */
  explicit CPUContext(unsigned int random_seed) : random_seed_(random_seed) {}

  /*! \brief Constructor with the specified device option */
  explicit CPUContext(const DeviceOption& option)
      : random_seed_(
            option.has_random_seed() ? option.random_seed()
                                     : DEFAULT_RNG_SEED) {}

  /*! \brief Destructor */
  virtual ~CPUContext() {}

  /*! \brief Alloc the memory */
  static void* New(size_t nbytes) {
    void* data = malloc(nbytes);
    CHECK(data) << "\nAllocate memory with " << nbytes << " bytes failed.";
    return data;
  }

  /*! \brief Zero-Reset the memory */
  static void Memset(size_t nbytes, void* ptr) {
    memset(ptr, 0, nbytes);
  }

  /*! \brief Copy the memory */
  template <class DestContext, class SrcContext>
  static void Memcpy(size_t nbytes, void* dest, const void* src) {
    memcpy(dest, src, nbytes);
  }

  /*! \brief Free the memory */
  static void Delete(void* data) {
    free(data);
  }

  /*! \brief Zero-Reset the memory asynchronously */
  void MemsetAsync(size_t nbytes, void* ptr) {
    memset(ptr, 0, nbytes);
  }

  /*! \brief Copy the memory asynchronously */
  template <class DestContext, class SrcContext>
  void MemcpyAsync(size_t nbytes, void* dest, const void* src) {
    memcpy(dest, src, nbytes);
  }

  /*! \brief Switch to the device of this context */
  void SwitchToDevice() {}

  /*! \brief Switch to the device with the given stream */
  void SwitchToDevice(const int stream_id) {}

  /*! \brief Copy the memory with given type asynchronously */
  template <typename T, class DestContext, class SrcContext>
  void Copy(int n, T* dest, const T* src) {
    if (dest == src) return;
    if (std::is_fundamental<T>::value) {
      Memcpy<DestContext, SrcContext>(
          n * sizeof(T), (void*)dest, (const void*)src);
    } else {
      for (int i = 0; i < n; i++) {
        dest[i] = src[i];
      }
    }
  }

  /*! \brief Synchronize the dispatched operations */
  void FinishDeviceComputation() {}

  /*! \brief Return the device index */
  int device_id() const {
    return 0;
  }

  /*! \brief Return the stream index */
  int stream_id() const {
    return 0;
  }

  /*! \brief Return the internal random generator */
  std::mt19937* rand_generator() {
    if (!rand_generator_.get()) {
      rand_generator_.reset(new std::mt19937(random_seed_));
    }
    return rand_generator_.get();
  }

  /*! \brief Set the stream index */
  void set_stream_id(int stream_id) {}

 private:
  /*! \brief Store the random seed */
  unsigned int random_seed_;

  /*! \brief Store the internal random generator */
  unique_ptr<std::mt19937> rand_generator_;
};

#define CPU_FP16_NOT_SUPPORTED \
  LOG(FATAL) << "FP16 is unsupported for CPUContext.";

} // namespace dragon

#endif // DRAGON_CORE_CONTEXT_H_
