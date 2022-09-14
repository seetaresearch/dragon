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

#ifndef DRAGON_CORE_TENSOR_H_
#define DRAGON_CORE_TENSOR_H_

#include "dragon/core/common.h"
#include "dragon/core/context.h"
#include "dragon/core/memory.h"

namespace dragon {

/*!
 * \brief The base tensor class, manage memory or not.
 *
 * Tensor is usually constructed with the shape info:
 *
 * \code{.cpp}
 * auto* a = new dragon::Tensor(std::vector<int64_t>({2, 3}));
 * auto* b = dragon::Tensor().Reshape({2, 3}); // Equivalent
 * \endcode
 *
 * To allocate the data, type meta and device context are also required:
 *
 * \code{.cpp}
 * auto meta = dragon::TypeMeta::Make<float>();
 * auto* raw_data = a->raw_mutable_data<dragon::CPUContext>(meta);
 * auto* data = b->mutable_data<float, dragon::CPUContext>();
 * \endcode
 *
 * Memory will be reset if required bytes is larger than capacity:
 * \code{.cpp}
 * std::cout << a->nbytes() << " " << a->capacity() << std::endl; // 24, 24
 * std::cout << a->Reshape({2, 4})->size() << std::endl; // 8
 * std::cout << a->nbytes() << " " << a->capacity() << std::endl; // 32, 0
 * a->mutable_data<float, dragon::CPUContext>();
 * a->Reshape({2, 3});
 * std::cout << a->nbytes() << " " << a->capacity() << std::endl; // 24, 32
 * \endcode
 */
class DRAGON_API Tensor {
 public:
  /*! \brief Default constructor */
  Tensor() : name_("") {}

  /*! \brief Constructor with the name */
  explicit Tensor(const string& name) : name_(name) {}

  /*! \brief Constructor with the int64 dimensions */
  explicit Tensor(const vec64_t& dims) {
    Reshape(dims);
  }

  /*! \brief Constructor with the int32 dimensions */
  explicit Tensor(const vec32_t& dims) {
    Reshape(vec64_t(dims.begin(), dims.end()));
  }

  /*! \brief Constructor with the type meta */
  explicit Tensor(const TypeMeta& meta) {
    set_meta(meta);
  }

  /*! \brief Destructor of the external tensor */
  std::function<void()> ExternalDeleter = nullptr;

  /*! \brief Destructor */
  virtual ~Tensor() {
    if (ExternalDeleter != nullptr) {
      ExternalDeleter();
    }
  }

  /*! \brief Change the dimensions */
  Tensor* Reshape(const vec64_t& dims) {
    dims_ = dims;
    strides_.resize(dims.size());
    size_t new_size = 1;
    int64_t d;
    for (int i = (int)dims.size() - 1; i >= 0; i--) {
      d = dims[i];
      strides_[i] = (int64_t)new_size;
      CHECK_GE(d, 0);
      if (d > 0) new_size *= d;
    }
    size_ = new_size;
    return this;
  }

  /*! \brief Change the dimensions as the other */
  Tensor* ReshapeLike(const Tensor& other) {
    return Reshape(other.dims_);
  }

  /*! \brief Copy memory from a tensor */
  template <class Context>
  Tensor* CopyFrom(Tensor& other, Context* ctx) {
    if ((void*)&other == (void*)this) return this;
    CHECK_EQ(size_, other.size_);
    meta_ = other.meta_;
    auto* src = other.template raw_data<Context>();
    auto* dst = raw_mutable_data<Context>();
    if (dst == src) return this;
    ctx->template MemcpyAsync<Context, Context>(nbytes(), dst, src);
    return this;
  }

  /*! \brief Copy memory from a vector */
  template <typename TensorType, typename VectorType>
  Tensor* CopyFrom(const vector<VectorType>& other) {
    if (other.size() > 0) {
      Reshape({(int64_t)other.size()});
      auto* data = this->template mutable_data<TensorType, CPUContext>();
      for (int i = 0; i < count(); i++) {
        data[i] = static_cast<VectorType>(other[i]);
      }
    }
    return this;
  }

  /*! \brief Copy memory to a vector */
  template <typename TensorType, typename VectorType>
  void CopyTo(vector<VectorType>& dest) {
    dest.resize(size());
    auto* data = this->template data<TensorType, CPUContext>();
    for (int i = 0; i < count(); i++) {
      dest[i] = static_cast<VectorType>(data[i]);
    }
  }

  /*! \brief Map memory from a tensor */
  Tensor* MapFrom(Tensor* other, size_t offset = 0) {
    if (other == nullptr) {
      if (mapped_memory_ != nullptr) {
        mapped_memory_ = nullptr;
        capacity_ = (memory_ != nullptr ? memory_->size() : 0);
        offset_ = 0;
      }
    } else {
      auto* new_memory = other->memory();
      if (new_memory != nullptr) {
        CHECK_LE(size_ * meta_.itemsize(), new_memory->size())
            << "\nMap from a memory with smaller capacity.";
        mapped_memory_ = new_memory;
        capacity_ = new_memory->size();
        offset_ = offset;
      }
    }
    version_ = -1;
    return this;
  }

  /*! \brief Reset tensor to release all resources */
  void Reset() {
    dims_.clear();
    strides_.clear();
    meta_ = TypeMeta();
    memory_.reset();
    mapped_memory_ = nullptr;
    size_ = capacity_ = offset_ = 0;
    if (ExternalDeleter != nullptr) {
      ExternalDeleter();
      ExternalDeleter = nullptr;
    }
  }

  /*! \brief Return whether the data type is matched */
  template <typename T>
  bool IsType() {
    return meta_.Match<T>();
  }

  /*! \brief Return a string formatting the given dimensions */
  static string DimString(const vector<int64_t>& dims) {
    if (dims.size() == 0) return "(0,)";
    std::stringstream ss;
    ss << "(";
    for (size_t i = 0; i < dims.size() - 1; i++)
      ss << dims[i] << ",";
    if (dims.size() == 1) {
      ss << dims[0] << ",)";
    } else {
      ss << dims.back() << ")";
    }
    return ss.str();
  }

  /*! \brief Return a string formatting the tensor dimensions */
  string DimString() const {
    return DimString(dims_);
  }

  /*! \brief Return the tensor name */
  const string& name() const {
    return name_;
  }

  /*! \brief Return whether the tensor name is set */
  bool has_name() const {
    return !name_.empty();
  }

  /*! \brief Return the tensor version */
  int64_t version() const {
    return version_;
  }

  /*! \brief Return the number of elements */
  size_t size() const {
    return size_;
  }

  /*! \brief Return the byte length of memory */
  size_t capacity() const {
    return capacity_;
  }

  /*! \brief Return the byte length of all elements */
  size_t nbytes() const {
    return size_ * meta_.itemsize();
  }

  /*! \brief Return the type meta */
  const TypeMeta& meta() const {
    return meta_;
  }

  /*! \brief Return a canonical axis  */
  int64_t axis(const int64_t i) const {
    CHECK(i >= -ndim() && i < ndim())
        << "\nTensor(" << name() << ") "
        << "required the axis of " << i << ", "
        << "while the num of dimensions is " << ndim() << ".";
    return i < 0 ? i + ndim() : i;
  }

  /*! \brief Return the number of dimensions */
  int ndim() const {
    return (int)dims_.size();
  }

  /*! \brief Return the dimension of given axis */
  int64_t dim(int64_t i) const {
    return dims_[axis(i)];
  }

  /*! \brief Return the stride of given axis */
  int64_t stride(int64_t i) const {
    return strides_[axis(i)];
  }

  /*! \brief Return the dimensions */
  const vec64_t& dims() const {
    return dims_;
  }

  /*! \brief Return the strides */
  const vec64_t& strides() const {
    return strides_;
  }

  /*! \brief Return the number of elements counting along all axes */
  int64_t count() const {
    return (int64_t)size_;
  }

  /*! \brief Return the number of elements counting along given axes */
  int64_t count(int64_t start, int64_t end) const {
    int64_t num = 1;
    for (int64_t i = start; i < end; i++) {
      num *= dim(i);
    }
    return num;
  }

  /*! \brief Return the number of elements counting from given axis */
  int64_t count(int64_t start) const {
    return count(start, ndim());
  }

  /*! \brief Return whether the number of elements is zero */
  bool empty() const {
    return size_ == 0;
  }

  /*! \brief Return whether the memory is set */
  bool has_memory() const {
    return memory_ != nullptr || mapped_memory_ != nullptr;
  }

  /*! \brief Return the memory */
  UnifiedMemory* memory(bool required = false, bool owned = false) {
    if (capacity_ < size_ * meta_.itemsize()) {
      mapped_memory_ = nullptr;
      memory_.reset();
      capacity_ = 0;
    }
    auto* ptr = (owned || !mapped_memory_ ? memory_.get() : mapped_memory_);
    if (required) CHECK(ptr) << "\nAccess the empty memory.";
    return ptr;
  }

  /*! \brief Return the memory state */
  UnifiedMemory::State memory_state() {
    return memory(true)->state();
  }

  /*! \brief Return the raw data pointer */
  template <class Context>
  const void* raw_data();

  /*! \brief Get the raw mutable data pointer */
  template <class Context>
  void raw_mutable_data(void** data_ptr);

  /*! \brief Return the raw mutable data pointer */
  template <class Context>
  void* raw_mutable_data() {
    CHECK_NE(meta_.id(), 0) << "\nTensor(" << name_ << "): unknown type, "
                            << "or does not have a type.";
    if (mapped_memory_ != nullptr) {
      if (version_ == 0) {
        MapFrom(nullptr);
      } else {
        version_ = 0;
      }
    }
    void* data_ptr;
    raw_mutable_data<Context>(&data_ptr);
    if (data_ptr) return data_ptr;
    CHECK_GT(size_, 0) << "\nInvalid tensor size.";
    capacity_ = ((size_ * meta_.itemsize() + 511) / 512) * 512;
    memory_.reset(new UnifiedMemory(meta_, capacity_));
    raw_mutable_data<Context>(&data_ptr);
    if (meta_.ctor()) meta_.ctor()(data_ptr, size_);
    return data_ptr;
  }

  /*! \brief Return the typed data pointer */
  template <typename T, class Context>
  const T* data() {
    CHECK(meta_.Match<T>()) << "\nThe type of Tensor(" << name() << ") is "
                            << dtypes::to_string(meta_) << ", while requesting "
                            << dtypes::to_string(TypeMeta::Make<T>()) << ".";
    return static_cast<const T*>(raw_data<Context>());
  }

  /*! \brief Return the typed mutable data pointer */
  template <typename T, class Context>
  T* mutable_data() {
    meta_ = TypeMeta::Make<T>();
    return static_cast<T*>(raw_mutable_data<Context>());
  }

  /*! \brief Set the tensor version */
  void set_version(int64_t version) {
    version_ = version;
  }

  /*! \brief Set the type meta */
  Tensor* set_meta(const TypeMeta& meta) {
    meta_ = meta;
    return this;
  }

  /*! \brief Set the managed memory */
  void set_memory(UnifiedMemory* memory) {
    if (memory != nullptr) {
      if (memory != memory_.get()) {
        memory_.reset(memory);
      }
      mapped_memory_ = nullptr;
      capacity_ = memory->size();
      offset_ = 0;
    }
  }

 private:
  /*! \brief The tensor name */
  string name_;

  /*! \brief The type meta */
  TypeMeta meta_;

  /*! \brief The number of elements */
  size_t size_ = 0;

  /*! \brief The byte length of memory */
  size_t capacity_ = 0;

  /*! \brief The byte offset of memory */
  size_t offset_ = 0;

  /*! \brief The tensor version */
  int64_t version_ = -1;

  /*! \brief The dimensions */
  vec64_t dims_;

  /*! \brief The strides */
  vec64_t strides_;

  /*! \brief The managed memory */
  unique_ptr<UnifiedMemory> memory_;

  /*! \brief The mapped memory */
  UnifiedMemory* mapped_memory_ = nullptr;

  DISABLE_COPY_AND_ASSIGN(Tensor);
};

} // namespace dragon

#endif // DRAGON_CORE_TENSOR_H_
