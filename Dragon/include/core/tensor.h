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

#ifndef DRAGON_CORE_TENSOR_H_
#define DRAGON_CORE_TENSOR_H_

#include "core/common.h"
#include "core/mixedmem.h"

namespace dragon {

class Tensor {
 public:
     /*! \brief Default Constructor */
    Tensor() : name_("") {}

    /*! \brief Constructor with the known name */
    Tensor(const string& name) : name_(name) {}

    /*! \brief Constructor with the known int64 dimensions */
    Tensor(const vector<int64_t>& dims) { Reshape(dims); }

    /*! \brief Constructor with the known int32 dimensions */
    Tensor(const vector<int>& dims) {
        Reshape(vector<int64_t>(dims.begin(), dims.end()));
    }

    /*! \brief Reshape to the given dimensions */
    Tensor* Reshape(const vector<int64_t>& dims) {
        dims_ = dims; strides_.resize(dims.size());
        size_t new_size = 1; int64_t d;
        for (int i = (int)dims.size() - 1; i >= 0; i--) {
            d = dims[i]; strides_[i] = (int64_t)new_size;
            CHECK_GE(d, 0);
            if (d > 0) new_size *= d;
        }
        if (own_mem_) {
            if (capacity_ < new_size * meta_.itemsize()) {
                memory_.reset();
                capacity_ = 0;
            }
        } else {
            if (ex_memory_ && !is_shared_ &&
                capacity_ < new_size * meta_.itemsize()) {
                delete ex_memory_;
                ex_memory_ = nullptr;
                capacity_ = 0;
            }
        }
        size_ = new_size;
        return this;
    }

    /*! \brief Reshape the dimensions like the given tensor */
    Tensor* ReshapeLike(const Tensor& other) { return Reshape(other.dims_); }

    /*! \brief Return the tensor name */
    const string& name() const { return name_; }

    /*! \brief Return a canonical axis  */
    int64_t axis(const int64_t i) const {
        CHECK(i >= -ndim() && i < ndim())
            << "\nTensor(" << name() << ") "
            << "tried to get the dimension of axis "
            << (i < 0 ? i + ndim() : i) << ", "
            << "while the num of dimensions is " << ndim() << ".";
        return i < 0 ? i + ndim() : i;
    }

    /*! \brief Return the number of dimensions */
    int ndim() const { return (int)dims_.size(); }

    /*! \brief Return the dimension of given axis */
    int64_t dim(int64_t i) const{ return dims_[axis(i)]; }

    /*! \brief Return all the dimensions */
    const vector<int64_t>& dims() const { return dims_; }

    /*! \brief Return the total number of elements of this tensor */
    size_t size() const { return size_; }

    /*! \brief Return the total number bytes of this tensor */
    size_t nbytes() const { return size_ * meta_.itemsize(); }

    /*! \brief Return the capacity of the internal memory */
    size_t capacity() const { return capacity_; }

    /*! \brief Return the number of elements along the [start, end) axes */
    int64_t count(int64_t start, int64_t end) const {
        int64_t nelements = 1;
        for (int64_t i = start; i < end; i++) nelements *= dim(i);
        return nelements;
    }

    /*! \brief Return the total number of elements of this tensor */
    int64_t count() const { return (int64_t)size_; }

    /*! \brief Return the number of elements from the start axis */
    int64_t count(int64_t start) const { return count(start, ndim()); }

    /*! \brief Return the stride of given axis */
    int64_t stride(int64_t i) const { return strides_[axis(i)]; }

    /*! \brief Return all the strides */
    const vector<int64_t>& strides() const { return strides_; }

    /*! \brief Return a string to describe the given dimensions */
    static string DimString(const vector<int64_t>& dims) {
        if (dims.size() == 0) return "(0,)";
        std::stringstream ss;
        ss << "(";
        for (int i = 0; i < dims.size() - 1; i++)
            ss << dims[i] << ",";
        if (dims.size() == 1) ss << dims[0] << ",)";
        else ss << dims.back() << ")";
        return ss.str();
    }

    /*! \brief Return a string to describe the dimensions of this tensor */
    string DimString() const { return DimString(dims_); }

    /*! \brief Return the version of this tensor */
    int version() const { return version_; }

    /*! \brief Set the version of this tensor */
    void set_version(int version) { version_ = version; }

    /*! \brief Whether this tensor holds a valid memory */
    bool has_memory() const { return memory_ || ex_memory_ != nullptr; }

    /*! \brief Return the pointer of internal memory */
    MixedMemory* memory() const { return own_mem_ ? memory_.get() : ex_memory_; }

    /*! \brief Set the memory from a external pointer */
    void set_memory(MixedMemory* mem) {
        memory_.reset(mem); capacity_ = mem->nbytes();
    }

    /*! \brief Return the state of the internal memory */
    MixedMemory::State memory_state() const {
        MixedMemory* mem = memory();
        CHECK(mem) << "\nMemory access before allowcating.";
        return memory()->state();
    }

    /*! \brief Switch the memory to the specific device */
    void SwitchToDevice(int device_id) {
        MixedMemory* mem = memory();
        if (mem) mem->SwitchToDevice(device_id);
    }

    /*! \brief Return the type meta of this tensor */
    const TypeMeta& meta() const { return meta_; }

    /*! \brief Set the type meta */
    void SetMeta(const TypeMeta& meta) { meta_ = meta; }

    /*! \brief Whether the data type of this tensor is <T> */
    template <typename T>
    bool IsType() { return meta_.Match<T>(); }

    /*! \brief Try to get the raw mutable data pointer */
    template <class Context>
    void mutable_data_ptr(void** data_ptr) {
        MixedMemory* mem = memory();
        if (!mem) {
            *data_ptr = nullptr;
        } else {
            if (TypeMeta::Id<Context>() ==
                TypeMeta::Id<CPUContext>()) {
                *data_ptr = mem->mutable_cpu_data(nbytes());
            } else if (TypeMeta::Id<Context>() ==
                TypeMeta::Id<CUDAContext>()) {
                *data_ptr = mem->mutable_cuda_data(nbytes());
            } else if (TypeMeta::Id<Context>() ==
                TypeMeta::Id<CNMLContext>()) {
                *data_ptr = mem->mutable_cnml_data();
            } else {
                LOG(FATAL) << "Unknown memory type.\n"
                    << "Only CPU, CUDA and CNML are supported.";
            }
        }
    }

    /*! \brief Try to get the raw const data pointer */
    template <class Context>
    const void* const_data_ptr() const {
        MixedMemory* mem = memory();
        CHECK(mem) << "\nMemory access before allowcating.";
        if (TypeMeta::Id<Context>() ==
            TypeMeta::Id<CPUContext>()) {
            return mem->cpu_data(nbytes());
        } else if (TypeMeta::Id<Context>() ==
            TypeMeta::Id<CUDAContext>()) {
            return mem->cuda_data(nbytes());
        } else if (TypeMeta::Id<Context>() ==
            TypeMeta::Id<CNMLContext>()) {
            return mem->cnml_data();
        } else {
            LOG(FATAL) << "Unknown memory type.\n"
                << "Only CPU, CUDA, and CNML are supported.";
            return nullptr;
        }
    }

    /*! \brief Try to allocate the raw data memory */
    template <class Context>
    void* raw_mutable_data(const TypeMeta& meta) {
        void* data_ptr;
        mutable_data_ptr<Context>(&data_ptr);
        // Return the memory directly
        if (meta_ == meta && data_ptr) return data_ptr;
        // Return the new memory
        meta_ = meta;
        CHECK_GT(size_, 0);
        if (own_mem_) {
            memory_.reset(new MixedMemory(
                meta_, size_* meta_.itemsize()));
        } else {
            if (data_ptr) delete ex_memory_;
            ex_memory_ = new MixedMemory(
                meta_, size_* meta_.itemsize());
        }
        // Malloc
        mutable_data_ptr<Context>(&data_ptr);
        // Call the constructors
        if (meta_.ctor()) meta_.ctor()(data_ptr, size_);
        capacity_ = size_ * meta_.itemsize();
        return data_ptr;
    }

    /*! \brief Get the raw mutable data pointer */
    template <class Context>
    void* raw_mutable_data() {
        CHECK_NE(meta_.id(), 0)
            << "\nTensor(" << name_ << "): unknown type, "
            << "or does not have a type.";
        return raw_mutable_data<Context>(meta_);
    }

    /*! \brief Get the raw const data pointer */
    template <class Context>
    const void* raw_data() const {
        return const_data_ptr<Context>();
    }

    /*! \brief Get the typed mutable data pointer */
    template <typename T, class Context>
    T* mutable_data() {
        void* data_ptr;
        mutable_data_ptr<Context>(&data_ptr);
        if (data_ptr) {
            auto meta = TypeMeta::Make<T>();
            if (meta_ == meta) {
                return static_cast<T*>(data_ptr);
            } else if (capacity_ >=
                size_ * meta.itemsize()) {
                meta_ = meta;
                return static_cast<T*>(data_ptr);
            }
        }
        return static_cast<T*>(raw_mutable_data
            <Context>(TypeMeta::Make<T>()));
    }

    /*! \brief Get the typed const data pointer */
    template <typename T, class Context>
    const T* data() const {
        CHECK(meta_ == TypeMeta::Make<T>())
            << "\nThe DType of Tensor(" << name() << ") is "
            << TypeMetaToString(meta_) << ", while required "
            << TypeMetaToString(TypeMeta::Make<T>()) << ".";
        return static_cast<const T*>(raw_data<Context>());
    }

    /*! \brief Copy the contents from the given tensor */
    template <class Context>
    void CopyFrom(const Tensor& other, Context* ctx) {
        if ((void*)&other == (void*)this) return;
        CHECK_EQ(size_, other.size_);
        auto* src = other.template raw_data<Context>();
        auto* dst = raw_mutable_data<Context>(other.meta_);
        ctx->template MemcpyAsync<Context, Context>(
            nbytes(), dst, src);
    }

    /*! \brief Move the external memory */
    void Move(MixedMemory* mem) {
        if (mem != nullptr) {
            ex_memory_ = mem;
        } else {
            ex_memory_ = new MixedMemory(
                TypeMeta::Make<float>(), 4);
        }
        own_mem_ = false;
        capacity_ = ex_memory_->nbytes();
    }

    /*! \brief Share the external memory */
    void Share(MixedMemory* mem) {
        Move(mem); is_shared_ = true;
    }

    /*! \brief Reset the memory */
    void Reset() {
        size_ = capacity_ = 0; meta_ = TypeMeta();
        dims_.clear(); strides_.clear(); memory_.reset();
        if (DECREFPyArray) DECREFPyArray();
    }

    /*! \brief Reset the owned PyArray memory */
    std::function<void()> DECREFPyArray;

    /*! \brief Deconstructor */
    ~Tensor() { /*! DO NOT CALL DECREFARRAY */ }

 private:
    /*! \brief The name of this tensor */
    string name_;

    /*! \brief The type meta of this tensor */
    TypeMeta meta_;

    /*! \brief Store the size and capacity */
    size_t size_ = 0, capacity_ = 0;

    /*! \brief Store the version for shared tensor */
    int version_ = -1;

    /*! \brief Store the dimensions and strides */
    vector<int64_t> dims_, strides_;

    /*! \brief The internal memory */
    shared_ptr<MixedMemory> memory_;

    /*! \brief Store the external memory pointer */
    MixedMemory* ex_memory_ = nullptr;

    /*! \brief External memory indicators */
    bool is_shared_ = false, own_mem_ = true;
};

}  // namespace dragon

#endif  // DRAGON_CORE_TENSOR_H_