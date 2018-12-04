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

typedef int64_t TIndex;
typedef size_t TSize;

class Tensor {
 public:
    Tensor() {}
    Tensor(const vector<TIndex>& dims) { Reshape(dims); }
    Tensor(const string& name) : name_(name) {}

    void Reshape(const vector<TIndex>& dims) {
        dims_ = dims;
        TIndex new_size = 1;
        for (auto d : dims_) {
            CHECK_GE(d, 0);
            if (d > 0) new_size *= d;
        }
        if (own_mem_) {
            if (size_ != new_size &&
                capacity_ < TIndex(new_size * meta_.itemsize())) {
                memory_.reset();
                capacity_ = 0;
            }
        } else {
            if (ex_memory_ && !is_shared_ &&
                capacity_ < TIndex(new_size * meta_.itemsize())) {
                delete ex_memory_;
                ex_memory_ = nullptr;
                capacity_ = 0;
            }
        }
        size_ = new_size;
    }

    void ReshapeLike(const Tensor& other) { Reshape(other.dims_); }

    inline const string& name() const { return name_; }

    inline TIndex axis(const TIndex i) const {
        CHECK_GE(i, -(TIndex)ndim());
        CHECK_LT(i, (TIndex)ndim());
        if (i < 0) return i + ndim();
        else return i;
    }

    inline TSize ndim() const { return dims_.size(); }
    inline TIndex dim(const TIndex i) const{ return dims_[axis(i)];}
    inline const vector<TIndex>& dims() const { return dims_; }

    inline TSize nbytes() const { return size_ * meta_.itemsize(); }
    inline TSize capacity() const { return capacity_; }

    inline TIndex count(const TIndex start, const TIndex end) const {
        TIndex ret = 1;
        for (TIndex i = start; i < end; i++) ret *= dim(i);
        return ret;
    }

    inline TIndex count() const { return size_; }

    inline TIndex count(const TIndex start) const {
        return count(start, ndim());
    }

    inline TIndex offset(
        const TIndex            n,
        const TIndex            c = 0,
        const TIndex            h = 0,
        const TIndex            w = 0) {
        CHECK_LE(n, dim(0));
        CHECK_LE(c, dim(1));
        CHECK_LE(h, dim(2));
        CHECK_LE(w, dim(3));
        return ((n * dim(1) + c) * dim(2) + h) * dim(3) + w;
    }

    inline TIndex offset(const vector<TIndex>& vec) {
        CHECK_LE(vec.size(), ndim());
        TIndex offset = 0;
        for (int i = 0; i < ndim(); i++) {
            offset = offset * dim(i);
            if (vec.size() > i) offset += vec[i];
        }
        return offset;
    }

    static inline string DimString(
        const vector<TIndex>&   dims) {
        if (dims.size() == 0) return "(0,)";
        std::stringstream ss;
        ss << "(";
        for (int i = 0; i < dims.size() - 1; i++)
            ss << dims[i] << ",";
        if (dims.size() == 1) ss << dims[0] << ",)";
        else ss << dims.back() << ")";
        return ss.str();
    }

    inline string DimString() const { return DimString(dims_); }

    inline bool is_corrupted() const { return is_corrupted_; }
    inline void Corrupt() { is_corrupted_ = true; }

    inline bool has_memory() const {
        return memory_ || ex_memory_ != nullptr;
    }

    MixedMemory* memory() const {
        return own_mem_ ? memory_.get() : ex_memory_;
    }

    void set_memory(MixedMemory* mem) {
        memory_.reset(mem); capacity_ = mem->nbytes();
    }

    MixedMemory::State memory_state() const {
        MixedMemory* mem = memory();
        CHECK(mem) << "\nMemory access before allowcating.";
        return memory()->state();
    }

    void SwitchToDevice() {
        MixedMemory* mem = own_mem_ ? memory_.get() : ex_memory_;
        if (mem) mem->SwitchToDevice();
    }

    const TypeMeta& meta() const { return meta_; }
    void SetMeta(const TypeMeta& meta) { meta_ = meta; }
    template <typename T>
    inline bool IsType() { return meta_.Match<T>(); }

    template <class Context>
    void mutable_data_ptr(void** data_ptr) {
        MixedMemory* mem = memory();
        if (!mem) {
            *data_ptr = nullptr;
        } else {
            if (TypeMeta::Id<Context>() ==
                    TypeMeta::Id<CPUContext>()) {
                *data_ptr = mem->mutable_cpu_data();
            } else if (TypeMeta::Id<Context>() ==
                    TypeMeta::Id<CUDAContext>()) {
                *data_ptr = mem->mutable_cuda_data();
            } else if (TypeMeta::Id<Context>() == 
                    TypeMeta::Id<CNMLContext>()) {
                *data_ptr = mem->mutable_cnml_data();
            } else {
                LOG(FATAL) << "Unknown memory type.\n"
                           << "Only CPU, CUDA and CNML are supported.";
            }
        }
    }

    template <class Context>
    const void* const_data_ptr() const {
        MixedMemory* mem = memory();
        CHECK(mem) << "\nMemory access before allowcating.";
        if (TypeMeta::Id<Context>() ==
                TypeMeta::Id<CPUContext>()) {
             return mem->cpu_data();
        } else if (TypeMeta::Id<Context>() ==
                TypeMeta::Id<CUDAContext>()) {
             return mem->cuda_data();
        } else if (TypeMeta::Id<Context>() == 
                TypeMeta::Id<CNMLContext>()) {
            return mem->cnml_data();
        } else {
             LOG(FATAL) << "Unknown memory type.\n"
                        << "Only CPU, CUDA, and CNML are supported.";
             return nullptr;
        }
    }

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
        capacity_ = size_ * meta_.itemsize(), require_init_ = true;
        return data_ptr;
    }

    template <class Context>
    void* raw_mutable_data() {
        CHECK_NE(meta_.id(), 0)
            << "\nTensor(" << name_ << "): unknown type, "
            << "or does not have a type.";
        return raw_mutable_data<Context>(meta_);
    }

    template <class Context>
    const void* raw_data() const {
        return const_data_ptr<Context>();
    }

    template <typename T, class Context>
    T* mutable_data() {
        void* data_ptr;
        mutable_data_ptr<Context>(&data_ptr);
        if (data_ptr && meta_ == TypeMeta::Make<T>())
            return static_cast<T*>(data_ptr);
        return static_cast<T*>(
            raw_mutable_data<Context>(TypeMeta::Make<T>()));
    }

    template <typename T, class Context>
    T* mutable_data(Context* ctx) {
        auto* data = mutable_data<T, Context>();
        if (!require_init_) return data;
        ctx->MemsetAsync(nbytes(), (void*)data);
        require_init_ = false;
        return data;
    }

    template <typename T, class Context>
    const T* data() const {
        CHECK(meta_ == TypeMeta::Make<T>())
            << "\nThe DType of Tensor(" << name() << ") is "
            << TypeMetaToString(meta_) << ", while required "
            << TypeMetaToString(TypeMeta::Make<T>());
        return static_cast<const T*>(raw_data<Context>());
    }

    template <class Context>
    inline void CopyFrom(const Tensor& other, Context* ctx) {
        if ((void*)&other == (void*)this) return;
        CHECK_EQ(size_, other.size_);
        auto* src = other.template raw_data<Context>();
        auto* dst = raw_mutable_data<Context>(other.meta_);
        ctx->template MemcpyAsync<Context, Context>(
            nbytes(), dst, src);
        require_init_ = false;
    }

    inline void Move(MixedMemory* mem) {
        if (mem != nullptr) {
            ex_memory_ = mem;
            require_init_ = false;
        } else {
            ex_memory_ = new MixedMemory(
                TypeMeta::Make<float>(), 4);
            require_init_ = true;
        } own_mem_ = false;
        capacity_ = (TIndex)ex_memory_->nbytes();
    }

    inline void Share(MixedMemory* mem) {
        Move(mem); is_shared_ = true;
        require_init_ = false;
    }

    inline void Reset() {
        size_ = capacity_ = 0;
        meta_ = TypeMeta();
        dims_.clear();
        memory_.reset();
        if (DECREFPyArray) DECREFPyArray();
    }

    std::function<void()> DECREFPyArray;
    ~Tensor() { /*! DO NOT CALL DECREFARRAY */ }

 private:
    vector<TIndex> dims_;
    TIndex size_ = 0, capacity_ = 0;
    TypeMeta meta_;
    string name_;
    shared_ptr<MixedMemory> memory_;
    MixedMemory* ex_memory_ = nullptr;
    bool is_corrupted_ = false, is_shared_ = false;
    bool own_mem_ = true, require_init_ = true;
};

}  // namespace dragon

#endif  // DRAGON_CORE_TENSOR_H_