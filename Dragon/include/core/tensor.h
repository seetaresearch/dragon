// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_CORE_TENSOR_H_
#define DRAONG_CORE_TENSOR_H_

#include <vector>

#include "core/common.h"
#include "core/typeid.h"
#include "core/mixedmem.h"

namespace dragon {

typedef int64_t TIndex;
typedef size_t TSize;

class Tensor {
 public:
    Tensor() {} 
    Tensor(const string& name) : name_(name) {} 

    void Reshape(const vector<TIndex>& dims) {
        dims_ = dims;
        TIndex new_size = 1;
        for (auto d : dims_) {
            CHECK_GT(d, 0);
            new_size *= d;
        }
        if (size_ != new_size && 
            capacity_ < TIndex(new_size * meta_.itemsize())) {
            memory_.reset();
            capacity_ = 0;
        }
        size_ = new_size;
    }

    void ReshapeLike(const Tensor& other) { 
        Reshape(other.dims_); 
    }

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

    inline TIndex count(const TIndex start, const TIndex end) const {
        TIndex ret = 1;
        for (TIndex i = start; i < end; i++) ret *= dim(i);
        return ret;
    }
    inline TIndex count() const { return size_; }
    inline TIndex count(const TIndex start) const { return count(start, ndim()); }

    inline TIndex offset(const TIndex n, const TIndex c = 0, 
                         const TIndex h = 0, const TIndex w = 0) {
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

    inline string dim_string() const {
        std::stringstream ss;
        ss << "(";
        for (int i = 0; i < ndim() - 1; i++) ss << dim(i) << ",";
        ss << dim(ndim() - 1) << ")";
        return ss.str();
    }

    MixedMemory::State memory_state() const { return memory_->state(); }
    MixedMemory* memory() const { return memory_.get(); }
    void SwitchToDevice() { if(memory_) memory_->SwitchToDevice(); }

    const TypeMeta& meta() const { return meta_; }
    void SetMeta(const TypeMeta& meta) { meta_ = meta; }
    template <typename T> inline bool IsType() { return meta_.Match<T>(); }

    template <class Context>
    const void* raw_data() const {
        CHECK(memory_.get()) << "memory access before allowcating.";
        if (TypeMeta::Id<Context>() == TypeMeta::Id<CPUContext>())
            return memory_->cpu_data();
        else if (TypeMeta::Id<Context>() == TypeMeta::Id<CUDAContext>())
            return memory_->cuda_data();
        else LOG(FATAL) << "unknown memory type access. only CPU or CUDA are supported.";
        return nullptr;
    }

    template <typename T, class Context>
    const T* data() const {
        return static_cast<const T*>(raw_data<Context>());
    }

    template <class Context>
    void active_data_ptr(void** data_ptr) {
        if (!memory_) {
            *data_ptr = nullptr;
        } else {
            if (TypeMeta::Id<Context>() == TypeMeta::Id<CPUContext>()) {
                *data_ptr = memory_->mutable_cpu_data();
            } else if (TypeMeta::Id<Context>() == TypeMeta::Id<CUDAContext>()) {
                *data_ptr = memory_->mutable_cuda_data();
            }
        }
    }

    template <class Context>
    void* raw_mutable_data(const TypeMeta& meta) {
        void* data_ptr;
        active_data_ptr<Context>(&data_ptr);
        if (meta_ == meta && data_ptr) {
            return data_ptr;
        } else {
            meta_ = meta;    //  copy-assign the meta
            CHECK_GT(size_, 0);    //  must specify a valid size
            memory_.reset(new MixedMemory(meta, size_* meta_.itemsize()));
            //  malloc
            if (TypeMeta::Id<Context>() == TypeMeta::Id<CPUContext>())
                data_ptr = memory_->mutable_cpu_data();
            else if (TypeMeta::Id<Context>() == TypeMeta::Id<CUDAContext>())
                data_ptr = memory_->mutable_cuda_data();
            //  init for each structed element if necessary
            if (meta.ctor()) meta_.ctor()(data_ptr, size_);
        }
        capacity_ = size_ * meta_.itemsize();
        return data_ptr;
    }

    template <class Context>
    void* raw_mutable_data() {
        CHECK_NE(meta_.id(), 0) 
            << "\nTensor(" << name_ << "): unknown type, "
            << "or does not have a type.";
        return raw_mutable_data<Context>(meta_);
    }

    template <typename T, class Context>
    T* mutable_data() {
        void* data_ptr;
        active_data_ptr<Context>(&data_ptr);
        if (data_ptr && meta_ == TypeMeta::Make<T>()) return static_cast<T*>(data_ptr);
        return static_cast<T*>(raw_mutable_data<Context>(TypeMeta::Make<T>()));
    }

    void Share(const Tensor& other) {
        CHECK_EQ(size_, other.size_);
        memory_ = other.memory_;
        meta_ = other.meta_;
        capacity_ = other.capacity_;
    }

    void Replace(const Tensor& other) {
        memory_ = other.memory_;
        meta_ = other.meta_;
        capacity_ = other.capacity_;
        size_ = other.size_;
        dims_ = other.dims_;
    }

    void Reset() {
        size_ = capacity_ = 0;
        meta_ = TypeMeta();
        dims_.clear();
        memory_.reset();
    }

    void Release() {
        memory_.reset();
    }

 private:
    vector<TIndex> dims_;
    TIndex size_ = 0, capacity_ = 0;
    TypeMeta meta_;
    string name_;
    shared_ptr<MixedMemory> memory_;
};

}    // namespace dragon

#endif    // DRAONG_CORE_TENSOR_H_