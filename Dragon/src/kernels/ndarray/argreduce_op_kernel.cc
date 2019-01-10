#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! ArgMax <T = ?, Device = CPU> */

template <typename T>
void _ArgMax(
    const int               outer_dim,
    const int               inner_dim,
    const int               axis_dim,
    const int               top_k,
    const T*                x,
    int64_t*                indices,
    T*                      values) {
    for (int oix = 0; oix < outer_dim; ++oix) {
        for (int iix = 0; iix < inner_dim; ++iix) {
            const T* X = x + (oix * axis_dim * inner_dim + iix);
            const int y_offset = oix * top_k * inner_dim + iix;
            vector< pair<T, int64_t> > vec(axis_dim);
            for (int j = 0; j < axis_dim; ++j)
                vec[j] = std::make_pair(X[j * inner_dim], j);
            std::partial_sort(
                vec.begin(), vec.begin() + top_k, vec.end(),
                    std::greater< pair<T, int64_t> >());
            for (int j = 0; j < top_k; ++j) {
                indices[y_offset + j * inner_dim] = vec[j].second;
                if (values) values[y_offset + j * inner_dim] = vec[j].first;
            }
        }
    }
}

/*! ArgMin <T = ?, Device = CPU> */

template <typename T>
void _ArgMin(
    const int               outer_dim,
    const int               inner_dim,
    const int               axis_dim,
    const int               top_k,
    const T*                x,
    int64_t*                indices,
    T*                      values) {
    for (int oix = 0; oix < outer_dim; ++oix) {
        for (int iix = 0; iix < inner_dim; ++iix) {
            const T* X = x + (oix * axis_dim * inner_dim + iix);
            const int y_offset = oix * top_k * inner_dim + iix;
            vector< pair<T, int64_t> > vec(axis_dim);
            for (int j = 0; j < axis_dim; ++j)
                vec[j] = std::make_pair(X[j * inner_dim], j);
            std::partial_sort(vec.begin(), vec.begin() + top_k, vec.end());
            for (int j = 0; j < top_k; ++j) {
                indices[y_offset + j * inner_dim] = vec[j].second;
                if (values) values[y_offset + j * inner_dim] = vec[j].first;
            }
        }
    }
}

/*! Kernel Launchers */

#define DEFINE_ARGREDUCE_KERNEL_LAUNCHER(name, T) \
    template<> void name<T, CPUContext>( \
        const int               outer_dim, \
        const int               inner_dim, \
        const int               axis_dim, \
        const int               top_k, \
        const T*                x, \
        int64_t*                indices, \
        T*                      values, \
        CPUContext*             ctx) { \
        _##name<T>(outer_dim, inner_dim, axis_dim, \
            top_k, x, indices, values); \
    }

DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMax, bool);
DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMax, int8_t);
DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMax, uint8_t);
DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMax, int);
DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMax, int64_t);
DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMax, float);
DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMax, double);

DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMin, bool);
DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMin, int8_t);
DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMin, uint8_t);
DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMin, int);
DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMin, int64_t);
DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMin, float);
DEFINE_ARGREDUCE_KERNEL_LAUNCHER(ArgMin, double);

/*! ArgMax <T = float16, Device = CPU> */

template<> void ArgMax<float16, CPUContext>(
    const int               outer_dim,
    const int               inner_dim,
    const int               axis_dim,
    const int               top_k,
    const float16*          x,
    int64_t*                indices,
    float16*                values,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

/*! ArgMin <T = float16, Device = CPU> */

template<> void ArgMin<float16, CPUContext>(
    const int               outer_dim,
    const int               inner_dim,
    const int               axis_dim,
    const int               top_k,
    const float16*          x,
    int64_t*                indices,
    float16*                values,
    CPUContext*             ctx) {
    CPU_FP16_NOT_SUPPORTED;
}

#undef DEFINE_ARGREDUCE_KERNEL_LAUNCHER

}  // namespace kernel

}  // namepsace dragon