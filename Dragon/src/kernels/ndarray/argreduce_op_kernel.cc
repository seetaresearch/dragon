#include "utils/op_kernel.h"
#include "utils/omp_alternative.h"

namespace dragon {

namespace kernel {

/*! Argmax <T = float32, Device = CPU> */

template<> void Argmax<float, CPUContext>(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const int               top_k,
    const float*            x,
    int64_t*                indices,
    float*                  values,
    CPUContext*             ctx) {
    vector<pair<float, int> > vec(axis_dim);
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < axis_dim; ++j) 
            vec[j] = std::make_pair(x[(i / inner_dim * axis_dim + j) *
                inner_dim + i % inner_dim], j);
        std::partial_sort(
            vec.begin(), vec.begin() + top_k, vec.end(),
                std::greater< pair<float, int> >());
        for (int j = 0; j < top_k; ++j) {
            TIndex y_idx = (i / inner_dim * top_k + j) *
                inner_dim + i % inner_dim;
            indices[y_idx] = vec[j].second;
            if (values) values[y_idx] = vec[j].first;
        }
    }
}

/*! Argmin <T = float32, Device = CPU> */

template<> void Argmin<float, CPUContext>(
    const int               count,
    const int               axis_dim,
    const int               inner_dim,
    const int               top_k,
    const float*            x,
    int64_t*                indices,
    float*                  values,
    CPUContext*             ctx) {
    vector<pair<float, int> > vec(axis_dim);
#ifdef WITH_OMP
    #pragma omp parallel for num_threads(GET_OMP_THREADS(count))
#endif
    for (int i = 0; i < count; ++i) {
        for (int j = 0; j < axis_dim; ++j) 
            vec[j] = std::make_pair(x[(i / inner_dim * axis_dim + j) *
                inner_dim + i % inner_dim], j);
        std::partial_sort(vec.begin(), vec.begin() + top_k, vec.end());
        for (int j = 0; j < top_k; ++j) {
            TIndex y_idx = (i / inner_dim * top_k + j) *
                inner_dim + i % inner_dim;
            indices[y_idx] = vec[j].second;
            if (values) values[y_idx] = vec[j].first;
        }
    }
}

}  // namespace kernel

}  // namepsace dragon