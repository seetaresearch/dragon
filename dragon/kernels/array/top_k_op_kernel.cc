#include "dragon/utils/op_kernels.h"

namespace dragon {

namespace kernel {

namespace {

template <typename T>
struct LargestComp {
  bool operator()(
      const std::pair<T, int64_t>& a,
      const std::pair<T, int64_t>& b) const {
    return a.first > b.first || (a.first == b.first && a.second < b.second);
  }
};

template <typename T>
struct SmallestComp {
  bool operator()(
      const std::pair<T, int64_t>& a,
      const std::pair<T, int64_t>& b) const {
    return a.first < b.first || (a.first == b.first && a.second < b.second);
  }
};

template <typename T, class Comp>
void _TopK(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const int top_k,
    const int largest,
    const T* x,
    T* value,
    int64_t* index) {
  for (int i = 0; i < outer_dim; ++i) {
    for (int j = 0; j < inner_dim; ++j) {
      auto* offset_x = x + (i * axis_dim * inner_dim + j);
      vector<std::pair<T, int64_t>> head_data;
      head_data.reserve(top_k);
      for (int k = 0; k < top_k && k < axis_dim; ++k) {
        head_data.emplace_back(*offset_x, k);
        offset_x += inner_dim;
      }
      std::priority_queue<
          std::pair<T, int64_t>,
          vector<std::pair<T, int64_t>>,
          Comp>
          pq(Comp(), std::move(head_data));
      if (largest > 0) {
        for (int k = top_k; k < axis_dim; ++k) {
          if (pq.top().first < *offset_x) {
            pq.pop();
            pq.emplace(*offset_x, k);
          }
          offset_x += inner_dim;
        }
      } else {
        for (int k = top_k; k < axis_dim; ++k) {
          if (pq.top().first > *offset_x) {
            pq.pop();
            pq.emplace(*offset_x, k);
          }
          offset_x += inner_dim;
        }
      }
      auto y_offset = i * top_k * inner_dim + j + (top_k - 1) * inner_dim;
      while (!pq.empty()) {
        const auto& p = pq.top();
        value[y_offset] = p.first;
        index[y_offset] = p.second;
        pq.pop();
        y_offset -= inner_dim;
      }
    }
  }
}

} // namespace

/* ------------------- Launcher Separator ------------------- */

template <>
void TopK<float16, CPUContext>(
    const int outer_dim,
    const int inner_dim,
    const int axis_dim,
    const int top_k,
    const int largest,
    const float16* x,
    float16* value,
    int64_t* index,
    CPUContext* ctx) {
  CPU_FP16_NOT_SUPPORTED;
}

#define DEFINE_KERNEL_LAUNCHER(T)                                           \
  template <>                                                               \
  void TopK<T, CPUContext>(                                                 \
      const int outer_dim,                                                  \
      const int inner_dim,                                                  \
      const int axis_dim,                                                   \
      const int top_k,                                                      \
      const int largest,                                                    \
      const T* x,                                                           \
      T* value,                                                             \
      int64_t* index,                                                       \
      CPUContext* ctx) {                                                    \
    if (largest > 0) {                                                      \
      _TopK<T, LargestComp<T>>(                                             \
          outer_dim, inner_dim, axis_dim, top_k, largest, x, value, index); \
    } else {                                                                \
      _TopK<T, SmallestComp<T>>(                                            \
          outer_dim, inner_dim, axis_dim, top_k, largest, x, value, index); \
    }                                                                       \
  }

DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernel

} // namespace dragon
