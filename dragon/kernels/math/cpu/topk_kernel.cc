#include "dragon/kernels/math/op_kernels.h"

namespace dragon {

namespace kernels {

namespace {

template <typename T>
struct LargestFunctor {
  bool operator()(
      const std::pair<T, int64_t>& lhs,
      const std::pair<T, int64_t>& rhs) const {
    return lhs.first > rhs.first ||
        (lhs.first == rhs.first && lhs.second < rhs.second);
  }
};

template <typename T>
struct SmallestFunctor {
  bool operator()(
      const std::pair<T, int64_t>& lhs,
      const std::pair<T, int64_t>& rhs) const {
    return lhs.first < rhs.first ||
        (lhs.first == rhs.first && lhs.second < rhs.second);
  }
};

template <typename T, class CompareFunctor>
void _TopK(
    const int N,
    const int S,
    const int C,
    const int K,
    const int largest,
    const T* x,
    T* value,
    int64_t* index) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < S; ++j) {
      auto* offset_x = x + (i * C * S + j);
      vector<std::pair<T, int64_t>> head_data;
      head_data.reserve(K);
      for (int k = 0; k < K && k < C; ++k) {
        head_data.emplace_back(*offset_x, k);
        offset_x += S;
      }
      std::priority_queue<
          std::pair<T, int64_t>,
          vector<std::pair<T, int64_t>>,
          CompareFunctor>
          pq(CompareFunctor(), std::move(head_data));
      if (largest > 0) {
        for (int k = K; k < C; ++k) {
          if (pq.top().first < *offset_x) {
            pq.pop();
            pq.emplace(*offset_x, k);
          }
          offset_x += S;
        }
      } else {
        for (int k = K; k < C; ++k) {
          if (pq.top().first > *offset_x) {
            pq.pop();
            pq.emplace(*offset_x, k);
          }
          offset_x += S;
        }
      }
      auto y_offset = i * K * S + j + (K - 1) * S;
      while (!pq.empty()) {
        const auto& p = pq.top();
        value[y_offset] = p.first;
        index[y_offset] = p.second;
        pq.pop();
        y_offset -= S;
      }
    }
  }
}

} // namespace

#define DEFINE_KERNEL_LAUNCHER(T) \
  template <>                     \
  void TopK<T, CPUContext>(       \
      const int N,                \
      const int S,                \
      const int C,                \
      const int K,                \
      const int largest,          \
      const T* x,                 \
      T* value,                   \
      int64_t* index,             \
      CPUContext* ctx) {          \
    CPU_UNSUPPORTED_DTYPE(T);     \
  }

DEFINE_KERNEL_LAUNCHER(float16);
DEFINE_KERNEL_LAUNCHER(bfloat16);
#undef DEFINE_KERNEL_LAUNCHER

#define DEFINE_KERNEL_LAUNCHER(T)                                         \
  template <>                                                             \
  void TopK<T, CPUContext>(                                               \
      const int N,                                                        \
      const int S,                                                        \
      const int C,                                                        \
      const int K,                                                        \
      const int largest,                                                  \
      const T* x,                                                         \
      T* value,                                                           \
      int64_t* index,                                                     \
      CPUContext* ctx) {                                                  \
    if (largest > 0) {                                                    \
      _TopK<T, LargestFunctor<T>>(N, S, C, K, largest, x, value, index);  \
    } else {                                                              \
      _TopK<T, SmallestFunctor<T>>(N, S, C, K, largest, x, value, index); \
    }                                                                     \
  }

DEFINE_KERNEL_LAUNCHER(uint8_t);
DEFINE_KERNEL_LAUNCHER(int8_t);
DEFINE_KERNEL_LAUNCHER(int);
DEFINE_KERNEL_LAUNCHER(int64_t);
DEFINE_KERNEL_LAUNCHER(float);
DEFINE_KERNEL_LAUNCHER(double);
#undef DEFINE_KERNEL_LAUNCHER

} // namespace kernels

} // namespace dragon
