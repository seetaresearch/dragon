#include "core/context.h"
#include "contrib/rcnn/bbox_utils.h"

namespace dragon {

namespace rcnn {

template <typename T>
T IoU(const T A[], const T B[]) {
    if (A[0] > B[2] || A[1] > B[3] ||
        A[2] < B[0] || A[3] < B[1]) return 0;
    const T x1 = std::max(A[0], B[0]);
    const T y1 = std::max(A[1], B[1]);
    const T x2 = std::min(A[2], B[2]);
    const T y2 = std::min(A[3], B[3]);
    const T width = std::max((T)0, x2 - x1 + 1);
    const T height = std::max((T)0, y2 - y1 + 1);
    const T area = width * height;
    const T A_area = (A[2] - A[0] + 1) * (A[3] - A[1] + 1);
    const T B_area = (B[2] - B[0] + 1) * (B[3] - B[1] + 1);
    return area / (A_area + B_area - area);
}

template <> void ApplyNMS<float, CPUContext>(
    const int               num_boxes,
    const int               max_keeps,
    const float             thresh,
    const float*            boxes,
    int64_t*                keep_indices,
    int&                    num_keep,
    CPUContext*             ctx) {
    int count = 0;
    std::vector<char> is_dead(num_boxes);
    for (int i = 0; i < num_boxes; ++i) is_dead[i] = 0;
    for (int i = 0; i < num_boxes; ++i) {
        if (is_dead[i]) continue;
        keep_indices[count++] = i;
        if (count == max_keeps) break;
        for (int j = i + 1; j < num_boxes; ++j)
            if (!is_dead[j] && IoU(&boxes[i * 5],
                                   &boxes[j * 5]) > thresh)
                is_dead[j] = 1;
    }
    num_keep = count;
}

}  // namespace rcnn

}  // namespace dragon