#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/array/initialize_op.h"
#include "dragon/utils/conversions.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLRangeOp<Context>::DoRunWithType() {
  int num_args;
  double start = 0., limit, delta;
  slice(0, &num_args);
  if (num_args == 2) {
    limit = slice(0), delta = slice(1);
  } else if (num_args == 3) {
    start = slice(0), limit = slice(1), delta = slice(2);
  } else {
    LOG(FATAL) << "Unexcepted number of slice arguments: " << num_args;
  }

  auto count = (int64_t)std::ceil((limit - start) / delta);
  CHECK_GT(count, 0) << "\nInvalid generating range: "
                     << "[" << start << ", " << limit
                     << ") with delta = " << delta << ".";

  CNNLSetTensorDesc<T>(output_desc_, {count});
  if (TypeMeta::Id<T>() == TypeMeta::Id<float16>()) {
    const auto start_fpcast = float(start);
    const auto delta_fpcast = float(delta);
    CNNL_CHECK(cnnlArange_v2(
        ctx()->cnnl_handle(),
        CNNL_COMPUTATION_FAST,
        &start_fpcast,
        &delta_fpcast,
        output_desc_,
        Output(0)->Reshape({count})->template mutable_data<T, Context>()));
  } else {
    const auto start_fpcast = convert::To<T>(float(start));
    const auto delta_fpcast = convert::To<T>(float(delta));
    CNNL_CHECK(cnnlArange_v2(
        ctx()->cnnl_handle(),
        CNNL_COMPUTATION_FAST,
        &start_fpcast,
        &delta_fpcast,
        output_desc_,
        Output(0)->Reshape({count})->template mutable_data<T, Context>()));
  }
}

DEPLOY_CNNL_OPERATOR(Range);

DEFINE_OP_REPEATED_ARG(double, CNNLRangeOp, slice);

} // namespace dragon

#endif // USE_MLU
