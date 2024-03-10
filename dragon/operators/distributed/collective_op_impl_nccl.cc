#include "dragon/operators/distributed/collective_op_impl_nccl.h"
#include "dragon/core/context_cuda.h"

namespace dragon {

void NCCLCollectiveOpImpl::SetComm(
    const int64_t mpi_comm,
    const int64_t mpi_group,
    const vec64_t ranks,
    const int64_t root) {
  if (mpi_comm == 0) return;
#ifdef USE_MPI
  // The given group should be created before.
  group_str_ = str::to(ranks);
  mpi_comm_ = (MPI_Comm)mpi_comm;
  mpi_group_ = (MPI_Group)mpi_group;
  CHECK(mpi_group != 0) << "\nSet comm from an invalid MPI group.";
  // Collect comm and rank information.
  MPI_Comm_size(mpi_comm_, &comm_size_);
  MPI_Comm_rank(mpi_comm_, &comm_rank_);
  // Translate the world root into the group.
  MPI_Group world;
  MPI_Comm_group(MPI_COMM_WORLD, &world);
  auto world_root = int(ranks[root]);
  MPI_Group_translate_ranks(world, 1, &world_root, mpi_group_, &comm_root_);
  CHECK(comm_root_ != MPI_UNDEFINED) << "\nComm root is not in the MPI group.";
#else
  MPI_NOT_COMPILED;
#endif
}

// clang-format off
template <typename T>
void NCCLCollectiveOpImpl::Broadcast(const T* x, T* y, int N, cudaStream_t stream) {
#ifdef USE_NCCL
  NCCL_CHECK(ncclBroadcast(x, y, N, data_type<T>(), comm_root_, nccl_comm(), stream));
#else
  NCCL_NOT_COMPILED;
#endif
}

template <typename T>
void NCCLCollectiveOpImpl::AllReduce(const T* x, T* y, int N, cudaStream_t stream) {
#ifdef USE_NCCL
  NCCL_CHECK(ncclAllReduce(x, y, N, data_type<T>(), reduction(), nccl_comm(), stream));
#else
  NCCL_NOT_COMPILED;
#endif
}

template <typename T>
void NCCLCollectiveOpImpl::ReduceScatter(const T* x, T* y, int N, cudaStream_t stream) {
#ifdef USE_NCCL
  NCCL_CHECK(ncclReduceScatter(x, y, N, data_type<T>(), reduction(), nccl_comm(), stream));
#else
  NCCL_NOT_COMPILED;
#endif
}

template <typename T>
void NCCLCollectiveOpImpl::AllGather(const T* x, T* y, int N, cudaStream_t stream) {
#ifdef USE_NCCL
  NCCL_CHECK(ncclAllGather(x, y, N, data_type<T>(), nccl_comm(), stream));
#else
  NCCL_NOT_COMPILED;
#endif
}

ncclComm_t NCCLCollectiveOpImpl::nccl_comm() {
#ifdef USE_NCCL
  auto& ctx = CUDAContext::objects();
  auto device = ctx.GetDevice();
  auto ret = ctx.nccl_comm(device, group_str_, nullptr, comm_size_, comm_rank_);
  if (ret == nullptr) {
    ncclUniqueId comm_uuid;
    if (comm_rank_ == comm_root_) NCCL_CHECK(ncclGetUniqueId(&comm_uuid));
#ifdef USE_MPI
    MPI_Bcast((uint8_t*)&comm_uuid, sizeof(comm_uuid), MPI_UNSIGNED_CHAR, comm_root_, mpi_comm_);
#else
    MPI_NOT_COMPILED;
#endif
    ret = ctx.nccl_comm(device, group_str_, &comm_uuid, comm_size_, comm_rank_);
  }
  return ret;
#else
  NCCL_NOT_COMPILED;
  return nullptr;
#endif
}
// clang-format on

template <typename T>
ncclDataType_t NCCLCollectiveOpImpl::data_type() {
#ifdef USE_NCCL
  static Map<TypeId, ncclDataType_t> m{
      {TypeMeta::Id<bool>(), ncclUint8},
      {TypeMeta::Id<uint8_t>(), ncclUint8},
      {TypeMeta::Id<int8_t>(), ncclInt8},
      {TypeMeta::Id<int>(), ncclInt32},
      {TypeMeta::Id<int64_t>(), ncclInt64},
      {TypeMeta::Id<float16>(), ncclFloat16},
      {TypeMeta::Id<bfloat16>(), ncclBfloat16},
      {TypeMeta::Id<float>(), ncclFloat32},
      {TypeMeta::Id<double>(), ncclFloat64},
  };
  auto it = m.find(TypeMeta::Id<T>());
  CHECK(it != m.end()) << "\nUnsupported dtype for NCCL.";
  return it->second;
#else
  NCCL_NOT_COMPILED;
  return 0;
#endif
}

ncclRedOp_t NCCLCollectiveOpImpl::reduction() {
#ifdef USE_NCCL
  static Map<string, ncclRedOp_t> m{
      {"SUM", ncclSum},
      {"PROD", ncclProd},
      {"MIN", ncclMin},
      {"MAX", ncclMax},
  };
  auto it = m.find(reduction_);
  CHECK(it != m.end()) << "\nUnsupported NCCL reduction: " << reduction_;
  return it->second;
#else
  NCCL_NOT_COMPILED;
  return 0;
#endif
}

#define INSTANTIATE_API(T)                                                 \
  template DRAGON_API ncclDataType_t NCCLCollectiveOpImpl::data_type<T>(); \
  template DRAGON_API void NCCLCollectiveOpImpl::Broadcast(                \
      const T*, T*, int, cudaStream_t);                                    \
  template DRAGON_API void NCCLCollectiveOpImpl::AllReduce(                \
      const T*, T*, int, cudaStream_t);                                    \
  template DRAGON_API void NCCLCollectiveOpImpl::ReduceScatter(            \
      const T*, T*, int, cudaStream_t);                                    \
  template DRAGON_API void NCCLCollectiveOpImpl::AllGather(                \
      const T*, T*, int, cudaStream_t);

INSTANTIATE_API(bool);
INSTANTIATE_API(int8_t);
INSTANTIATE_API(uint8_t);
INSTANTIATE_API(int);
INSTANTIATE_API(int64_t);
INSTANTIATE_API(float16);
INSTANTIATE_API(bfloat16);
INSTANTIATE_API(float);
INSTANTIATE_API(double);
#undef INSTANTIATE_API

} // namespace dragon
