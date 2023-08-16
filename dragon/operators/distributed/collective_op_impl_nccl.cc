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
  mpi_comm_ = (MPI_Comm)mpi_comm;
  mpi_group_ = (MPI_Group)mpi_group;
  CHECK(mpi_group != 0) << "\nFailed to initialize an invalid mpi group.";
  // Collect comm and rank information.
  MPI_Comm_size(mpi_comm_, &comm_size_);
  MPI_Comm_rank(mpi_comm_, &comm_rank_);
  // Translate the world root into the group.
  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  auto world_root = int(ranks[root]);
  group_str_ = str::to(ranks);
  MPI_Group_translate_ranks(
      world_group, 1, &world_root, mpi_group_, &comm_root_);
  CHECK(comm_root_ != MPI_UNDEFINED) << "\nRoot is not in the MPI group.";
#else
  MPI_NOT_COMPILED;
#endif
}

template <typename T>
void NCCLCollectiveOpImpl::Bcast(T* buf, int count, cudaStream_t stream) {
#ifdef USE_NCCL
  NCCL_CHECK(
      ncclBcast(buf, count, data_type<T>(), comm_root_, nccl_comm(), stream));
#else
  NCCL_NOT_COMPILED;
#endif
}

template <typename T>
void NCCLCollectiveOpImpl::AllReduce(
    const T* sendbuf,
    T* recvbuf,
    int count,
    cudaStream_t stream) {
#ifdef USE_NCCL
  NCCL_CHECK(ncclAllReduce(
      sendbuf, recvbuf, count, data_type<T>(), ncclSum, nccl_comm(), stream));
#else
  NCCL_NOT_COMPILED;
#endif
}

template <typename T>
void NCCLCollectiveOpImpl::AllGather(
    const T* sendbuf,
    T* recvbuf,
    int sendcount,
    cudaStream_t stream) {
#ifdef USE_NCCL
  NCCL_CHECK(ncclAllGather(
      sendbuf, recvbuf, sendcount, data_type<T>(), nccl_comm(), stream));
#else
  NCCL_NOT_COMPILED;
#endif
}

template <typename T>
void NCCLCollectiveOpImpl::ReduceScatter(
    const T* sendbuf,
    T* recvbuf,
    int recvcount,
    cudaStream_t stream) {
#ifdef USE_NCCL
  NCCL_CHECK(ncclReduceScatter(
      sendbuf,
      recvbuf,
      recvcount,
      data_type<T>(),
      ncclSum,
      nccl_comm(),
      stream));
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
    if (comm_rank_ == comm_root_) {
      NCCL_CHECK(ncclGetUniqueId(&comm_uuid));
    }
#ifdef USE_MPI
    MPI_Bcast(
        (uint8_t*)&comm_uuid,
        sizeof(comm_uuid),
        MPI_UNSIGNED_CHAR,
        comm_root_,
        mpi_comm_);
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

#define INSTANTIATE_API(T)                                                     \
  template DRAGON_API ncclDataType_t NCCLCollectiveOpImpl::data_type<T>();     \
  template DRAGON_API void NCCLCollectiveOpImpl::Bcast(T*, int, cudaStream_t); \
  template DRAGON_API void NCCLCollectiveOpImpl::AllReduce(                    \
      const T*, T*, int, cudaStream_t);                                        \
  template DRAGON_API void NCCLCollectiveOpImpl::AllGather(                    \
      const T*, T*, int, cudaStream_t);                                        \
  template DRAGON_API void NCCLCollectiveOpImpl::ReduceScatter(                \
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
