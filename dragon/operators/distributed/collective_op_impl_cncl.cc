#include "dragon/operators/distributed/collective_op_impl_cncl.h"
#include "dragon/core/context_mlu.h"

namespace dragon {

void CNCLCollectiveOpImpl::SetComm(
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
void CNCLCollectiveOpImpl::AllReduce(
    const T* sendbuf,
    T* recvbuf,
    int count,
    cnrtQueue_t stream) {
#ifdef USE_MLU
  CNCL_CHECK(cnclAllReduce(
      sendbuf, recvbuf, count, data_type<T>(), cnclSum, cncl_comm(), stream));
#else
  MLU_NOT_COMPILED;
#endif
}

template <typename T>
void CNCLCollectiveOpImpl::Bcast(T* buf, int count, cnrtQueue_t stream) {
#ifdef USE_MLU
  CNCL_CHECK(
      cnclBcast(buf, count, data_type<T>(), comm_root_, cncl_comm(), stream));
#else
  MLU_NOT_COMPILED;
#endif
}

template <typename T>
void CNCLCollectiveOpImpl::AllGather(
    const T* sendbuf,
    T* recvbuf,
    int sendcount,
    cnrtQueue_t stream) {
#ifdef USE_MLU
  CNCL_CHECK(cnclAllGather(
      sendbuf, recvbuf, sendcount, data_type<T>(), cncl_comm(), stream));
#else
  MLU_NOT_COMPILED;
#endif
}

template <typename T>
void CNCLCollectiveOpImpl::ReduceScatter(
    const T* sendbuf,
    T* recvbuf,
    int recvcount,
    cnrtQueue_t stream) {
#ifdef USE_MLU
  CNCL_CHECK(cnclReduceScatter(
      sendbuf,
      recvbuf,
      recvcount,
      data_type<T>(),
      cnclSum,
      cncl_comm(),
      stream));
#else
  MLU_NOT_COMPILED;
#endif
}

cnclComm_t CNCLCollectiveOpImpl::cncl_comm() {
#ifdef USE_MLU
  auto& ctx = MLUContext::objects();
  auto device = ctx.GetDevice();
  auto ret = ctx.cncl_comm(device, group_str_, nullptr, comm_size_, comm_rank_);
  if (ret == nullptr) {
    cnclCliqueId comm_uuid;
    if (comm_rank_ == comm_root_) {
      CNCL_CHECK(cnclGetCliqueId(&comm_uuid));
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
    ret = ctx.cncl_comm(device, group_str_, &comm_uuid, comm_size_, comm_rank_);
  }
  return ret;
#else
  MLU_NOT_COMPILED;
  return nullptr;
#endif
}

template <typename T>
cnclDataType_t CNCLCollectiveOpImpl::data_type() {
#ifdef USE_MLU
  static Map<TypeId, cnclDataType_t> m{
      {TypeMeta::Id<bool>(), cnclUint8},
      {TypeMeta::Id<uint8_t>(), cnclUint8},
      {TypeMeta::Id<int8_t>(), cnclInt8},
      {TypeMeta::Id<int>(), cnclInt32},
      {TypeMeta::Id<float16>(), cnclFloat16},
      {TypeMeta::Id<float>(), cnclFloat32},
  };
  auto it = m.find(TypeMeta::Id<T>());
  CHECK(it != m.end()) << "\nUnsupported CNCL type: "
                       << dtypes::to_string(TypeMeta::Make<T>());
  return it->second;
#else
  MLU_NOT_COMPILED;
  return 0;
#endif
}

#define INSTANTIATE_API(T)                                                    \
  template DRAGON_API cnclDataType_t CNCLCollectiveOpImpl::data_type<T>();    \
  template DRAGON_API void CNCLCollectiveOpImpl::Bcast(T*, int, cnrtQueue_t); \
  template DRAGON_API void CNCLCollectiveOpImpl::AllReduce(                   \
      const T*, T*, int, cnrtQueue_t);                                        \
  template DRAGON_API void CNCLCollectiveOpImpl::AllGather(                   \
      const T*, T*, int, cnrtQueue_t);                                        \
  template DRAGON_API void CNCLCollectiveOpImpl::ReduceScatter(               \
      const T*, T*, int, cnrtQueue_t);

INSTANTIATE_API(bool);
INSTANTIATE_API(int8_t);
INSTANTIATE_API(uint8_t);
INSTANTIATE_API(int);
INSTANTIATE_API(int64_t);
INSTANTIATE_API(float16);
INSTANTIATE_API(float);
INSTANTIATE_API(double);
#undef INSTANTIATE_API

} // namespace dragon
