#include "dragon/operators/distributed/collective_op_impl_mpi.h"

namespace dragon {

void MPICollectiveOpImpl::SetComm(
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
  MPI_Group_translate_ranks(
      world_group, 1, &world_root, mpi_group_, &comm_root_);
  CHECK(comm_root_ != MPI_UNDEFINED) << "\nRoot is not in the MPI group.";
#else
  MPI_NOT_COMPILED;
#endif
}

template <typename T>
void MPICollectiveOpImpl::Recv(T* buf, int count, int from) {
#ifdef USE_MPI
  MPI_Recv(buf, count, data_type<T>(), from, 0, mpi_comm_, MPI_STATUS_IGNORE);
#else
  MPI_NOT_COMPILED;
#endif
}

template <typename T>
void MPICollectiveOpImpl::IRecv(T* buf, int count, int from, MPI_Request* req) {
#ifdef USE_MPI
  MPI_Irecv(buf, count, data_type<T>(), from, 0, mpi_comm_, req);
#else
  MPI_NOT_COMPILED;
#endif
}

template <typename T>
void MPICollectiveOpImpl::Send(const T* buf, int count, int to) {
#ifdef USE_MPI
  MPI_Send(buf, count, data_type<T>(), to, 0, mpi_comm_);
#else
  MPI_NOT_COMPILED;
#endif
}

template <typename T>
void MPICollectiveOpImpl::SendRecv(
    const T* sendbuf,
    int sendcount,
    int to,
    T* recvbuf,
    int recvcount,
    int from) {
#ifdef USE_MPI
  MPI_Sendrecv(
      sendbuf,
      sendcount,
      data_type<T>(),
      to,
      0,
      recvbuf,
      recvcount,
      data_type<T>(),
      from,
      0,
      mpi_comm_,
      MPI_STATUS_IGNORE);
#else
  MPI_NOT_COMPILED;
#endif
}

template <typename T>
void MPICollectiveOpImpl::Bcast(T* buf, int count) {
#ifdef USE_MPI
  MPI_Bcast(buf, count, data_type<T>(), comm_root_, mpi_comm_);
#else
  MPI_NOT_COMPILED;
#endif
}

template <typename T>
void MPICollectiveOpImpl::AllReduce(const T* sendbuf, T* recvbuf, int count) {
#ifdef USE_MPI
  MPI_Allreduce(
      sendbuf == recvbuf ? MPI_IN_PLACE : sendbuf,
      recvbuf,
      count,
      data_type<T>(),
      MPI_SUM,
      mpi_comm_);
#else
  MPI_NOT_COMPILED;
#endif
}

template <typename T>
void MPICollectiveOpImpl::AllGather(
    const T* sendbuf,
    T* recvbuf,
    int sendcount) {
#ifdef USE_MPI
  MPI_Allgather(
      sendbuf,
      sendcount,
      data_type<T>(),
      recvbuf,
      sendcount,
      data_type<T>(),
      mpi_comm_);
#else
  MPI_NOT_COMPILED;
#endif
}

template <typename T>
void MPICollectiveOpImpl::ReduceScatter(
    const T* sendbuf,
    T* recvbuf,
    int recvcount) {
#ifdef USE_MPI
  vector<int> recvcounts(comm_size_, recvcount);
  MPI_Reduce_scatter(
      sendbuf, recvbuf, recvcounts.data(), data_type<T>(), MPI_SUM, mpi_comm_);
#else
  MPI_NOT_COMPILED;
#endif
}

template <typename T>
MPI_Datatype MPICollectiveOpImpl::data_type() {
#ifdef USE_MPI
  static MPI_Datatype unknown_dtype = MPI_DATATYPE_NULL;
  static Map<TypeId, MPI_Datatype> m{
      {TypeMeta::Id<bool>(), MPI_CHAR},
      {TypeMeta::Id<int8_t>(), MPI_SIGNED_CHAR},
      {TypeMeta::Id<uint8_t>(), MPI_UNSIGNED_CHAR},
      {TypeMeta::Id<int>(), MPI_INT},
      {TypeMeta::Id<int64_t>(), MPI_LONG_LONG},
      {TypeMeta::Id<float16>(), MPI_UNSIGNED_SHORT},
      {TypeMeta::Id<bfloat16>(), MPI_UNSIGNED_SHORT},
      {TypeMeta::Id<float>(), MPI_FLOAT},
      {TypeMeta::Id<double>(), MPI_DOUBLE},
  };
  auto it = m.find(TypeMeta::Id<T>());
  return it != m.end() ? it->second : unknown_dtype;
#else
  MPI_NOT_COMPILED;
  return nullptr;
#endif
}

#define INSTANTIATE_API(T)                                                    \
  template DRAGON_API MPI_Datatype MPICollectiveOpImpl::data_type<T>();       \
  template DRAGON_API void MPICollectiveOpImpl::Send(const T*, int, int);     \
  template DRAGON_API void MPICollectiveOpImpl::Recv(T*, int, int);           \
  template DRAGON_API void MPICollectiveOpImpl::IRecv(                        \
      T*, int, int, MPI_Request*);                                            \
  template DRAGON_API void MPICollectiveOpImpl::Bcast(T*, int);               \
  template DRAGON_API void MPICollectiveOpImpl::AllReduce(const T*, T*, int); \
  template DRAGON_API void MPICollectiveOpImpl::AllGather(const T*, T*, int); \
  template DRAGON_API void MPICollectiveOpImpl::ReduceScatter(                \
      const T*, T*, int);                                                     \
  template DRAGON_API void MPICollectiveOpImpl::SendRecv(                     \
      const T*, int, int, T*, int, int);

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
