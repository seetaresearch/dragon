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
void MPICollectiveOpImpl::Send(const T* x, int N, int peer) {
#ifdef USE_MPI
  MPI_Send(x, N, data_type<T>(), peer, 0, mpi_comm_);
#else
  MPI_NOT_COMPILED;
#endif
}

template <typename T>
void MPICollectiveOpImpl::Recv(T* y, int N, int peer) {
#ifdef USE_MPI
  MPI_Recv(y, N, data_type<T>(), peer, 0, mpi_comm_, MPI_STATUS_IGNORE);
#else
  MPI_NOT_COMPILED;
#endif
}

template <typename T>
void MPICollectiveOpImpl::Broadcast(const T* x, T* y, int N) {
  CHECK(x == y) << "\nUnsupported non-inplace MPI broadcast.";
#ifdef USE_MPI
  MPI_Bcast(y, N, data_type<T>(), comm_root_, mpi_comm_);
#else
  MPI_NOT_COMPILED;
#endif
}

template <typename T>
void MPICollectiveOpImpl::AllReduce(const T* x, T* y, int N) {
#ifdef USE_MPI
  MPI_Allreduce(x == y ? MPI_IN_PLACE : x, y, N, data_type<T>(), reduction(), mpi_comm_);
#else
  MPI_NOT_COMPILED;
#endif
}

template <typename T>
void MPICollectiveOpImpl::ReduceScatter(const T* x, T* y, int N) {
#ifdef USE_MPI
  vector<int> counts(comm_size_, N);
  MPI_Reduce_scatter(x, y, counts.data(), data_type<T>(), reduction(), mpi_comm_);
#else
  MPI_NOT_COMPILED;
#endif
}

template <typename T>
void MPICollectiveOpImpl::AllGather(const T* x, T* y, int N) {
#ifdef USE_MPI
  MPI_Allgather(x, N, data_type<T>(), y, N, data_type<T>(), mpi_comm_);
#else
  MPI_NOT_COMPILED;
#endif
}
// clang-format on

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

MPI_Op MPICollectiveOpImpl::reduction() {
#ifdef USE_MPI
  static Map<string, MPI_Op> m{
      {"SUM", MPI_SUM},
      {"PROD", MPI_PROD},
      {"MIN", MPI_MIN},
      {"MAX", MPI_MAX},
  };
  auto it = m.find(reduction_);
  CHECK(it != m.end()) << "\nUnsupported MPI reduction: " << reduction_;
  return it->second;
#else
  MPI_NOT_COMPILED;
  return nullptr;
#endif
}

#define INSTANTIATE_API(T)                                                    \
  template DRAGON_API MPI_Datatype MPICollectiveOpImpl::data_type<T>();       \
  template DRAGON_API void MPICollectiveOpImpl::Send(const T*, int, int);     \
  template DRAGON_API void MPICollectiveOpImpl::Recv(T*, int, int);           \
  template DRAGON_API void MPICollectiveOpImpl::Broadcast(const T*, T*, int); \
  template DRAGON_API void MPICollectiveOpImpl::AllReduce(const T*, T*, int); \
  template DRAGON_API void MPICollectiveOpImpl::AllGather(const T*, T*, int); \
  template DRAGON_API void MPICollectiveOpImpl::ReduceScatter(                \
      const T*, T*, int);

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
