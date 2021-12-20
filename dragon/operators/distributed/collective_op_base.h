/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_DISTRIBUTED_COLLECTIVE_OP_BASE_H_
#define DRAGON_OPERATORS_DISTRIBUTED_COLLECTIVE_OP_BASE_H_

#ifdef USE_MPI

#include <mpi.h>

#include "dragon/core/operator.h"

namespace dragon {

template <class Context>
class CollectiveOpBase : public Operator<Context> {
 public:
  CollectiveOpBase(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws),
        comm_rank_(0),
        comm_size_(1),
        comm_root_(0),
        enable_nccl_(false),
        comm_((MPI_Comm)OP_SINGLE_ARG(int64_t, "comm", 0)),
        group_((MPI_Group)OP_SINGLE_ARG(int64_t, "group", 0)) {
    if ((int64_t)comm_ == 0) return;
    // The given group should be created before.
    CHECK((int64_t)group_ != 0) << "\nEncounter the invalid mpi group.";

    // Collect comm and rank information.
    MPI_Comm_size(comm_, &comm_size_);
    MPI_Comm_rank(comm_, &comm_rank_);

    // Translate the root into the group.
    MPI_Group world_group;
    auto root = OP_SINGLE_ARG(int, "root", 0);
    auto group_world_ranks = OP_REPEATED_ARG(int64_t, "ranks");
    auto group_world_root = int(group_world_ranks[root]);
    group_str_ = Tensor::DimString(group_world_ranks);
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Group_translate_ranks(
        world_group, 1, &group_world_root, group_, &comm_root_);
    CHECK(comm_root_ != MPI_UNDEFINED)
        << "\nRoot is not included in the group.";

    // Check whether the NCCL backend should be enabled.
    // If not, we will fallback to the MPI backend.
#ifdef USE_NCCL
    enable_nccl_ = OP_SINGLE_ARG(string, "backend", "MPI") == "NCCL";
    enable_nccl_ &= (TypeMeta::Id<Context>() == TypeMeta::Id<CUDAContext>());
#else
    enable_nccl_ = false;
#endif
  }

  template <typename T>
  void Recv(T* buf, int count, int from) {
    MPI_Recv(buf, count, mpi_data_type<T>(), from, 0, comm_, MPI_STATUS_IGNORE);
  }

  template <typename T>
  void IRecv(T* buf, int count, int from, MPI_Request* req) {
    MPI_Irecv(buf, count, mpi_data_type<T>(), from, 0, comm_, req);
  }

  template <typename T>
  void Send(const T* buf, int count, int to) {
    MPI_Send(buf, count, mpi_data_type<T>(), to, 0, comm_);
  }

  template <typename T>
  void SendRecv(
      const T* send_buf,
      int send_count,
      int to,
      T* recv_buf,
      int recv_count,
      int from) {
    MPI_Sendrecv(
        send_buf,
        send_count,
        mpi_data_type<T>(),
        to,
        0,
        recv_buf,
        recv_count,
        mpi_data_type<T>(),
        from,
        0,
        comm_,
        MPI_STATUS_IGNORE);
  }

  template <typename T>
  void Broadcast(T* buf, int count) {
    MPI_Bcast(buf, count, mpi_data_type<T>(), comm_root_, comm_);
  }

  template <typename T>
  void AllGather(const T* send_buf, T* recv_buf, int count) {
    MPI_Allgather(
        send_buf,
        count,
        mpi_data_type<T>(),
        recv_buf,
        count,
        mpi_data_type<T>(),
        comm_);
  }

  template <typename T>
  void AllReduce(T* send_buf, T* recv_buf, int count) {
    MPI_Allreduce(
        send_buf == recv_buf ? MPI_IN_PLACE : send_buf,
        recv_buf,
        count,
        mpi_data_type<T>(),
        MPI_SUM,
        comm_);
  }

  template <typename T>
  MPI_Datatype mpi_data_type() {
    static MPI_Datatype unknown_dtype = MPI_DATATYPE_NULL;
    static Map<TypeId, MPI_Datatype> m{
        {TypeMeta::Id<bool>(), MPI_CHAR},
        {TypeMeta::Id<int8_t>(), MPI_SIGNED_CHAR},
        {TypeMeta::Id<uint8_t>(), MPI_UNSIGNED_CHAR},
        {TypeMeta::Id<int>(), MPI_INT},
        {TypeMeta::Id<int64_t>(), MPI_LONG_LONG},
        {TypeMeta::Id<float16>(), MPI_UNSIGNED_SHORT},
        {TypeMeta::Id<float>(), MPI_FLOAT},
        {TypeMeta::Id<double>(), MPI_DOUBLE},
    };
    auto it = m.find(TypeMeta::Id<T>());
    return it != m.end() ? it->second : unknown_dtype;
  }

#ifdef USE_NCCL
  template <typename T>
  ncclDataType_t nccl_data_type() {
    static Map<TypeId, ncclDataType_t> m{
        {TypeMeta::Id<bool>(), ncclChar},
        {TypeMeta::Id<int8_t>(), ncclInt8},
        {TypeMeta::Id<uint8_t>(), ncclUint8},
        {TypeMeta::Id<int>(), ncclInt32},
        {TypeMeta::Id<int64_t>(), ncclInt64},
        {TypeMeta::Id<float16>(), ncclFloat16},
        {TypeMeta::Id<float>(), ncclFloat32},
        {TypeMeta::Id<double>(), ncclFloat64},
    };
    auto it = m.find(TypeMeta::Id<T>());
    CHECK(it != m.end()) << "\nUnsupported dtype for NCCL.";
    return it->second;
  }

  ncclComm_t nccl_comm() {
    auto ret = CUDAContext::objects().nccl_comm(
        this->ctx()->template device(),
        group_str_,
        nullptr,
        comm_size_,
        comm_rank_);
    if (ret == nullptr) {
      ncclUniqueId comm_uuid;
      if (comm_rank_ == comm_root_) {
        // Create a new socket listening at root.
        NCCL_CHECK(ncclGetUniqueId(&comm_uuid));
      }
      Broadcast((uint8_t*)&comm_uuid, sizeof(comm_uuid));
      ret = CUDAContext::objects().nccl_comm(
          this->ctx()->template device(),
          group_str_,
          &comm_uuid,
          comm_size_,
          comm_rank_);
    }
    return ret;
  }
#endif // USE_NCCL

 public:
  MPI_Comm comm_;
  MPI_Group group_;
  string group_str_;
  int comm_size_, comm_rank_, comm_root_;
  bool enable_nccl_;
};

#define USE_COLLECTIVE_FUNCTIONS               \
  using CollectiveOpBase<Context>::Recv;       \
  using CollectiveOpBase<Context>::IRecv;      \
  using CollectiveOpBase<Context>::Send;       \
  using CollectiveOpBase<Context>::SendRecv;   \
  using CollectiveOpBase<Context>::Broadcast;  \
  using CollectiveOpBase<Context>::AllGather;  \
  using CollectiveOpBase<Context>::AllReduce;  \
  using CollectiveOpBase<Context>::comm_;      \
  using CollectiveOpBase<Context>::comm_size_; \
  using CollectiveOpBase<Context>::comm_rank_; \
  using CollectiveOpBase<Context>::comm_root_; \
  using CollectiveOpBase<Context>::enable_nccl_

} // namespace dragon

#endif // USE_MPI

#endif // DRAGON_OPERATORS_DISTRIBUTED_COLLECTIVE_OP_BASE_H_
