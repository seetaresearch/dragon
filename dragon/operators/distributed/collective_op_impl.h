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

#ifndef DRAGON_OPERATORS_DISTRIBUTED_COLLECTIVE_OP_IMPL_H_
#define DRAGON_OPERATORS_DISTRIBUTED_COLLECTIVE_OP_IMPL_H_

#include "dragon/operators/distributed/collective_op_impl_cncl.h"
#include "dragon/operators/distributed/collective_op_impl_mpi.h"
#include "dragon/operators/distributed/collective_op_impl_nccl.h"

namespace dragon {

class CollectiveOpImpl {
 public:
  explicit CollectiveOpImpl(const string& backend = "MPI")
      : backend_(backend) {}

  void SetBackend(const string& backend) {
    backend_ = backend;
  }

  void SetComm(
      const int64_t mpi_comm,
      const int64_t mpi_group,
      const vec64_t ranks,
      const int64_t root = 0) {
    mpi_coll_.SetComm(mpi_comm, mpi_group, ranks, root);
    nccl_coll_.SetComm(mpi_comm, mpi_group, ranks, root);
    cncl_coll_.SetComm(mpi_comm, mpi_group, ranks, root);
  }

  template <class Context>
  const bool PreferNCCL(cudaStream_t& stream, Context* ctx) const {
    stream = nullptr;
#if defined(USE_CUDA) && defined(USE_NCCL)
    if (backend_ == "NCCL" &&
        TypeMeta::Id<Context>() == TypeMeta::Id<CUDAContext>()) {
      stream = reinterpret_cast<CUDAContext*>(ctx)->cuda_stream();
      return true;
    }
#endif
    return false;
  }

  template <class Context>
  const bool PreferCNCL(cnrtQueue_t& stream, Context* ctx) const {
    stream = nullptr;
#ifdef USE_MLU
    if (backend_ == "CNCL" &&
        TypeMeta::Id<Context>() == TypeMeta::Id<MLUContext>()) {
      stream = reinterpret_cast<MLUContext*>(ctx)->mlu_stream();
      return true;
    }
#endif
    return false;
  }

  template <typename T>
  void Bcast(T* buf, int count) {
    mpi_coll_.Bcast(buf, count);
  }

  template <typename T, class Context>
  void Bcast(T* buf, int count, Context* ctx) {
    cudaStream_t cuda_stream;
    cnrtQueue_t mlu_stream;
    if (PreferNCCL(cuda_stream, ctx)) {
      nccl_coll_.Bcast(buf, count, cuda_stream);
    } else if (PreferCNCL(mlu_stream, ctx)) {
      cncl_coll_.Bcast(buf, count, mlu_stream);
    } else {
      mpi_coll_.Bcast(buf, count);
    }
  }

  template <typename T>
  void AllReduce(const T* sendbuf, T* recvbuf, int count) {
    mpi_coll_.AllReduce(sendbuf, recvbuf, count);
  }

  template <typename T, class Context>
  void AllReduce(const T* sendbuf, T* recvbuf, int count, Context* ctx) {
    cudaStream_t cuda_stream;
    cnrtQueue_t mlu_stream;
    if (PreferNCCL(cuda_stream, ctx)) {
      nccl_coll_.AllReduce(sendbuf, recvbuf, count, cuda_stream);
    } else if (PreferCNCL(mlu_stream, ctx)) {
      cncl_coll_.AllReduce(sendbuf, recvbuf, count, mlu_stream);
    } else {
      mpi_coll_.AllReduce(sendbuf, recvbuf, count);
    }
  }

  template <typename T>
  void AllGather(const T* sendbuf, T* recvbuf, int sendcount) {
    mpi_coll_.AllGather(sendbuf, recvbuf, sendcount);
  }

  template <typename T, class Context>
  void AllGather(const T* sendbuf, T* recvbuf, int sendcount, Context* ctx) {
    cudaStream_t cuda_stream;
    cnrtQueue_t mlu_stream;
    if (PreferNCCL(cuda_stream, ctx)) {
      nccl_coll_.AllGather(sendbuf, recvbuf, sendcount, cuda_stream);
    } else if (PreferCNCL(mlu_stream, ctx)) {
      cncl_coll_.AllGather(sendbuf, recvbuf, sendcount, mlu_stream);
    } else {
      mpi_coll_.AllGather(sendbuf, recvbuf, sendcount);
    }
  }

  template <typename T>
  void ReduceScatter(const T* sendbuf, T* recvbuf, int recvcount) {
    mpi_coll_.ReduceScatter(sendbuf, recvbuf, recvcount);
  }

  template <typename T, class Context>
  void
  ReduceScatter(const T* sendbuf, T* recvbuf, int recvcount, Context* ctx) {
    cudaStream_t cuda_stream;
    cnrtQueue_t mlu_stream;
    if (PreferNCCL(cuda_stream, ctx)) {
      nccl_coll_.ReduceScatter(sendbuf, recvbuf, recvcount, cuda_stream);
    } else if (PreferCNCL(mlu_stream, ctx)) {
      cncl_coll_.ReduceScatter(sendbuf, recvbuf, recvcount, mlu_stream);
    } else {
      mpi_coll_.ReduceScatter(sendbuf, recvbuf, recvcount);
    }
  }

  const int comm_size() const {
    return mpi_coll_.comm_size();
  }

 private:
  string backend_;
  MPICollectiveOpImpl mpi_coll_;
  NCCLCollectiveOpImpl nccl_coll_;
  CNCLCollectiveOpImpl cncl_coll_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_DISTRIBUTED_COLLECTIVE_OP_IMPL_H_
