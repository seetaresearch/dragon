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

#ifndef DRAGON_OPERATORS_DISTRIBUTED_COLLECTIVE_OP_IMPL_CNCL_H_
#define DRAGON_OPERATORS_DISTRIBUTED_COLLECTIVE_OP_IMPL_CNCL_H_

#include "dragon/utils/device/common_mlu.h"
#include "dragon/utils/device/common_mpi.h"

namespace dragon {

class DRAGON_API CNCLCollectiveOpImpl {
 public:
  CNCLCollectiveOpImpl() : comm_size_(0), comm_rank_(0), comm_root_(0) {}

  void SetComm(
      const int64_t mpi_comm,
      const int64_t mpi_group,
      const vec64_t ranks,
      const int64_t root = 0);

  template <typename T>
  void Bcast(T* buf, int count, cnrtQueue_t stream);

  template <typename T>
  void AllReduce(const T* sendbuf, T* recvbuf, int count, cnrtQueue_t stream);

  template <typename T>
  void
  AllGather(const T* sendbuf, T* recvbuf, int sendcount, cnrtQueue_t stream);

  template <typename T>
  void ReduceScatter(
      const T* sendbuf,
      T* recvbuf,
      int recvcount,
      cnrtQueue_t stream);

  template <typename T>
  cnclDataType_t data_type();

  cnclComm_t cncl_comm();

  const int comm_size() const {
    return comm_size_;
  }

 private:
  int comm_size_;
  int comm_rank_;
  int comm_root_;
  MPI_Comm mpi_comm_;
  MPI_Group mpi_group_;
  string group_str_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_DISTRIBUTED_COLLECTIVE_OP_IMPL_CNCL_H_
