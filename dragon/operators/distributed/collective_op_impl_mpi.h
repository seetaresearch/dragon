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

#ifndef DRAGON_OPERATORS_DISTRIBUTED_COLLECTIVE_OP_IMPL_MPI_H_
#define DRAGON_OPERATORS_DISTRIBUTED_COLLECTIVE_OP_IMPL_MPI_H_

#include "dragon/core/common.h"
#include "dragon/utils/device/common_mpi.h"

namespace dragon {

class DRAGON_API MPICollectiveOpImpl {
 public:
  MPICollectiveOpImpl()
      : comm_size_(0), comm_rank_(0), comm_root_(0), reduction_("SUM") {}

  void SetComm(
      const int64_t mpi_comm,
      const int64_t mpi_group,
      const vec64_t ranks,
      const int64_t root = 0);

  void SetReduction(const string& reduction) {
    reduction_ = reduction;
  }

  template <typename T>
  void Send(const T* x, int N, int peer);

  template <typename T>
  void Recv(T* y, int N, int peer);

  template <typename T>
  void Broadcast(const T* x, T* y, int N);

  template <typename T>
  void AllReduce(const T* x, T* y, int N);

  template <typename T>
  void ReduceScatter(const T* x, T* y, int N);

  template <typename T>
  void AllGather(const T* x, T* y, int N);

  template <typename T>
  MPI_Datatype data_type();

  MPI_Op reduction();

  const int comm_rank() const {
    return comm_rank_;
  }

  const int comm_size() const {
    return comm_size_;
  }

 private:
  int comm_size_;
  int comm_rank_;
  int comm_root_;
  MPI_Comm mpi_comm_;
  MPI_Group mpi_group_;
  string reduction_;
};

} // namespace dragon

#endif // DRAGON_OPERATORS_DISTRIBUTED_COLLECTIVE_OP_IMPL_MPI_H_
