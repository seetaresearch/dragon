/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_OPERATORS_MPI_MPI_OP_BASE_H_
#define DRAGON_OPERATORS_MPI_MPI_OP_BASE_H_

#ifdef WITH_MPI

#include <mpi.h>

#include "core/operator.h"

namespace dragon {

template <class Context>
class MPIOpBase : public Operator<Context> {
 public:
    MPIOpBase(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          comm_((MPI_Comm)OpArg<int64_t>("comm", 0)),
          group_((MPI_Group)OpArg<int64_t>("group", 0)) {

        if (comm_ == MPI_COMM_NULL) return;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);
        MPI_Comm_size(comm_, &comm_size_);
        MPI_Comm_rank(comm_, &comm_rank_);

        MPI_Group world_group;
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);
        int world_root = OpArg<int64_t>("root", 0);
        MPI_Group_translate_ranks(
            world_group, 1, &world_root,
            group_, &comm_root_
        );

        CHECK(comm_root_ != MPI_UNDEFINED)
            << "\nRoot is not included in the group.";
    }

    template <typename T>
    MPI_Datatype mpi_dtype() {
        auto type_str = TypeMetaToString(TypeMeta::Make<T>());
        if (type_str == "uint8") return MPI_BYTE;
        else if (type_str == "int8") return MPI_CHAR;
        else if (type_str == "uint8") return MPI_UNSIGNED_CHAR;
        else if (type_str == "int32") return MPI_INT;
        else if (type_str == "int64") return MPI_LONG_LONG;
        else if (type_str == "float16") return MPI_UNSIGNED_SHORT;
        else if (type_str == "float32") return MPI_FLOAT;
        else if (type_str == "float64") return MPI_DOUBLE;
        return MPI_DATATYPE_NULL;
    }

    template <typename T>
    void Recv(T* buf, int count, int from) {
        MPI_Recv(
            buf, count,
            mpi_dtype<T>(),
            from, 0, comm_,
            MPI_STATUS_IGNORE
        );
    }

    template <typename T>
    void IRecv(
        T*                  buf,
        int                 count,
        int                 from,
        MPI_Request*        req) {
        MPI_Irecv(
            buf, count,
            mpi_dtype<T>(),
            from, 0, comm_, req
        );
    }

    template <typename T>
    void Send(const T* buf, int count, int to) {
        MPI_Send(
            buf, count,
            mpi_dtype<T>(),
            to, 0, comm_
        );
    }

    template <typename T>
    void SendRecv(
        const T*            send_buf,
        int                 send_count,
        int                 to,
        T*                  recv_buf,
        int                 recv_count,
        int                 from) {
        MPI_Sendrecv(
            send_buf,
            send_count,
            mpi_dtype<T>(), to, 0,
            recv_buf, recv_count,
            mpi_dtype<T>(), from, 0,
            comm_,
            MPI_STATUS_IGNORE
        );
    }

    template <typename T>
    void BCast(T* buf, int count) {
        MPI_Bcast(
            buf, count,
            mpi_dtype<T>(),
            comm_root_, comm_
        );
    }

 public:
    MPI_Comm comm_;
    MPI_Group group_;
    int world_size_, world_rank_;
    int comm_size_, comm_rank_, comm_root_;
};

#define USE_MPI_FUNCTIONS \
    using MPIOpBase<Context>::Recv; \
    using MPIOpBase<Context>::IRecv; \
    using MPIOpBase<Context>::Send; \
    using MPIOpBase<Context>::SendRecv; \
    using MPIOpBase<Context>::BCast; \
    using MPIOpBase<Context>::comm_; \
    using MPIOpBase<Context>::comm_size_; \
    using MPIOpBase<Context>::comm_rank_; \
    using MPIOpBase<Context>::comm_root_;

}  // namespace dragon

#endif  // WITH_MPI

#endif  // DRAGON_OPERATORS_MPI_MPI_OP_BASE_H_