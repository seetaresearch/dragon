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

#ifndef DRAGON_OPERATORS_MPI_MPI_BROADCAST_OP_H_
#define DRAGON_OPERATORS_MPI_MPI_BROADCAST_OP_H_

#ifdef WITH_MPI

#include "operators/mpi/mpi_op_base.h"

namespace dragon {

template <class Context>
class MPIBroadcastOp final : public MPIOpBase<Context> {
 public:
    MPIBroadcastOp(const OperatorDef& def, Workspace* ws)
        : MPIOpBase<Context>(def, ws) {}
    USE_OPERATOR_FUNCTIONS;
    USE_MPI_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();
};

template <class Context>
class MPIBroadcastGradientOp final : public MPIOpBase<Context> {
public:
    MPIBroadcastGradientOp(const OperatorDef& def, Workspace* ws)
        : MPIOpBase<Context>(def, ws) {}
    USE_OPERATOR_FUNCTIONS;
    USE_MPI_FUNCTIONS;

    void RunOnDevice() override;
    template <typename T> void RunImpl();
};

}  // namespace dragon

#endif  // WITH_MPI

#endif  //DRAGON_OPERATORS_MPI_MPI_BROADCAST_OP_H_