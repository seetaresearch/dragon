// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_MPI_MPI_BROADCAST_OP_H_
#define DRAGON_OPERATORS_MPI_MPI_BROADCAST_OP_H_

#ifdef WITH_MPI

#include "operators/mpi/base_mpi_op.h"

namespace dragon {

template <class Context>
class MPIBroadcastOp final : public ModelMPIBase<Context> {
 public:
    MPIBroadcastOp(const OperatorDef& op_def, Workspace* ws)
        : ModelMPIBase<Context>(op_def, ws) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

template <class Context>
class MPIBroadcastGradientOp final : public ModelMPIBase<Context> {
public:
    MPIBroadcastGradientOp(const OperatorDef& op_def, Workspace* ws)
        : ModelMPIBase<Context>(op_def, ws) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

}    // namespace dragon

#endif // WITH_MPI

#endif    //DRAGON_OPERATORS_MPI_MPI_BROADCAST_OP_H_



