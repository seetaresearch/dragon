// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_MPI_MPI_GATHER_OP_H_
#define DRAGON_OPERATORS_MPI_MPI_GATHER_OP_H_

#ifdef WITH_MPI

#include "operators/mpi/base_mpi_op.h"

namespace dragon {

template <class Context>
class MPIGatherOp final : public ModelMPIBase<Context> {
 public:
    MPIGatherOp(const OperatorDef& op_def, Workspace *ws)
        : ModelMPIBase<Context>(op_def, ws) {}

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

template <class Context>
class MPIGatherGradientOp final : public ModelMPIBase<Context> {
 public:
    MPIGatherGradientOp(const OperatorDef& op_def, Workspace *ws) 
        : ModelMPIBase<Context>(op_def, ws) {
        DISABLE_SHARE_GRADIENT;
    }

    void RunOnDevice() override;
    template <typename T> void RunWithType();
};

}    // namespace dragon

#endif // WITH_MPI

#endif    // DRAGON_OPERATORS_MPI_MPI_GATHER_OP_H_