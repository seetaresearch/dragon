// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_UPDATE_COLLECTIVE_UPDATE_OP_H_
#define DRAGON_OPERATORS_UPDATE_COLLECTIVE_UPDATE_OP_H_

#include "core/operator.h"

namespace dragon {

#ifdef WITH_MPI

template <class Context>
class CollectiveUpdateOp : public Operator<Context> {
 public:
     CollectiveUpdateOp(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          mode(OperatorBase::GetSingleArg<string>("mode", "UNKNOWN")) {
         InitMPI();
         if (mode.find("NCCL") != string::npos) InitNCCL();
     }

    void InitMPI();
    void InitNCCL();

    void RunOnDevice() override;
    void MPIAllReduceWithFloat();
    void NCCLAllReduceWithFloat();
    void MPIBcastWithFloat();
    void NCCLBcastWithFloat();

 protected:
    int comm_size, comm_rank, comm_root;
    int world_size, world_rank;
    Tensor* buffer;
    string  mode;

    MPI_Comm comm;
    MPI_Group group;

#ifdef WITH_MPI_NCCL
    ncclComm_t nccl_comm;
    cudaStream_t stream;
#endif
};

#endif    // WITH_MPI

}    // namespace dragon

#endif    // DRAGON_OPERATORS_UPDATE_COLLECTIVE_UPDATE_OP_H_