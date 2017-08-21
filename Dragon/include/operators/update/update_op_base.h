// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_UPDATE_UPDATE_OP_BASE_H_
#define DRAGON_OPERATORS_UPDATE_UPDATE_OP_BASE_H_

#include "core/operator.h"

namespace dragon {

template <class Context>
class UpdateOpBase : public Operator<Context> {
 public:
    UpdateOpBase(const OperatorDef& op_def, Workspace* ws) 
        : Operator<Context>(op_def, ws),
          allow_parallel(false),
          async_tag(-1),
          lr_mult(OperatorBase::GetSingleArg<float>("lr_mult", 1.0)),
          decay_mult(OperatorBase::GetSingleArg<float>("decay_mult", 1.0)),
          domain(OperatorBase::GetSingleArg<string>("domain", "_")), 
          mode(OperatorBase::GetSingleArg<string>("mode", "Sync")) { InitMPI(); }

    float param(const string& name) const;
    void InitMPI();

    void RunOnDevice() override;
    template <typename T> void ReduceRunWithType();
    template <typename T> void PreprocessRunWithType();
    virtual void ComputeRunWithFloat() = 0;
    template <typename T> void UpdateRunWithType();
    template <typename T> void RecvRunWithType();

 protected:
    float lr_mult, decay_mult;
    float l2_decay, clip_thresh, scale_factor;
    int comm_size, comm_rank, comm_root;
    int world_size, world_rank;
    bool allow_parallel;
    int async_tag;
    Tensor* buffer;
    string domain, mode;

#ifdef WITH_MPI
    MPI_Comm comm;
    MPI_Group group;
#endif  // WITH_MPI

#ifdef WITH_MPI_NCCL
    ncclComm_t nccl_comm;
    cudaStream_t stream;
#endif  // WITH_MPI_NCCL

};

}    // namespace dragon 

#endif    // DRAGON_OPERATORS_UPDATE_UPDATE_OP_BASE_H_