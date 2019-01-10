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

#ifndef DRAGON_OPERATORS_UPDATE_COLLECTIVE_UPDATE_OP_H_
#define DRAGON_OPERATORS_UPDATE_COLLECTIVE_UPDATE_OP_H_

#include "core/operator.h"

namespace dragon {

#ifdef WITH_MPI

template <class Context>
class CollectiveUpdateOp final : public Operator<Context> {
 public:
    CollectiveUpdateOp(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          mode(OperatorBase::Arg<string>("mode", "UNKNOWN")) {
         InitMPI();
         if (mode.find("NCCL") != string::npos) InitNCCL();
    }
    USE_OPERATOR_FUNCTIONS;

    ~CollectiveUpdateOp() {
        /*  TODO(PhyscalX): Temporarily disable it,
                            to avoid a unhandled error. */
#ifdef WITH_NCCL
        if (mode.find("NCCL") != string::npos) {
            /* ncclCommDestroy(nccl_comm); */
        }
#endif
    }

    void InitMPI();
    void InitNCCL();

    void RunOnDevice() override;

    template <typename T> void MPIAllReduce(
        Tensor*                 tensor,
        MPI_Datatype            dtype);

    template <typename T> void MPIBcast(
        Tensor*                 tensor,
        MPI_Datatype            dtype);

#ifdef WITH_NCCL
    template <typename T> void NCCLAllReduce(
        Tensor*                 tensor,
        ncclDataType_t          dtype,
        cudaStream_t&           stream);

    template <typename T> void NCCLBcast(
        Tensor*                 tensor,
        ncclDataType_t          dtype,
        cudaStream_t&           stream);
#endif

 protected:
    int comm_size, comm_rank, comm_root;
    int world_size, world_rank;
    string mode;

    MPI_Comm comm;
    MPI_Group group;

#ifdef WITH_NCCL
    ncclComm_t nccl_comm;
    CUDAClosure<Context> closure;
#endif
};

#endif  // WITH_MPI

}  // namespace dragon

#endif  // DRAGON_OPERATORS_UPDATE_COLLECTIVE_UPDATE_OP_H_