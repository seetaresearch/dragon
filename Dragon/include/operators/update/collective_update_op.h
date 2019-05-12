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

#include "operators/mpi/mpi_op_base.h"

namespace dragon {

#ifdef WITH_MPI

template <class Context>
class CollectiveUpdateOp final
    : public MPIOpBase<Context> {
 public:
    CollectiveUpdateOp(
        const OperatorDef&      def,
        Workspace*              ws)
        : MPIOpBase<Context>(def, ws),
          mode_(OpArg<string>("mode", "")) {
        if (mode_.find("NCCL") != string::npos) InitNCCL();
    }
    USE_OPERATOR_FUNCTIONS;
    USE_MPI_FUNCTIONS;

    ~CollectiveUpdateOp() {
        /*  TODO(PhyscalX): Temporarily disable it,
                            to avoid a unhandled error. */
#ifdef WITH_NCCL
        if (mode_.find("NCCL") != string::npos) {
            /* ncclCommDestroy(nccl_comm); */
        }
#endif
    }

    void InitNCCL();

    void RunOnDevice() override;
    template <typename T> void MPIBCast(Tensor*);
    template <typename T> void MPIAllReduce(Tensor*);

#ifdef WITH_NCCL
    template <typename T>
    void NCCLAllReduce(Tensor*, ncclDataType_t);

    template <typename T>
    void NCCLBcast(Tensor*, ncclDataType_t);
#endif

 protected:
    string mode_;

#ifdef WITH_NCCL
    ncclComm_t nccl_comm;
#endif
};

#endif  // WITH_MPI

}  // namespace dragon

#endif  // DRAGON_OPERATORS_UPDATE_COLLECTIVE_UPDATE_OP_H_