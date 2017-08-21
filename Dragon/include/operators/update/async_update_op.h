// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_UPDATE_ASYNC_UPDATE_OP_H_
#define DRAGON_OPERATORS_UPDATE_ASYNC_UPDATE_OP_H_

#ifdef WITH_MPI

#include "operators/update/update_op_base.h"
#include "utils/thread.h"

namespace dragon {

template <class Context>
class AsyncUpdateOp final: public UpdateOpBase<Context> {
 public:
    AsyncUpdateOp(const OperatorDef& op_def, Workspace* ws);

    int GetDelay(int tag);
    void UpdateTimestamp(int tag);

    void RunOnDevice() override;
    void ComputeRunWithFloat() override { /* do nothing */ }
    template <typename T> void RootRunWithType();
    template <typename T> void ThreadRunWithType();

 protected:
    string mode;
    unique_ptr<Tensor> recv_buffer;
    Tensor** acc_buffers;
    string* tags;
    TIndex update_count;
    int node_id, nsync, max_recv;
    Map<int, int> local_timestamp;
    std::unique_ptr<std::thread> thread;

#ifdef WITH_MPI_CUDA
    cudaStream_t stream;
    cublasHandle_t handle;
#endif

};

}    // namespace dragon

#endif    // WITH_MPI

#endif    // DRAGON_OPERATORS_UPDATE_ASYNC_UPDATE_OP_H_