#include "operators/update/async_update_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"

#ifdef WITH_MPI

#ifdef WITH_CUDA_AWARE
#include <cublas_v2.h>
#endif // WITH_CUDA_AWARE

namespace dragon {

template <class Context>
void AsyncUpdateOp<Context>::UpdateTimestamp(int tag) {
    Tensor* t = ws()->GetTensor("_t_" + this->domain + "async_timestamp");
    int* global_timestamp = t->template mutable_data<int, CPUContext>() ;
    ws()->LockTensor(t->name());
    global_timestamp[tag]++;
    local_timestamp[tag] = global_timestamp[tag];
    ws()->UnlockTensor(t->name());
}

template <class Context>
int AsyncUpdateOp<Context>::GetDelay(int tag){
    Tensor* t = ws()->GetTensor("_t_" + this->domain + "async_timestamp");
    int* global_timestamp = t->template mutable_data<int, CPUContext>();
    return global_timestamp[tag] - local_timestamp[tag] + 1;
}

template <class Context>
AsyncUpdateOp<Context>::AsyncUpdateOp(const OperatorDef& op_def, Workspace* ws)
    : UpdateOpBase<Context>(op_def, ws),
      update_count(0),
      node_id(OperatorBase::GetSingleArg<int>("node_id", 0)),
      mode(OperatorBase::GetSingleArg<string>("mode", "Async")),
      nsync(OperatorBase::GetSingleArg<int>("nsync", -1)) {

    //    make key-val tags
    Tensor* t = this->ws()->CreateTensor("_t_" + this->domain + "async_tags");
    t->Reshape(vector<TIndex>(1, InputSize()));
    tags = t->template mutable_data<string, CPUContext>();
    for (int i = 0; i < OutputSize(); i++) tags[i] = output(i)->name();

    //    make recv logs
    t = this->ws()->CreateTensor("_t_" + this->domain + "async_logs");
    t->Reshape(vector<TIndex>(1, InputSize()));

    //    make recv buffers
    acc_buffers = new Tensor*[InputSize()];      // for soft-sync
    recv_buffer.reset(new Tensor());    //    for async

    //    setup for server
    if (this->comm_rank == this->comm_root) {
        if (nsync == -1) nsync = this->comm_size; // fully async
        max_recv = this->comm_size / nsync;
        //    make global timestamp
        t = this->ws()->CreateTensor("_t_" + this->domain + "async_timestamp");
        t->Reshape(vector<TIndex>(1, InputSize()));
        //    make global buffers
        for (int i = 0; i < OutputSize(); i++)
            acc_buffers[i] = this->ws()->CreateTensor(tags[i] + "_grad_async_acc");
    }

    //    create independent stream for thread if using cuda-aware
#ifdef WITH_CUDA_AWARE
    cudaStreamCreate(&stream);
    cublasCreate_v2(&handle);
    cublasSetStream(handle, stream);
#endif
}

template <class Context> template <typename T>
void AsyncUpdateOp<Context>::RootRunWithType() {
    for (int i = 0; i < InputSize(); i++){
        auto* dXdata = input(i).template mutable_data<T, Context>();
        auto* Xdata = output(i)->template mutable_data<T, Context>();

        if (mode != "Async_No_Lock") ws()->LockTensor(output(i)->name());
        int delay = GetDelay(i); UpdateTimestamp(i);
        math::Axpy<T, Context>(input(i).count(), -1.0 / delay, dXdata, Xdata);
#ifdef WITH_CUDA_AWARE
        cudaStreamSynchronize(cudaStreamDefault);
#endif
        if (mode != "Async_No_Lock") ws()->UnlockTensor(output(i)->name());

        math::Set<T, Context>(input(i).count(), 0, dXdata);
    }
}

template <class Context>
void AsyncUpdateOp<Context>::RunOnDevice(){
    if (this->comm_rank != this->comm_root) return;

    if (input(0).template IsType<float>()) {
        if (node_id != this->comm_root && !thread) 
            thread = std::unique_ptr<std::thread>(
                new std::thread(std::bind(&AsyncUpdateOp::ThreadRunWithType<float>, this)));
        if (node_id == this->comm_root) RootRunWithType<float>(); 
    } else LOG(FATAL) << "unsupported input types.";
}

template <class Context> template <typename T>
void AsyncUpdateOp<Context>::ThreadRunWithType() {
    while (1) {
        //    pull from specfic client
        MPI_Status status;
        MPI_Probe(node_id, MPI_ANY_TAG, this->comm, &status);
        Tensor* X = ws()->GetTensor(tags[status.MPI_TAG]);
        if (X->count() == 0) continue; //    wait for server 
        recv_buffer->ReshapeLike(*X);
#ifdef WITH_CUDA_AWARE
        auto* Bdata = recv_buffer->template mutable_data<T, Context>();
#else 
        auto* Bdata = recv_buffer->template mutable_data<T, CPUContext>();
#endif
        MPI_Recv(Bdata, X->count(), MPI_FLOAT, status.MPI_SOURCE, status.MPI_TAG, this->comm, MPI_STATUS_IGNORE);
        //    update
#ifdef WITH_CUDA_AWARE
        auto* Xdata = X->template mutable_data<T, Context>();
        if (mode != "Async_No_Lock") ws()->LockTensor(output(status.MPI_TAG)->name());
        int delay = GetDelay(status.MPI_TAG);
        UpdateTimestamp(status.MPI_TAG);
        float alpha = - 1.0 / delay;
        cublasSaxpy_v2(handle, X->count(), &alpha, Bdata, 1, Xdata, 1);
        cudaStreamSynchronize(stream);
        if (mode != "Async_No_Lock") ws()->UnlockTensor(output(status.MPI_TAG)->name());
#else
        if (mode != "Async_No_Lock") ws()->LockTensor(output(status.MPI_TAG)->name());
        int delay = GetDelay(status.MPI_TAG);
        UpdateTimestamp(status.MPI_TAG);
        auto* Xdata = X->template mutable_data<T, CPUContext>();
        math::Axpy<T, CPUContext>(X->count(), -1.0 / delay, Bdata, Xdata);
        if (mode != "Async_No_Lock") ws()->UnlockTensor(output(status.MPI_TAG)->name());
#endif
        //    push back to this client
        MPI_Send(Xdata, X->count(), MPI_FLOAT, status.MPI_SOURCE, status.MPI_TAG, this->comm);
        //    do statistics
        update_count++;
        if (update_count % (100 * InputSize()) == 0)
            LOG(INFO) << "Server[" << node_id << "]: "
            << "update: " << update_count / InputSize() << " iters";
    }
}

DEPLOY_CPU(AsyncUpdate);
#ifdef WITH_CUDA
DEPLOY_CUDA(AsyncUpdate);
#endif

OPERATOR_SCHEMA(AsyncUpdate);

}    // namespace dragon

#endif // WITH_MPI
