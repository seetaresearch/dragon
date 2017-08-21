#include "operators/update/update_op_base.h"
#include "core/workspace.h"
#include "utils/math_functions.h"

namespace dragon {

template <class Context>
float UpdateOpBase<Context>::param(const string& name) const {
    return ws()->GetTensor(domain + name)
               ->template mutable_data<float, CPUContext>()[0];
}

template <class Context>
void UpdateOpBase<Context>::InitMPI() {
    if (this->args().count("comm") &&
        this->args().count("group") &&
        this->args().count("root")) {
#ifdef WITH_MPI
        comm = (MPI_Comm)OperatorBase::GetSingleArg<int64_t>("comm", 0);
        group = (MPI_Group)OperatorBase::GetSingleArg<int64_t>("group", 0);
        int world_root = OperatorBase::GetSingleArg<int>("root", 0);
        if (comm == MPI_COMM_NULL) return;
        allow_parallel = true;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(comm, &comm_size);
        MPI_Comm_rank(comm, &comm_rank);
        MPI_Group world_group;
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);
        MPI_Group_translate_ranks(world_group, 1, &world_root, group, &comm_root);
        CHECK(comm_root != MPI_UNDEFINED) << "MPI root is not included in layer group.";
#endif  // WITH_MPI

#ifdef WITH_MPI_NCCL
        ncclUniqueId id;
        if (comm_rank == comm_root) ncclGetUniqueId(&id);
        MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, comm);
        ctx().SwitchToDevice();
        NCCL_CHECK(ncclCommInitRank(&nccl_comm, comm_size, id, comm_rank));
        CUDA_CHECK(cudaStreamCreate(&stream));
#endif  // WITH_MPI_NCCL
    }
}

template <class Context> template <typename T>
void UpdateOpBase<Context>::ReduceRunWithType() {
    if (TypeMeta::Id<Context>() == TypeMeta::Id<CUDAContext>()) {
#ifdef WITH_MPI_NCCL
        TIndex count = input(0).count();
        auto* dXdata = input(0).template mutable_data<T, Context>();
        NCCL_CHECK(ncclAllReduce((const void*)dXdata,
                                       (void*)dXdata,
                                               count,
                                           ncclFloat,
                                             ncclSum,
                                           nccl_comm,
                                            stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        math::Scal<T, Context>(count, T(1.0 / comm_size), dXdata);
        return;
#endif
    }
#ifdef WITH_MPI  // WITH_MPI
    MPI_Request recv_req;
    TIndex count = input(0).count();
    TIndex segment_size = count / comm_size;
    TIndex residual = count % comm_size;
    vector<TIndex> segment_sizes(comm_size, segment_size);
    for (int i = 0; i < residual; i++) segment_sizes[i]++;
    vector<TIndex> segment_ends(comm_size);
    segment_ends[0] = segment_sizes[0];
    for (int i = 1; i < segment_ends.size(); i++) 
        segment_ends[i] = segment_sizes[i] + segment_ends[i - 1];
    buffer = ws()->GetBuffer();
    buffer->Reshape(vector<TIndex>(1, segment_sizes[0]));
#ifdef WITH_MPI_CUDA
    auto* Bdata = buffer->mutable_data<T, Context>();
    auto* dXdata = input(0).template mutable_data<T, Context>();
#else
    auto* Bdata = buffer->mutable_data<T, CPUContext>();
    auto* dXdata = input(0).template mutable_data<T, CPUContext>();
#endif // WITH_MPI_CUDA

    int recv_from = (comm_rank - 1 + comm_size) % comm_size;
    int send_to = (comm_rank + 1) % comm_size;

    //  scatter-reduce
    for (int i = 0; i < comm_size - 1; i++) {
        int recv_chunk = (comm_rank - i - 1 + comm_size) % comm_size;
        int send_chunk = (comm_rank - i + comm_size) % comm_size;
        auto* segment_send = &(dXdata[segment_ends[send_chunk] -
            segment_sizes[send_chunk]]);
        MPI_Irecv(Bdata, segment_sizes[recv_chunk],
            MPI_FLOAT, recv_from, 0, comm, &recv_req);
        MPI_Send(segment_send, segment_sizes[send_chunk],
            MPI_FLOAT, send_to, 0, comm);
        auto* segment_update = &(dXdata[segment_ends[recv_chunk] -
            segment_sizes[recv_chunk]]);
        MPI_Wait(&recv_req, MPI_STATUS_IGNORE);
#ifdef WITH_MPI_CUDA
        math::Axpy<T, Context>(segment_sizes[recv_chunk],
            1.0, Bdata, segment_update);
        cudaStreamSynchronize(cudaStreamDefault);
#else 
        math::Axpy<T, CPUContext>(segment_sizes[recv_chunk], 
            1.0, Bdata, segment_update);
#endif // WITH_MPI_CUDA
    }
    ws()->ReleaseBuffer(buffer);

    //  allgather
    for (int i = 0; i < comm_size - 1; i++) {
        int send_chunk = (comm_rank - i + 1 + comm_size) % comm_size;
        int recv_chunk = (comm_rank - i + comm_size) % comm_size;
        auto* segment_send = &(dXdata[segment_ends[send_chunk] - 
                                      segment_sizes[send_chunk]]);
        auto* segment_recv = &(dXdata[segment_ends[recv_chunk] -
                                      segment_sizes[recv_chunk]]);
        MPI_Sendrecv(segment_send, segment_sizes[send_chunk],
                     MPI_FLOAT, send_to, 0, segment_recv,
                     segment_sizes[recv_chunk], MPI_FLOAT, recv_from,
                     0, comm, MPI_STATUS_IGNORE);
    }

    //  ave-normalize
    if (comm_size > 1) {
#ifdef WITH_MPI_CUDA
        math::Scal<T, Context>(count, T(1.0 / comm_size), dXdata);
#else
        math::Scal<T, CPUContext>(count, T(1.0 / comm_size), dXdata);
#endif // WITH_MPI_CUDA
    }
#endif // WITH_MPI
}

template <class Context> template <typename T>
void UpdateOpBase<Context>::PreprocessRunWithType() {
    //  scale
    scale_factor = param("scale_gradient");
    if (scale_factor != 1) {
        auto* dXdata = input(0).template mutable_data<T, Context>();
        math::Scal<T, Context>(input(0).count(), scale_factor, dXdata);
    }
    //  clip
    clip_thresh = param("clip_gradient");
    if (clip_thresh > 0) {
        auto* dXdata = input(0).template mutable_data<T, Context>();
        T sumsq_grad = math::Dot<T, Context>(input(0).count(), dXdata, dXdata);
        const T l2norm = sqrt(sumsq_grad);
        if (l2norm > clip_thresh) {
            T factor = clip_thresh / l2norm;
            math::Scal<T, Context>(input(0).count(), factor, dXdata);
        }
    }
    //  decay
    l2_decay = param("l2_decay") * decay_mult;
    if (l2_decay > 0) {
        auto* dXdata = input(0).template mutable_data<T, Context>();
        auto* Xdata = output(0)->template data<T, Context>();
        math::Axpy<T, Context>(input(0).count(), l2_decay, Xdata, dXdata);
    }
}

template <class Context> template <typename T>
void UpdateOpBase<Context>::UpdateRunWithType() {
    if (!allow_parallel || (allow_parallel && mode == "Sync")) {
        auto* dXdata = input(0).template mutable_data<T, Context>();
        auto* Xdata = output(0)->template mutable_data<T, Context>();
        //  update
        math::Axpy<T, Context>(output(0)->count(), -1.0, dXdata, Xdata);
        //  clear accumulated grads
        math::Set<T, Context>(input(0).count(), 0, dXdata);
    } else {
#ifdef WITH_MPI
        if (comm_rank == comm_root) return;
        if (async_tag == -1) {
            Tensor* t = ws()->GetTensor("_t_" + domain + "async_tags");
            auto* tags = t->template data<string, CPUContext>();
            for (int i = 0; i < t->count(); i++) {
                if (output(0)->name() == tags[i]) {
                    async_tag = i;
                    break;
                }
            }
            CHECK(async_tag != -1);
        }
#ifdef WITH_MPI_CUDA
        auto* dXdata = input(0).template mutable_data<T, Context>();
#else
        auto* dXdata = input(0).template mutable_data<T, CPUContext>();
#endif // WITH_MPI_CUDA
        MPI_Send(dXdata, input(0).count(), MPI_FLOAT, this->comm_root, async_tag, this->comm);
#endif // WITH_MPI
    }
}

template <class Context> template <typename T>
void UpdateOpBase<Context>::RecvRunWithType() {
#ifdef WITH_MPI
    if (comm_rank != comm_root) {
#ifdef WITH_MPI_CUDA
        auto* Xdata = output(0)->template mutable_data<T, Context>();
#else
        auto* Xdata = output(0)->template mutable_data<T, CPUContext>();
        
#endif // WITH_MPI_CUDA
        MPI_Recv(Xdata, output(0)->count(), MPI_FLOAT, 
            this->comm_root, async_tag, this->comm, MPI_STATUS_IGNORE);
    }
#endif // WITH_MPI
}

template <class Context>
void UpdateOpBase<Context>::RunOnDevice() {
    CHECK(input(0).dims() == output(0)->dims())
        << "\nTensor and its gradient must have same dims if update.";
    if (input(0).count() == 0 || output(0)->count() == 0) return;

    if (input(0).template IsType<float>()) {
        if (allow_parallel && mode == "Sync") ReduceRunWithType<float>();
        PreprocessRunWithType<float>();
        ComputeRunWithFloat();
        UpdateRunWithType<float>();
        if (allow_parallel && mode != "Sync") RecvRunWithType<float>();
    } else {
        LOG(FATAL) << "unsupported input types.";
    }
} 

template class UpdateOpBase<CPUContext>;
#ifdef WITH_CUDA
template class UpdateOpBase<CUDAContext>;
#endif

}    // namespace dragon