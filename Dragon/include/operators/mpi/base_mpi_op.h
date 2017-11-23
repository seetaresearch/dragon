// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_OPERATORS_MPI_BASE_MPI_OP_H_
#define DRAGON_OPERATORS_MPI_BASE_MPI_OP_H_

#ifdef WITH_MPI

#include "core/operator.h"
#include "mpi/mpi.h"

namespace dragon {

template <class Context>
class ModelMPIBase : public Operator<Context> {
 public:
    ModelMPIBase(const OperatorDef& op_def, Workspace* ws)
        : Operator<Context>(op_def, ws),
          comm((MPI_Comm)OperatorBase::GetSingleArg<int64_t>("comm", 0)),
          group((MPI_Group)OperatorBase::GetSingleArg<int64_t>("group", 0)),
          dtype(OperatorBase::GetSingleArg<string>("dtype", "FLOAT32")) {

        if (comm == MPI_COMM_NULL) return;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(comm, &comm_size);
        MPI_Comm_rank(comm, &comm_rank);

        MPI_Group world_group;
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);
        int world_root = OperatorBase::GetSingleArg<int>("root", 0);
        MPI_Group_translate_ranks(world_group, 1, &world_root, group, &comm_root);

        CHECK(comm_root != MPI_UNDEFINED) << "MPI root is not included in layer group.";
    }

    MPI_Datatype mpi_dtype() {
        if (dtype == "FLOAT32") return MPI_FLOAT;
        else LOG(FATAL) << "Unsupported input type: " << dtype;
        return MPI_DATATYPE_NULL;
    }

 protected:
    MPI_Comm comm;
    MPI_Group group;
    int comm_size, comm_rank, comm_root;
    int world_size, world_rank;
    string dtype;
};

}    // namespace dragon

#endif // WITH_MPI

#endif // DRAGON_OPERATORS_MPI_BASE_MPI_OP_H_