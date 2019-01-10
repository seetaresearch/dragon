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

#ifndef DRAGON_OPERATORS_MPI_BASE_MPI_OP_H_
#define DRAGON_OPERATORS_MPI_BASE_MPI_OP_H_

#ifdef WITH_MPI

#include <mpi.h>

#include "core/operator.h"

namespace dragon {

template <class Context>
class ModelMPIBase : public Operator<Context> {
 public:
    ModelMPIBase(const OperatorDef& def, Workspace* ws)
        : Operator<Context>(def, ws),
          comm((MPI_Comm)OperatorBase::Arg<int64_t>("comm", 0)),
          group((MPI_Group)OperatorBase::Arg<int64_t>("group", 0)) {

        if (comm == MPI_COMM_NULL) return;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(comm, &comm_size);
        MPI_Comm_rank(comm, &comm_rank);

        MPI_Group world_group;
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);
        int world_root = OperatorBase::Arg<int64_t>("root", 0);
        MPI_Group_translate_ranks(world_group, 1, &world_root, group, &comm_root);

        CHECK(comm_root != MPI_UNDEFINED)
            << "\nMPI root is not included in layer group.";
    }

    template <typename T>
    MPI_Datatype mpi_dtype() {
        auto dtype = TypeMetaToString(TypeMeta::Make<T>());
        if (dtype == "int8") return MPI_CHAR;
        else if (dtype == "uint8") return MPI_UNSIGNED_CHAR;
        else if (dtype == "int32") return MPI_INT;
        else if (dtype == "int64") return MPI_LONG_LONG;
        else if (dtype == "float16") return MPI_UNSIGNED_SHORT;
        else if (dtype == "float32") return MPI_FLOAT;
        else if (dtype == "float64") return MPI_DOUBLE;
        return MPI_DATATYPE_NULL;
    }

 public:
    MPI_Comm comm;
    MPI_Group group;
    int comm_size, comm_rank, comm_root;
    int world_size, world_rank;
};

#define USE_MODEL_MPI_FUNCTIONS \
    using ModelMPIBase<Context>::comm; \
    using ModelMPIBase<Context>::comm_size; \
    using ModelMPIBase<Context>::comm_rank; \
    using ModelMPIBase<Context>::comm_root;

}  // namespace dragon

#endif  // WITH_MPI

#endif  // DRAGON_OPERATORS_MPI_BASE_MPI_OP_H_