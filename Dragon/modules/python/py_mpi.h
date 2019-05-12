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

#ifndef DRAGON_PYTHON_PY_MPI_H_
#define DRAGON_PYTHON_PY_MPI_H_

#include "py_dragon.h"

#ifdef WITH_MPI
#include <mpi.h>
#endif

namespace dragon {

namespace python {

void AddMPIMethods(pybind11::module& m) {
    m.def("MPIInit", []() {
#ifdef WITH_MPI
        // Enabling the multi-threads for Python is meaningless
        // While we will still hold this interface here
        int thread_type;
        char* mt_is_required = nullptr;
        mt_is_required = getenv("DRAGON_MPI_THREADS_ENABLE");
        if (mt_is_required != nullptr && string(mt_is_required) == "1") {
            MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &thread_type);
            CHECK_EQ(thread_type, MPI_THREAD_MULTIPLE)
                << "\nRequire to enable <MPI_THREAD_MULTIPLE> support.";
        } else {
            MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &thread_type);
        }
#else
        LOG(FATAL) << "MPI was not compiled.";
#endif
    });

    m.def("MPIRank", []() {
#ifdef WITH_MPI
        int world_rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        return world_rank;
#else
        LOG(FATAL) << "MPI was not compiled.";
#endif
    });

    m.def("MPISize", []() {
#ifdef WITH_MPI
        int world_size;
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        return world_size;
#else
        LOG(FATAL) << "MPI was not compiled.";
#endif
    });

    m.def("MPICreateGroup", [](
        int                 local_root,
        const vec32_t&      incl,
        const vec32_t&      excl) {
#ifdef WITH_MPI
        int world_size, err_code;
        MPI_Group world_group, local_group;
        MPI_Comm local_comm;
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);

        set<int> all_ranks;
        for (int i = 0; i < world_size; i++) all_ranks.insert(i);
        local_group = world_group;

        // Check include ranks
        if (!incl.empty()) {
            all_ranks.clear();
            for (auto e : incl) all_ranks.insert(e);
            err_code = MPI_Group_incl(
                world_group,
                (int)incl.size(),
                incl.data(),
                &local_group
            );
            CHECK(err_code == MPI_SUCCESS)
                << "\nFailed to create MPI Group.";
        }

        // Check exclude ranks
        if (!excl.empty()) {
            all_ranks.clear(); Set<int> tmp;
            for (auto e : excl) tmp.insert(e);
            for (int i = 0; i < world_size; i++)
                if (!tmp.count(i)) all_ranks.insert(i);
            err_code = MPI_Group_excl(
                world_group,
                (int)excl.size(),
                excl.data(),
                &local_group
            );
            CHECK(err_code == MPI_SUCCESS)
                << "\nFailed to create MPI Group.";
        }

        err_code = MPI_Comm_create(
            MPI_COMM_WORLD, local_group, &local_comm);
        CHECK(err_code == MPI_SUCCESS)
            << "\nFailed to create MPI Group.";

        if (local_comm != MPI_COMM_NULL) {
            int world_rank, local_size;
            MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
            if (world_rank == local_root) {
                MPI_Comm_size(local_comm, &local_size);
                std::stringstream ss;
                ss << "Rank[" << world_rank << "]: "
                    << "Create a mpi group of " << local_size << " members";
                ss << "\nGroup: [";
                for (auto rank : all_ranks) {
                    if (rank != local_root) ss << rank << ", ";
                    else ss << rank << "*, ";
                }
                string log_info = ss.str();
                log_info[log_info.size() - 2] = ']';
                LOG(INFO) << log_info;
            }
        }
        return vector<long>({ (long)local_comm, (long)local_group });
#else
        LOG(FATAL) << "MPI was not compiled.";
#endif
    });

    m.def("MPIFinalize", []() {
#ifdef WITH_MPI
        MPI_Finalize();
#else
        LOG(FATAL) << "MPI was not compiled.";
#endif
    });
}

}  // namespace python

}  // namespace dragon

#endif  // DRAGON_PYTHON_PY_MPI_H_