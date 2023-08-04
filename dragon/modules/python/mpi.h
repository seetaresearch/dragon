/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_MODULES_PYTHON_MPI_H_
#define DRAGON_MODULES_PYTHON_MPI_H_

#include <dragon/core/common.h>
#include <dragon/utils/device/common_mpi.h>

#include "dragon/modules/python/common.h"

namespace dragon {

namespace python {

void RegisterModule_mpi(py::module& m) {
  /*! \brief Return whether MPI is available */
  m.def("mpiIsAvailable", []() {
#ifdef USE_MPI
    return true;
#else
    return false;
#endif
  });

  /*! \brief Initialize the MPI environment */
  m.def("mpiInitialize", []() {
#ifdef USE_MPI
    // Enabling multi-threads for Python is meaningless.
    int thread_type;
    char* mt_is_required = nullptr;
    mt_is_required = getenv("DRAGON_MPI_THREAD_MULTIPLE");
    if (mt_is_required != nullptr && string(mt_is_required) == "1") {
      MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &thread_type);
      CHECK_EQ(thread_type, MPI_THREAD_MULTIPLE)
          << "\nFailed to initialize with <MPI_THREAD_MULTIPLE>.";
    } else {
      MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &thread_type);
    }
#else
    MPI_NOT_COMPILED;
#endif
  });

  /*! \brief Return the world rank of current node */
  m.def("mpiWorldRank", []() {
    int world_rank = 0;
#ifdef USE_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
#endif
    return world_rank;
  });

  /*! \brief Return the world size of current node */
  m.def("mpiWorldSize", []() {
    int world_size = 0;
#ifdef USE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
#endif
    return world_size;
  });

  /*! \brief Create a MPI group from the ranks */
  m.def("mpiCreateGroup", [](const vec32_t& ranks, bool verbose = false) {
#ifdef USE_MPI
    // Skip the empty ranks.
    if (ranks.empty()) return vector<long>();

    int world_size;
    MPI_Comm group_comm;
    MPI_Group world_group, group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_CHECK(
        MPI_Group_incl(world_group, (int)ranks.size(), ranks.data(), &group),
        "Failed to include given ranks into MPIGroup.");

    // Create a new group from the world group.
    MPI_CHECK(
        MPI_Comm_create(MPI_COMM_WORLD, group, &group_comm),
        "Failed to create MPIGroup using given ranks.");

    if (verbose && group_comm != MPI_COMM_NULL) {
      // Log the debug string at the first rank.
      int world_rank, local_size;
      MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
      if (world_rank == ranks[0]) {
        std::stringstream ss;
        ss << "Rank[" << world_rank << "]: "
           << "Create a group of " << ranks.size() << " members.";
        ss << "\nGroup: [";
        for (auto rank : ranks) {
          ss << rank << (rank != ranks[0] ? ", " : "*, ");
        }
        string debug_string = ss.str();
        debug_string[debug_string.size() - 2] = ']';
        LOG(INFO) << debug_string;
      }
    }
    return vector<long>({(long)group_comm, (long)group});
#else
    MPI_NOT_COMPILED;
    return vector<long>();
#endif
  });

  /*! \brief Finalize the MPI environment */
  m.def("mpiFinalize", []() {
#ifdef USE_MPI
    MPI_Finalize();
#else
    MPI_NOT_COMPILED;
#endif
  });
}

} // namespace python

} // namespace dragon

#endif // DRAGON_MODULES_PYTHON_MPI_H_
