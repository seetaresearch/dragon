/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *    <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_MODULES_PYTHON_MPI_H_
#define DRAGON_MODULES_PYTHON_MPI_H_

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "dragon/modules/python/common.h"

namespace dragon {

namespace python {

namespace mpi {

#define MPI_CHECK(condition, error_string)                     \
  do {                                                         \
    int error_code = condition;                                \
    CHECK_EQ(error_code, MPI_SUCCESS) << "\n" << error_string; \
  } while (0)

void RegisterModule(py::module& m) {
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

  /*! \brief Return the world rank of current node */
  m.def("mpiWorldRank", []() {
#ifdef USE_MPI
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    return world_rank;
#else
    LOG(FATAL) << "MPI was not compiled.";
#endif
  });

  /*! \brief Return the world size of current node */
  m.def("mpiWorldSize", []() {
#ifdef USE_MPI
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    return world_size;
#else
    LOG(FATAL) << "MPI was not compiled.";
#endif
  });

  /*! \brief Create a MPI group from the ranks */
  m.def("mpiCreateGroup", [](const vec32_t& ranks, bool verbose = false) {
#ifdef USE_MPI
    // Skip the empty ranks to avoid asserting
    if (ranks.empty()) return vector<long>();

    int world_size;
    MPI_Comm group_comm;
    MPI_Group world_group, group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    MPI_CHECK(
        MPI_Group_incl(world_group, (int)ranks.size(), ranks.data(), &group),
        "Failed to include the specified ranks.");

    // Create a new group from the world group
    // Each process should call this function
    // to synchronize the information
    MPI_CHECK(
        MPI_Comm_create(MPI_COMM_WORLD, group, &group_comm),
        "Failed to create the group from ranks.");

    if (verbose && group_comm != MPI_COMM_NULL) {
      // Log the debug string at the first rank
      int world_rank, local_size;
      MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
      if (world_rank == ranks[0]) {
        std::stringstream ss;
        ss << "Rank[" << world_rank << "]: "
           << "Create a group of " << ranks.size() << " members.";
        ss << "\nGroup: [";
        for (auto rank : ranks) {
          if (rank != ranks[0])
            ss << rank << ", ";
          else
            ss << rank << "*, ";
        }
        string debug_string = ss.str();
        debug_string[debug_string.size() - 2] = ']';
        LOG(INFO) << debug_string;
      }
    }
    return vector<long>({(long)group_comm, (long)group});
#else
    LOG(FATAL) << "MPI was not compiled.";
    return vector<long>();
#endif
  });

  /*! \brief Finalize the MPI environment */
  m.def("mpiFinalize", []() {
#ifdef USE_MPI
    MPI_Finalize();
#else
    LOG(FATAL) << "MPI was not compiled.";
#endif
  });
}

#undef MPI_CHECK

} // namespace mpi

} // namespace python

} // namespace dragon

#endif // DRAGON_MODULES_PYTHON_MPI_H_
