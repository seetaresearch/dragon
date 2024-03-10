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

#ifndef DRAGON_UTILS_DEVICE_COMMON_MPI_H_
#define DRAGON_UTILS_DEVICE_COMMON_MPI_H_

#ifdef USE_MPI
#include <mpi.h>

#define MPI_CHECK(condition, error_string)                     \
  do {                                                         \
    int error_code = condition;                                \
    CHECK_EQ(error_code, MPI_SUCCESS) << "\n" << error_string; \
  } while (0)
#else
typedef struct ompi_communicator_t* MPI_Comm;
typedef struct ompi_datatype_t* MPI_Datatype;
typedef struct ompi_group_t* MPI_Group;
typedef struct ompi_request_t* MPI_Request;
typedef struct ompi_op_t* MPI_Op;
#define MPI_NOT_COMPILED LOG(FATAL) << "MPI library is not built with."
#endif

#endif // DRAGON_UTILS_DEVICE_COMMON_MPI_H_
