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

namespace dragon {

namespace python {

#ifdef WITH_MPI

#include <mpi.h>

inline PyObject* MPIInitCC(PyObject* self, PyObject* args) {
    int thread_type;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &thread_type);
    CHECK_EQ(thread_type, MPI_THREAD_MULTIPLE)
        << "\nRequire to enable <MPI_THREAD_MULTIPLE> support.";
    Py_RETURN_TRUE;
}

inline PyObject* MPIFinalizeCC(PyObject* self, PyObject* args) {
    MPI_Finalize();
    Py_RETURN_TRUE;
}

inline PyObject* MPIRankCC(PyObject* self, PyObject* args) {
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    return PyInt_FromLong(world_rank);
}

inline PyObject* MPISizeCC(PyObject* self, PyObject* args) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    return PyInt_FromLong(world_size);
}

inline PyObject* MPICreateGroupCC(PyObject* self, PyObject* args) {
    PyObject *incl, *excl, *ret;
    int local_root, world_size;
    if (!PyArg_ParseTuple(args, "iOO", &local_root, &incl, &excl)) {
        PyErr_SetString(PyExc_ValueError,
            "Excepted the local root, include and exclued list.");
        return nullptr;
    }
    MPI_Group world_group, local_group;
    MPI_Comm local_comm;
    int err_code;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    set<int> all_ranks;
    for (int i = 0; i < world_size; i++) all_ranks.insert(i);
    local_group = world_group;

    // Check inclue ranks
    int size = (int)PyList_Size(incl);
    if (size > 0) {
        all_ranks.clear();
        unique_ptr<int> incl_ranks(new int[size]);
        int* ranks = incl_ranks.get();
        for (int i = 0; i < size; i++) {
            ranks[i] = _PyInt_AsInt(PyList_GetItem(incl, i));
            all_ranks.insert(ranks[i]);
        }
        err_code = MPI_Group_incl(world_group, size, ranks, &local_group);
        CHECK(err_code == MPI_SUCCESS) << "\nFail to create mpi group.";
    }

    // Check exclude ranks
    size = (int)PyList_Size(excl);
    if (size > 0) {
        all_ranks.clear(); Set<int> tmp;
        unique_ptr<int> excl_ranks(new int[size]);
        int* ranks = excl_ranks.get();
        for (int i = 0; i < size; i++) {
            ranks[i] = _PyInt_AsInt(PyList_GetItem(excl, i));
            tmp.insert(ranks[i]);
        }
        for (int i = 0; i < world_size; i++)
            if (!tmp.count(i)) all_ranks.insert(i);
        err_code = MPI_Group_excl(world_group, size, ranks, &local_group);
        CHECK(err_code == MPI_SUCCESS) << "Fail to create mpi group.";
    }

    err_code = MPI_Comm_create(MPI_COMM_WORLD, local_group, &local_comm);
    CHECK(err_code == MPI_SUCCESS) << "Fail to create mpi group.";

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
            string log_info = ss.str(); log_info[log_info.size() - 2] = ']';
            LOG(INFO) << log_info;
        }
    }
    ret = PyList_New(2);
    PyList_SetItem(ret, 0, PyInt_FromLong((long)local_comm));
    PyList_SetItem(ret, 1, PyInt_FromLong((long)local_group));
    return ret;
}

#else  // WITH_MPI

#define MPI_NOT_IMPLEMENTED \
    LOG(FATAL) << "MPI was not compiled."; \
    Py_RETURN_TRUE

inline PyObject* MPIInitCC(PyObject* self, PyObject* args) { MPI_NOT_IMPLEMENTED; }
inline PyObject* MPIFinalizeCC(PyObject* self, PyObject* args) { MPI_NOT_IMPLEMENTED; }
inline PyObject* MPIRankCC(PyObject* self, PyObject* args) { MPI_NOT_IMPLEMENTED; }
inline PyObject* MPISizeCC(PyObject* self, PyObject* args) { MPI_NOT_IMPLEMENTED; }
inline PyObject* MPICreateGroupCC(PyObject* self, PyObject* args) { MPI_NOT_IMPLEMENTED; }

#endif  // WITH_MPI

}  // namespace python

}  // namespace dragon

#endif  // DRAGON_PYTHON_PY_MPI_H_