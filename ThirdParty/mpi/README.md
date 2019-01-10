Open MPI: Open Source High Performance Computing
===================================================

https://www.open-mpi.org/

Note
----

This folder is kept for the specified open-mpi.

Following file structure will be considered by our CMakeLists:

    .
    ├── bin                           # Binary files
        ├── mpirun
        └── ...
    ├── include                       # Include files
        ├── mpi.h
        └── ...
    ├── lib                           # Library files
        ├── libmpi.so
        └── ...
    ├── src                           # Source files
        └── ...
    ├── build.sh                      # Build script  
    └── README.md