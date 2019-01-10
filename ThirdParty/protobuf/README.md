Protocol Buffers - Google's data interchange format
===================================================

https://developers.google.com/protocol-buffers/

Note
----

This folder is kept for the specified protobuf, or the released libraries of Visual Studio.

Following file structure will be considered by our CMakeLists:

    .
    ├── bin                           # Binary files
        ├── protoc
        └── protoc.exe
    ├── include                       # Include files
        └── google
            └── protobuf
                └── *.h
    ├── lib                           # Library files
        ├── libprotobuf.so
        └── protobuf.lib
    └── README.md