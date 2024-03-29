include(CheckCXXCompilerFlag)

# ---[ Check if CXX14 is supported
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---[ Use ``-fPIC`` for all compilers
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ---[ Compiler flags
if (MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_ENABLE_EXTENDED_ALIGNED_STORAGE")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4003 /wd4114 /wd4244 /wd4251 /wd4267 /wd4273 /wd4275 /wd4800 /wd4819 /wd4996")
  string(REPLACE "/W3" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  string(REPLACE "/MD" "/MT" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  if (USE_AVX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX")
  endif()
  if (USE_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
  endif()
  if (USE_FMA)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /D__FMA__")
  endif()
  if (USE_OPENMP)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
  endif()
else()  # GNU, Clang, AppleClang
  set(CMAKE_ORIGIN)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -std=c++14")
  if (USE_NATIVE_ARCH)
    check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
    if (COMPILER_SUPPORTS_MARCH_NATIVE)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    endif()
  else()
    if (USE_AVX)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
    endif()
    if (USE_AVX2)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")
    endif()
    if (USE_FMA)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma")
    endif()
  endif()
  if (USE_MPS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-objc-arc -Wno-unguarded-availability-new")
  endif()
  if (USE_OPENMP)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    endif()
  endif()
endif()
