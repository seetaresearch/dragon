:: ###############################################################
:: Command file to build on Windows for Visual Studio 2019 (VC142)
:: ###############################################################

@echo off
setlocal

:: Build variables
set ORIGINAL_DIR=%cd%
set REPO_ROOT=%~dp0%..
set DRAGON_ROOT=%REPO_ROOT%\dragon
set THIRD_PARTY_DIR=%REPO_ROOT%\third_party
set CMAKE_GENERATOR="Visual Studio 16 2019"

:: Build options
set BUILD_PYTHON=ON
set BUILD_RUNTIME=OFF

:: Optional libraries
set USE_CUDA=ON
set USE_CUDNN=ON
set USE_OPENMP=ON
set USE_AVX=ON
set USE_AVX2=ON
set USE_FMA=ON

:: Protobuf SDK options
set PROTOBUF_SDK_ROOT_DIR=%THIRD_PARTY_DIR%\protobuf

:: Protobuf Compiler options
:: Set the protobuf compiler(i.e., protoc) if necessary.
:: If not, a compiler in the sdk or environment will be used.
set PROTOBUF_PROTOC_EXECUTABLE=%PROTOBUF_SDK_ROOT_DIR%\bin\protoc

:: Python options
:: Set your python "interpreter" if necessary.
:: If not, a default interpreter will be used.
:: set PYTHON_EXECUTABLE=X:/Anaconda3/python

if %BUILD_PYTHON% == ON (
  if NOT DEFINED PYTHON_EXECUTABLE (
    for /F %%i in ('python -c "import sys;print(sys.executable)"') do (set PYTHON_EXECUTABLE=%%i)
  )
)

echo=
echo -------------------------  BUILDING CONFIGS -------------------------
echo=

echo -- DRAGON_ROOT=%DRAGON_ROOT%
echo -- CMAKE_GENERATOR=%CMAKE_GENERATOR%

if not exist %DRAGON_ROOT%\build mkdir %DRAGON_ROOT%\build
cd %DRAGON_ROOT%\build

cmake .. ^
  -G%CMAKE_GENERATOR% ^
  -Ax64 ^
  -DBUILD_PYTHON=%BUILD_PYTHON% ^
  -DBUILD_RUNTIME=%BUILD_RUNTIME% ^
  -DUSE_CUDA=%USE_CUDA% ^
  -DUSE_CUDNN=%USE_CUDNN% ^
  -DUSE_OPENMP=%USE_OPENMP% ^
  -DUSE_AVX=%USE_AVX% ^
  -DUSE_AVX2=%USE_AVX2% ^
  -DUSE_FMA=%USE_FMA% ^
  -DTHIRD_PARTY_DIR=%THIRD_PARTY_DIR% ^
  -DPROTOBUF_SDK_ROOT_DIR=%PROTOBUF_SDK_ROOT_DIR% ^
  -DPROTOBUF_PROTOC_EXECUTABLE=%PROTOBUF_PROTOC_EXECUTABLE% ^
  -DPYTHON_EXECUTABLE=%PYTHON_EXECUTABLE% ^
  || goto :label_error

echo=
echo -------------------------  BUILDING CONFIGS -------------------------
echo=

cmake --build . --target INSTALL --config Release -- /maxcpucount:%NUMBER_OF_PROCESSORS% || goto :label_error
cd %DRAGON_ROOT%
%PYTHON_EXECUTABLE% setup.py install || goto :label_error

echo=
echo Built successfully
cd %ORIGINAL_DIR%
endlocal
pause
exit /b 0

:label_error
echo=
echo Building failed
cd %ORIGINAL_DIR%
endlocal
pause
exit /b 1
