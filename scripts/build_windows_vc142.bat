:: ###############################################################
:: Command file to build on Windows for Visual Studio 2019 (VC142)
:: ###############################################################

@echo off
setlocal

:: Build Variables
set ORIGINAL_DIR=%cd%
set SOURCE_DIR=%~dp0%\..
set BUILD_DIR=%~dp0%\..\build
set THIRD_PARTY_DIR=%SOURCE_DIR%\third_party
set CMAKE_GENERATOR="Visual Studio 16 2019"

:: Build Options
set BUILD_PYTHON=ON
set BUILD_RUNTIME=OFF

:: Library Options
set USE_CUDA=ON
set USE_CUDNN=ON
set USE_OPENMP=ON
set USE_AVX=ON
set USE_AVX2=ON
set USE_FMA=ON

:: Project Variables
set PROTOBUF_SDK_ROOT_DIR=%THIRD_PARTY_DIR%\protobuf
set PROTOBUF_PROTOC_EXECUTABLE=%PROTOBUF_SDK_ROOT_DIR%\bin\protoc
if %BUILD_PYTHON% == ON (
  if NOT DEFINED PYTHON_EXECUTABLE (
    for /F %%i in ('python -c "import sys;print(sys.executable)"') do (set PYTHON_EXECUTABLE=%%i)
  )
)

echo -- CMAKE_GENERATOR=%CMAKE_GENERATOR%
if not exist %BUILD_DIR% mkdir %BUILD_DIR% && cd %BUILD_DIR%

cmake %SOURCE_DIR%  ^
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

cmake --build . --target INSTALL --config Release -- /maxcpucount:%NUMBER_OF_PROCESSORS% || goto :label_error
cd %SOURCE_DIR% && pip install . | goto :label_error

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
