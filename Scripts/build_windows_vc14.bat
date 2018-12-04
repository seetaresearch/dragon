:: #############################################################################
:: Example command to build on Windows for Visual Studio 2015 (VC14).
:: #############################################################################

@echo off
setlocal

SET ORIGINAL_DIR=%cd%
SET REPO_ROOT=%~dp0%..
SET DRAGON_ROOT=%REPO_ROOT%\Dragon
SET THIRD_PARTY_DIR=%REPO_ROOT%\3rdparty
SET CMAKE_GENERATOR="Visual Studio 14 2015 Win64"

:: Build options
SET BUILD_CXX_API=ON
SET BUILD_PYTHON_API=ON

:: Python options
:: Set your python "interpreter" if necessary
:: If not, a default interpreter will be used
:: SET PYTHON_EXECUTABLE=C:/Anaconda3/python

if %BUILD_PYTHON_API% == ON (
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
  -DBUILD_PYTHON_API=%BUILD_PYTHON_API% ^
  -DBUILD_CXX_API=%BUILD_CXX_API% ^
  -D3RDPARTY_DIR=%THIRD_PARTY_DIR% ^
  -DPYTHON_EXECUTABLE=%PYTHON_EXECUTABLE% ^
  || goto :label_error

echo=
echo -------------------------  BUILDING CONFIGS -------------------------
echo=

cmake --build . --target INSTALL --config Release -- /maxcpucount:%NUMBER_OF_PROCESSORS% || goto :label_error
cd %DRAGON_ROOT%\python
%PYTHON_EXECUTABLE% setup.py install  || goto :label_error

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