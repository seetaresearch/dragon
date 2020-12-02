:: #########################################################
:: Command file to build on Windows for Sphinx documentation
:: #########################################################

@echo off

:: You can set these variables from the command line
if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set BUILDDIR=..\..\_build\api\python
set ALLSPHINXOPTS=-d %BUILDDIR%\doctrees %SPHINXOPTS% .
if NOT "%PAPER%" == "" (
	set ALLSPHINXOPTS=-D latex_paper_size=%PAPER% %ALLSPHINXOPTS%
)

if "%1" == "" goto help

if "%1" == "help" (
	:help
	echo.Please use `make ^<target^>` where ^<target^> is one of
	echo.  html       to make standalone HTML files
	echo.  latex      to make LaTeX files, you can set PAPER=a4 or PAPER=letter
	echo.  latexpdf   to make LaTeX files and run them through pdflatex
	goto end
)

if "%1" == "clean" (
	for /d %%i in (%BUILDDIR%\*) do rmdir /q /s %%i
	del /q /s %BUILDDIR%\*
	goto end
)

:: Check if sphinx-build is available and fallback to Python version if any
%SPHINXBUILD% 2> nul
if errorlevel 9009 goto sphinx_python
goto sphinx_ok

:sphinx_python

set SPHINXBUILD=python -m sphinx.__init__
%SPHINXBUILD% 2> nul
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

:sphinx_ok

if "%1" == "html" (
	%SPHINXBUILD% -b html %ALLSPHINXOPTS% %BUILDDIR%
	if errorlevel 1 exit /b 1
	echo.
	echo.Build finished. The HTML pages are in %BUILDDIR%.
	goto end
)

if "%1" == "latex" (
	%SPHINXBUILD% -b latex %ALLSPHINXOPTS% %BUILDDIR%-latex
	if errorlevel 1 exit /b 1
	echo.
	echo.Build finished; the LaTeX files are in %BUILDDIR%-latex.
	goto end
)

if "%1" == "latexpdf" (
	%SPHINXBUILD% -b latex %ALLSPHINXOPTS% %BUILDDIR%-latex
	cd %BUILDDIR%-latex
	make all-pdf
	cd %~dp0
	echo.
	echo.Build finished; the PDF files are in %BUILDDIR%-latex.
	goto end
)

:end
