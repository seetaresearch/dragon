# Makefile for Sphinx documentation
# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
PAPER         =
BUILDDIR      = ../../_build/api/cc

# User-friendly check for sphinx-build
ifeq ($(shell which $(SPHINXBUILD) >/dev/null 2>&1; echo $$?), 1)
$(error The '$(SPHINXBUILD)' command was not found. Make sure you have Sphinx installed.)
endif

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .
NUMBER_OF_PROCESSORS:=$(shell getconf _NPROCESSORS_ONLN)

.PHONY: help clean html latex latexpdf

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  doxygen    to make Doxygen XML files"
	@echo "  html       to make standalone HTML files"
	@echo "  latex      to make LaTeX files, you can set PAPER=a4 or PAPER=letter"
	@echo "  latexpdf   to make LaTeX files and run them through pdflatex"

clean:
	rm -rf $(BUILDDIR)/*

doxygen:
	mkdir -p $(BUILDDIR)_doxygen && doxygen
	@echo
	@echo "Build finished. The Doxygen XML files are in $(BUILDDIR)_doxygen/xml."

html:
	$(SPHINXBUILD) -b html -j ${NUMBER_OF_PROCESSORS} $(ALLSPHINXOPTS) $(BUILDDIR)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)."

latex:
	$(SPHINXBUILD) -b latex -j ${NUMBER_OF_PROCESSORS} $(ALLSPHINXOPTS) $(BUILDDIR)-latex
	@echo
	@echo "Build finished; the LaTeX files are in $(BUILDDIR)-latex."
	@echo "Run \`make' in that directory to run these through (pdf)latex" \
	      "(use \`make latexpdf' here to do that automatically)."

latexpdf:
	$(SPHINXBUILD) -b latex -j ${NUMBER_OF_PROCESSORS} $(ALLSPHINXOPTS) $(BUILDDIR)-latex
	@echo "Running LaTeX files through pdflatex..."
	$(MAKE) -C $(BUILDDIR)-latex all-pdf
	@echo "pdflatex finished; the PDF files are in $(BUILDDIR)-latex."
