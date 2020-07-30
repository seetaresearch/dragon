Building Dragon Documentation
=============================

This page will help you to build the following documentations:

Python API: https://dragon.seetatech.com/api/python

C++ API: https://dragon.seetatech.com/api/cc

Requirements
------------

- sphinx >= 3.0.2

```bash
pip install sphinx
```

- sphinx_seeta_theme

```bash
pip install sphinx_seeta_theme
```

- doxygen (C++ API only)

See: http://www.doxygen.org/download.html

Build Documentation of Python API
---------------------------------

```bash
cd dragon/docs/api/python && make html
```

Then, open the ``docs/_build/api/python/index.html`` in your browser.

Build Documentation of C++ API
------------------------------

```bash
cd dragon/docs/api/cc && make doxygen && make html
```

Then, open the ``docs/_build/api/cc/index.html`` in your browser.
