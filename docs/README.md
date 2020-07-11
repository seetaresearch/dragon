Building Dragon Documentation
=============================

This page will help you to build the following documentations:

Dragon C++ API: https://dragon.seetatech.com/api/cc

Dragon Python API: https://dragon.seetatech.com/api/python

Build Documentation of C++ API
------------------------------

```bash
cd dragon/docs/api/cc
doxygen Doxyfile
```

Then, open the ```docs/api/cc/html/index.html``` in your browser.

Build Documentation of Python API
---------------------------------

```bash
pip install sphinx_seeta_theme
cd dragon/docs/api/python
make html
```

Then, open the ```docs/api/python/index.html``` in your browser.
