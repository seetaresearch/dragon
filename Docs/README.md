Building Dragon Documentation
=============================

This page will help you to build the following documentations:

Dragon CXX API: http://dragon.seetatech.com/api/cpp/index.html

Dragon Python API: http://dragon.seetatech.com/api/python/index.html


Build Documentation of CXX API
------------------------------

```bash
cd Dragon/Docs/api/cxx
doxygen Doxyfile
```

Then, open the ```./html/index.html``` in your browser.


Build Documentation of Python API
---------------------------------

```bash
pip install sphinx_bootstrap_theme
cd Dragon/Docs/api/python
make html
```

Then, open the ```./_build/html/index.html``` in your browser.