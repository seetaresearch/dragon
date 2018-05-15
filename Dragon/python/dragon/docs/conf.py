# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

import sys
import os
import sphinx_bootstrap_theme

# basic
html_static_path = ['_static']
templates_path = ['_templates']
exclude_patterns = ['_build']

source_suffix = '.rst'
master_doc = 'index'
pygments_style = 'sphinx'
todo_include_todos = True

# extensions
sys.path.insert(0, os.path.abspath('_extensions'))
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    #'sphinx.ext.viewcode',
    'mathmacro',
]

# project
project = ''
copyright = '2017, Ting Pan'
author = 'Ting Pan'
html_logo = "dragon.png"
html_title = ""
html_short_title = ""
html_favicon = 'images/favicon.png'

version = ''
release = ''
language = None

# theme
html_theme = 'bootstrap'
html_theme_path = sphinx_bootstrap_theme.get_html_theme_path()

html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = False

html_theme_options = {
    'globaltoc_depth': -1,
    'navbar_class': "navbar navbar-inverse",
    'navbar_fixed_top': "true",
    'bootswatch_theme': "yeti",
}

html_sidebars = {'index': ['localtoc.html'],
                 'install': ['localtoc.html'],
                 'contents/**': ['localtoc.html']}

# overloads
def setup(app):
    app.config.values['autodoc_member_order'] = ('bysource', True)