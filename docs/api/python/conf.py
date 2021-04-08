# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Sphinx configuration for Python API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def path_to(href, index=False):
    if index:
        if len(href) == 0:
            return 'index.html'
        return href + '/index.html'
    else:
        return href + '.html'


# Basic
html_static_path = ['../_static']
master_doc = 'index'
source_suffix = '.rst'

# Extension
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinxcontrib.katex',
    # 'sphinx_seeta_theme.ext.viewcode',
]
napoleon_use_rtype = False

# Project
project = 'dragon'
copyright = 'Copyright (c) 2017-present, SeetaTech, Co.,Ltd'
author = 'SeetaTech, Co.,Ltd'
with open('../../../dragon/version.txt', 'r') as f:
    version = f.read().strip()

# HTML
html_theme = 'seeta'
html_title = ''
html_short_title = ''
html_logo = '../_static/images/dragon.png'
html_favicon = '../_static/favicon.ico'
html_copy_source = False
html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = False
html_scaled_image_link = False
html_theme_options = {
    'navbar_links': {
        'Install': path_to('../../install', 1),
        'API': [
            ('master', path_to('../../api/python', 1)),
            ('versions...', path_to('../../versions', 1)),
        ],
        'Github': 'https://github.com/seetaresearch/dragon',
    },
    'navbar_logo_link': path_to('../..', 1),
    'sidebar_title': 'Python v{}'.format(version),
    'sidebar_title_link': path_to('../../versions', 1),
    'breadcrumb_links': [
        ('Dragon', path_to('../..', 1)),
        ('API', path_to('../../versions', 1)),
        ('Dragon v{}'.format(version.replace('a0', '')), path_to('../../api', 1)),
        ('Python', path_to('', 1)),
    ],
}
html_sidebars = {
    'index': ['localtoc.html'],
    'caffe': ['localtoc.html'],
    'caffe/**': ['localtoc.html'],
    'dali': ['localtoc.html'],
    'dali/**': ['localtoc.html'],
    'dragon': ['localtoc.html'],
    'dragon/**': ['localtoc.html'],
    'onnx': ['localtoc.html'],
    'onnx/**': ['localtoc.html'],
    'tensorflow': ['localtoc.html'],
    'tensorflow/**': ['localtoc.html'],
    'tensorlayer/**': ['localtoc.html'],
    'tensorrt': ['localtoc.html'],
    'tensorrt/**': ['localtoc.html'],
    'torch': ['localtoc.html'],
    'torch/**': ['localtoc.html'],
    'torchvision/**': ['localtoc.html'],
    '_modules/**': ['localtoc.html'],
    'search': ['localtoc.html'],
}

# LaTex
latex_documents = [(
    master_doc,
    'dragon.tex',
    'Dragon - Python API',
    author,
    'manual',
)]
latex_elements = {
    'utf8extra': '',
    'inputenc': '',
    'babel': r'''\usepackage[english]{babel}''',
    'preamble': r'''
\usepackage{enumitem}
\usepackage{tocloft}
\renewcommand{\cfttoctitlefont}{\huge\bfseries}
\usepackage{fontspec}
\setmainfont{Source Serif Pro}
\setsansfont{Source Serif Pro}
\setmonofont{Source Serif Pro}
\setcounter{tocdepth}{2}
\usepackage[draft]{minted}
\fvset{breaklines=true, breakanywhere=true}
\setlength{\headheight}{13.6pt}
\setlength{\itemindent}{-1pt}
\addto\captionsenglish{\renewcommand{\chaptername}{}}
\makeatletter
    \renewcommand*\l@subsection{\@dottedtocline{2}{3.8em}{3.8em}}
    \fancypagestyle{normal}{
        \fancyhf{}
        \fancyfoot[LE,RO]{{\py@HeaderFamily\thepage}}
        \fancyfoot[LO]{{\py@HeaderFamily\nouppercase{\rightmark}}}
        \fancyfoot[RE]{{\py@HeaderFamily\nouppercase{\leftmark}}}
        \fancyhead[LE,RO]{{\py@HeaderFamily}}
     }
\makeatother
''',
    'maketitle': r'''
\pagenumbering{Roman} %% % to avoid page 1 conflict with actual page 1

\makeatletter
\begin{titlepage}

    \noindent\rule[0.25\baselineskip]{\textwidth}{1pt}

    \vspace*{5mm}
    \begin{figure}[!h]
        \raggedleft
        \includegraphics[scale=0.3]{logo.png}
    \end{figure}

    \raggedleft
    \vspace*{5mm}
    \textbf{\Huge \@title}

    \vspace*{40mm}
    \LARGE \@author

    \vspace*{40mm}
    \LARGE \today

\end{titlepage}
\makeatother

\pagenumbering{arabic}
''',
    'pointsize': '10pt',
    'classoptions': ',oneside',
    'figure_align': 'H',
    'fncychap': '\\usepackage[Sonny]{fncychap}',
    'printindex': '',
    'sphinxsetup': ' \
        hmargin={0.75in,0.75in}, \
        vmargin={0.5in,1in}, \
        verbatimhintsturnover=false, \
        verbatimsep=0.75em, \
        verbatimhintsturnover=false, \
        verbatimwithframe=false, \
        VerbatimColor={rgb}{0.949,0.949,0.949}, \
        HeaderFamily=\\rmfamily\\bfseries',
}

latex_domain_indices = False
latex_engine = 'xelatex'
latex_logo = '../_static/images/logo.png'
