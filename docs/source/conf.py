# Path Setup
import os
import sys
import sphinx_rtd_theme
sys.path.append(os.path.join(os.path.dirname(__name__), ".."))

# Project information
project = "Metaheuristics"
copyright = "2019, Matt Buckley"
author = "Matt Buckley"

version = '0.1'
release = '0.1.0'


# General Configuration
extensions = ['sphinx.ext.viewcode',
              'sphinx.ext.autodoc',
              'sphinx_rtd_theme',
              'numpydoc']
templates_path = ['_templates']
source_suffix = ['.rst', '.md']
master_doc = 'index'
exclude_patterns = []
pygments_style = 'sphinx'
todo_include_todos = False


# Options for HTML output
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
numpydoc_show_class_members = True
class_members_toctree = False


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'metaheuristicsdoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'metaheuristics.tex', 'metaheuristics Documentation',
     'Matt Buckley', 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'metaheuristics', 'metaheuristics Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'metaheuristics', 'metaheuristics Documentation',
     author, 'metaheuristics', 'One line description of project.',
     'Miscellaneous'),
]
