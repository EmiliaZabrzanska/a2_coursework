# Configuration file for the Sphinx documentation builder.

import os
import sys

# Add the project root to sys.path so autodoc can find the package
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------

project = 'Medical Imaging Coursework'
copyright = '2026, Emilia Zabrzanska'
author = 'Emilia Zabrzanska'
release = '0.1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',       # auto-generate docs from docstrings
    'sphinx.ext.napoleon',      # support for NumPy-style docstrings
    'sphinx.ext.viewcode',      # add links to source code
    'sphinx.ext.mathjax',       # render LaTeX math in docs
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']