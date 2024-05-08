# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'survivors'
copyright = '2024, Iulii Vasilev'
author = 'Iulii Vasilev'
release = '1.6.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import sphinx_rtd_theme

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.githubpages",
    "sphinx_gallery.gen_gallery",
    "sphinx_rtd_theme",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"  # 'alabaster'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']

html_context = {
    'display_github': False,  # Add 'Edit on Github' link instead of 'View page source'
    'last_updated': True,
    'commit': False,
    'css_files': ['_static/css/theme.css',],
}

sphinx_gallery_conf = {
    "doc_module": "survivors",
    "show_memory": False,
    "examples_dirs": ["../doc"],
    "gallery_dirs": ["auto_examples"],
    "plot_gallery": "True",
}
