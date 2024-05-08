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
    # "sphinx.ext.githubpages",
    # "sphinxcontrib.jquery",
    "sphinx_gallery.gen_gallery",
    "sphinx_rtd_theme",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autosummary_generate = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


# First error: https://github.com/readthedocs/sphinx_rtd_theme/issues/1452
# https://blog.finxter.com/fixed-modulenotfounderror-no-module-named-sphinxcontrib-jquery/
# Does not work: https://github.com/readthedocs/readthedocs-sphinx-ext/issues/27
# https://github.com/Koverse/connections-documentation/blob/dc7466ab9001c7d478993b4ecbeeb66d1165d84e/conf.py#L116
# Similar problem: https://stackoverflow.com/questions/59486442/python-sphinx-css-not-working-on-github-pages
# Example: https://coderefinery.github.io/documentation/gh_workflow/

html_theme = "alabaster"  # "sphinx_rtd_theme"  # 'alabaster'
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_static_path = ['_static']

sphinx_gallery_conf = {
    "doc_module": "survivors",
    "show_memory": False,
    "examples_dirs": ["../doc"],
    "gallery_dirs": ["auto_examples"],
    "plot_gallery": "True",
}
