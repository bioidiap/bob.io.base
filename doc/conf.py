import os
import time

from importlib.metadata import distribution

# -- General configuration -----------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
needs_sphinx = "1.3"

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.todo",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "auto_intersphinx",
]

# Be picky about warnings
nitpicky = True

# Ignores stuff we can't easily resolve on other project's sphinx manuals
nitpick_ignore = []

# Allows the user to override warnings from a separate file
if os.path.exists("nitpick-exceptions.txt"):
    for line in open("nitpick-exceptions.txt"):
        if line.strip() == "" or line.startswith("#"):
            continue
        dtype, target = line.split(None, 1)
        target = target.strip()
        nitpick_ignore.append((dtype, target))

# Always includes todos
todo_include_todos = True

# Generates auto-summary automatically
autosummary_generate = True

# Create numbers on figures with captions
numfig = True

# If we are on OSX, the 'dvipng' path maybe different
dvipng_osx = "/Library/TeX/texbin/dvipng"
if os.path.exists(dvipng_osx):
    pngmath_dvipng = dvipng_osx

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "bob.io.base"
package = distribution(project)

copyright = "%s, Idiap Research Institute" % time.strftime("%Y")

# The short X.Y version.
version = package.version
# The full version, including alpha/beta/rc tags.
release = version

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["links.rst"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
pygments_dark_style = "monokai"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# Some variables which are useful for generated material
project_variable = project.replace(".", "_")
short_description = package.metadata["Summary"]
owner = ["Idiap Research Institute"]

# -- Options for HTML output ---------------------------------------------------


html_theme = "sphinx_rtd_theme"

html_title = f"{project} {release}"


html_logo = "img/logo.png"
html_favicon = "img/favicon.ico"

# -- Post configuration --------------------------------------------------------

# Default processing flags for sphinx
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

auto_intersphinx_packages = [
    ("python", "3"),
    "numpy",
    "matplotlib",
    "pillow",
]
