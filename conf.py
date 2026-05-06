"""Sphinx configuration for the Ginkgo documentation site."""
import os
from datetime import date

project = "Ginkgo"
author = "The Ginkgo authors"
copyright = f"{date.today().year}, {author}"

# Versioning: read from env var; default to "stable" for local dev.
ginkgo_doc_version = os.environ.get("GINKGO_DOC_VERSION", "stable")

# Expose the version to templates for the version switcher.
html_context = {
    "ginkgo_doc_version": ginkgo_doc_version,
    "ginkgo_doc_version_label": "develop" if ginkgo_doc_version == "develop" else "stable",
}

# When building the develop subsite, prefix all internal links with /develop/.
if ginkgo_doc_version == "develop":
    html_baseurl = "https://gko-project.org/develop/"
else:
    html_baseurl = "https://gko-project.org/"

extensions = [
    "myst_parser",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinxcontrib.mermaid",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"
exclude_patterns = ["_build", ".venv", "Thumbs.db", ".DS_Store", "docs/superpowers", "README.md", "feedback.md"]

# Treat warnings as errors during CI builds.
nitpicky = False

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
templates_path = ["_templates"]
html_css_files = [
    "ginkgo-theme/custom.css",
]
html_js_files = [
    "ginkgo-theme/citation-copy.js",
]

# External link aliases. {gko-api}`gko::matrix::Csr` resolves to the standalone
# Doxygen deploy. The Doxygen URL pattern is fragile; centralising it here
# means a single edit if the deploy ever moves.
extlinks = {
    "gko-api": (
        "https://ginkgo-project.github.io/ginkgo-generated-documentation/"
        "doc/develop/classgko_1_1%s.html",
        "%s",
    ),
}

# MyST roles need to be allowed for inline use.
myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath"]
# Auto-generate URL-anchor IDs for headings up to h2 so cross-page section links
# (e.g., [Allocators](executor-model.md#allocators)) resolve.
myst_heading_anchors = 2
html_title = "Ginkgo"

html_logo = "_static/ginkgo-theme/leaf.png"
html_favicon = "_static/ginkgo-theme/leaf.png"

html_theme_options = {
    "show_prev_next": False,
    "navigation_with_keys": False,
    # Light mode uses PyData's default "tango" (light bg, dark tokens).
    # Dark mode uses "monokai" (dark bg, bright tokens).
    "pygments_dark_style": "monokai",
    "logo": {
        "image_light": "_static/ginkgo-theme/leaf.png",
        "image_dark": "_static/ginkgo-theme/leaf.png",
        "text": "Ginkgo",
    },
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "version-switcher", "navbar-icon-links"],
    "navbar_persistent": ["search-button"],
    "header_links_before_dropdown": 6,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/ginkgo-project/ginkgo",
            "icon": "fab fa-github",
        },
    ],
    "use_edit_page_button": False,
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version", "theme-version"],
    "secondary_sidebar_items": ["page-toc"],
    "show_toc_level": 2,
}
