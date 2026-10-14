"""Sphinx configuration for the Ginkgo documentation site."""
import json
import os
import re
import zipfile
from datetime import date
from pathlib import Path

from docutils import nodes
from sphinx import addnodes

project = "Ginkgo"
author = "The Ginkgo authors"
copyright = f"{date.today().year}, {author}"

# Versioning. Each docs branch documents one Ginkgo ref, pinned in
# `.ginkgo-ref` (e.g. `v1.11.0` or `develop`). CI passes the switcher
# version and the site URL explicitly; local builds derive the version
# from the pin so the switcher highlights the right entry.
_ginkgo_ref_file = Path(__file__).resolve().parent / ".ginkgo-ref"


def _resolve_doc_version(env_value, ref_file):
    """Return the switcher version for this build.

    `GINKGO_DOC_VERSION` wins when set. Otherwise a release-tag pin
    `vMAJOR.MINOR.PATCH` maps to `MAJOR.MINOR`, any other pin (a branch
    such as `develop`) is used as-is, and a missing pin means `develop`.
    """
    if env_value:
        return env_value
    if not ref_file.is_file():
        return "develop"
    ref = ref_file.read_text(encoding="utf-8").strip()
    match = re.fullmatch(r"v(\d+)\.(\d+)\.\d+", ref)
    return f"{match.group(1)}.{match.group(2)}" if match else ref


ginkgo_doc_version = _resolve_doc_version(
    os.environ.get("GINKGO_DOC_VERSION"), _ginkgo_ref_file
)
# The theme's version-warning banner compares Sphinx's `version` with the
# preferred entry in switcher.json, so it must equal the switcher version.
version = release = ginkgo_doc_version
html_baseurl = os.environ.get("GINKGO_DOC_BASEURL", "https://gko-project.org/")


def _resolve_stable_url(env_value, switcher_file):
    """Return the URL of the stable docs, the preferred switcher entry.

    CI passes it from `main`'s version list in `GINKGO_DOC_STABLE_URL`;
    local builds read this branch's copy of the list.
    """
    if env_value:
        return env_value
    entries = json.loads(switcher_file.read_text(encoding="utf-8"))
    return next(e["url"] for e in entries if e.get("preferred"))


ginkgo_stable_url = _resolve_stable_url(
    os.environ.get("GINKGO_DOC_STABLE_URL"),
    Path(__file__).resolve().parent / "_static" / "switcher.json",
)

extensions = [
    "myst_parser",
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinxcontrib.mermaid",
    "breathe",
]

source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
master_doc = "index"

# Signatures longer than this render with one parameter per line — much
# easier to scan for functions with many template / formal parameters
# (e.g., distributed-matrix constructors, factory `create` overloads).
# Sphinx auto-indents the continuation lines to align under the opening
# parenthesis.
maximum_signature_line_length = 70
exclude_patterns = [
    "_build", ".venv", "Thumbs.db", ".DS_Store",
    "README.md", "feedback.md",
    "_doxygen/xml", "_doxygen/Doxyfile",
    # `code-samples/<name>/` directories hold runnable companions to
    # how-to pages — C++ sources, CMakeLists, and a README. Sphinx
    # ingests README.md from each, then complains it's not in any
    # toctree, so we exclude the whole tree from the documentation
    # build. The bundles are still zipped + served as downloads by the
    # `_zip_code_samples` hook in setup().
    "code-samples",
]

# Breathe: bridges Doxygen XML into Sphinx. The XML directory is produced by
# `make doxygen-xml` (which runs Doxygen against a Ginkgo source checkout
# pointed to by $GINKGO_SOURCE_DIR). If the directory is missing, the rest of
# the site still builds — only pages that embed `{doxygenclass}` / friends
# will emit warnings. See README for the workflow.
_doxygen_xml_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_doxygen", "xml")
breathe_projects = {"ginkgo": _doxygen_xml_dir}
breathe_default_project = "ginkgo"
breathe_show_include = False
breathe_domain_by_extension = {"hpp": "cpp", "h": "cpp", "cuh": "cpp"}

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

# MathJax: MyST's dollarmath extension wraps every page body in a
# `tex2jax_ignore` container so it can pre-render `$...$` itself without
# MathJax double-processing. That blanket ignore also swallows the math
# Breathe emits from Doxygen `\f$...\f$` comments. Expanding
# `processHtmlClass` to include `math` rescues those spans (Sphinx writes
# them as `<span class="math notranslate nohighlight">…</span>`) without
# disturbing MyST's own pre-rendering pipeline.
mathjax3_config = {
    "options": {
        "ignoreHtmlClass": "tex2jax_ignore",
        "processHtmlClass": "tex2jax_process|mathjax_process|math",
    },
}
# myst-parser also rewrites processHtmlClass to add its own `output_area`
# entry; the merged result still includes `math`, so the override warning
# is harmless. Silence it.
suppress_warnings = ["myst.mathjax"]
# Auto-generate URL-anchor IDs for headings up to h3 so cross-page section links
# (e.g., [Allocators](executor-model.md#allocators) or
# [A worked example](distributed.md#a-worked-example)) resolve.
myst_heading_anchors = 3
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
    # Version switcher. The list of versions is `_static/switcher.json` on
    # `main`, fetched from the production site at page load, so adding a
    # version updates the dropdown on every older build too. Builds never
    # fetch it themselves.
    "switcher": {
        "json_url": "https://gko-project.org/_static/switcher.json",
        "version_match": ginkgo_doc_version,
    },
    "check_switcher": False,
    # The develop docs show the permanent banner below instead.
    "show_version_warning_banner": ginkgo_doc_version != "develop",
}

# The develop docs are served at the site root, so they carry a banner
# that cannot be dismissed (the theme's announcement has no close button;
# custom.css colours it orange) and links to the stable docs.
if ginkgo_doc_version == "develop":
    html_theme_options["announcement"] = (
        '<span class="ginkgo-develop-banner">'
        '<i class="fa-solid fa-triangle-exclamation" aria-hidden="true"></i> '
        "You are reading the documentation for Ginkgo's "
        "<strong>unstable development branch</strong>; pages and APIs can "
        "change at any time. For the latest release, see the "
        f'<a href="{ginkgo_stable_url}">stable documentation</a>.'
        "</span>"
    )


def _split_class_bases(app, doctree, fromdocname):
    """Move long base-class clauses out of class signatures.

    Sphinx's C++ domain renders `class Foo : public Bar, public Baz, ...`
    inline in a single declaration line. For mixin-heavy types like
    `gko::matrix::Csr` (11+ bases) the resulting heading is a wall of
    tokens. This walks every class/struct desc node, locates the `:`
    that introduces the inheritance list, captures the nodes from there
    up to (but excluding) the trailing headerlink, and reinserts them
    as a styled `Inherits from: …` paragraph at the top of the desc
    content — matching the layout Doxygen produces.
    """
    for desc in doctree.traverse(addnodes.desc):
        if desc.get("objtype") not in ("class", "struct"):
            continue
        sig = next(
            (c for c in desc.children if isinstance(c, addnodes.desc_signature)),
            None,
        )
        if sig is None:
            continue
        content = next(
            (c for c in desc.children if isinstance(c, addnodes.desc_content)),
            None,
        )

        # The class declaration sits on the last desc_signature_line when
        # template lines wrap above; otherwise the signature itself holds
        # the inline tokens directly.
        if sig.children and isinstance(sig.children[-1], addnodes.desc_signature_line):
            line = sig.children[-1]
        else:
            line = sig

        # Locate the `:` punctuation that introduces inheritance — must
        # come AFTER a desc_name (the class name) so we don't confuse it
        # with template-parameter colons or scope resolutions.
        colon_idx = None
        for i, child in enumerate(line.children):
            if (
                isinstance(child, addnodes.desc_sig_punctuation)
                and child.astext().strip() == ":"
                and any(isinstance(s, addnodes.desc_name) for s in line.children[:i])
            ):
                colon_idx = i
                break
        if colon_idx is None:
            continue

        # Preserve the trailing headerlink in the signature.
        end_idx = len(line.children)
        for j in range(len(line.children) - 1, colon_idx, -1):
            ch = line.children[j]
            if isinstance(ch, nodes.reference) and "headerlink" in ch.get("classes", []):
                end_idx = j
                break

        extracted = list(line.children[colon_idx:end_idx])
        if not extracted:
            continue
        del line.children[colon_idx:end_idx]

        # Skip the leading `:` and surrounding whitespace — the "Inherits
        # from:" label replaces them.
        i = 0
        while i < len(extracted):
            ch = extracted[i]
            if (
                isinstance(ch, addnodes.desc_sig_punctuation)
                and ch.astext().strip() == ":"
            ):
                i += 1
                continue
            if isinstance(ch, (addnodes.desc_sig_space,)):
                i += 1
                continue
            if isinstance(ch, nodes.Text) and not ch.astext().strip():
                i += 1
                continue
            break

        # Split the extracted nodes into one group per top-level base.
        # Top-level commas are commas at angle-bracket depth 0 — the
        # commas inside template arguments like `Csr<default_precision,
        # int32>` must stay attached to their group.
        groups = [[]]
        depth = 0
        for ch in extracted[i:]:
            text = ch.astext().strip() if hasattr(ch, "astext") else ""
            if isinstance(ch, addnodes.desc_sig_punctuation):
                if text == "<":
                    depth += 1
                    groups[-1].append(ch)
                    continue
                if text == ">":
                    depth -= 1
                    groups[-1].append(ch)
                    continue
                if text == "," and depth == 0:
                    groups.append([])
                    continue
            # Drop leading whitespace/space nodes that follow a comma
            # boundary so each line starts at the access specifier.
            if not groups[-1] and isinstance(
                ch, (addnodes.desc_sig_space, nodes.Text)
            ) and not text:
                continue
            groups[-1].append(ch)

        # Drop trailing empty group if the extracted slice happened to
        # end on a comma.
        groups = [g for g in groups if g]

        label = nodes.paragraph(classes=["cpp-inherits-label-line"])
        label += nodes.inline(
            text="Inherits from",
            classes=["cpp-inherits-label"],
        )

        base_list = nodes.bullet_list(classes=["cpp-inherits-list"])
        for group in groups:
            item = nodes.list_item()
            line = nodes.paragraph(classes=["cpp-inherits-item"])
            for node in group:
                line += node
            item += line
            base_list += item

        if content is None:
            content = addnodes.desc_content()
            desc += content
        content.insert(0, base_list)
        content.insert(0, label)


def _format_subclassed_by(app, doctree, fromdocname):
    """Reformat Breathe's "Subclassed by ..." paragraph as a bulleted list.

    Breathe emits the subclass list as one long paragraph: the literal text
    "Subclassed by " followed by comma-separated cross-references. For
    heavily-inherited bases like LinOp or Transposable that produces an
    unreadable wall of names. This walks each paragraph whose first text
    child starts with "Subclassed by", takes the remaining children, splits
    them on top-level commas (so template-argument commas inside
    `Foo<a, b>` references stay attached to their group), and rewrites the
    paragraph as a "Subclassed by:" label followed by a `bullet_list` of
    `list_item`s — one per subclass.
    """
    paragraphs_to_replace = []
    for para in doctree.traverse(nodes.paragraph):
        if not para.children:
            continue
        first = para.children[0]
        if not isinstance(first, nodes.Text):
            continue
        text = first.astext()
        if not text.startswith("Subclassed by"):
            continue
        paragraphs_to_replace.append(para)

    for para in paragraphs_to_replace:
        # Drop the "Subclassed by " prefix from the first text node.
        leading = para.children[0].astext()
        remainder_text = leading[len("Subclassed by"):].lstrip(" ,")
        rest = list(para.children[1:])
        if remainder_text:
            rest.insert(0, nodes.Text(remainder_text))

        # Split on top-level commas. Each separator is a Text node that
        # contains a ',' character at depth zero (Breathe inserts these
        # between cross-refs). Template-argument commas are inside
        # reference nodes so they're not at the paragraph level.
        groups = [[]]
        for child in rest:
            if isinstance(child, nodes.Text):
                pieces = child.astext().split(",")
                for i, piece in enumerate(pieces):
                    piece = piece.strip()
                    if piece:
                        groups[-1].append(nodes.Text(piece))
                    if i < len(pieces) - 1:
                        groups.append([])
            else:
                groups[-1].append(child)

        groups = [g for g in groups if g]
        if not groups:
            continue

        label = nodes.paragraph(classes=["cpp-subclassed-label-line"])
        label += nodes.inline(
            text="Subclassed by",
            classes=["cpp-subclassed-label"],
        )

        sub_list = nodes.bullet_list(classes=["cpp-subclassed-list"])
        for group in groups:
            item = nodes.list_item()
            line = nodes.paragraph(classes=["cpp-subclassed-item"])
            for node in group:
                line += node
            item += line
            sub_list += item

        # Replace the original paragraph in its parent.
        parent = para.parent
        index = parent.index(para)
        parent.remove(para)
        parent.insert(index, sub_list)
        parent.insert(index, label)


def _zip_code_samples(app):
    """Zip every `code-samples/<name>/` directory into a sibling .zip.

    Pages that want to offer a downloadable bundle of runnable companion
    code (sources + CMakeLists + README) reference the resulting .zip via
    Sphinx's `{download}` role. Generating the zip on every build keeps
    the bundle in lock-step with the sources in git — no chance of
    drift between the page snippets and the downloadable archive.

    Each sample's local `build/` directory is excluded so cmake artifacts
    don't bloat the archive. The READMEs link to the published how-to
    pages at the site root; in the archive those links point at this
    build's version of the site, so a 1.11 bundle opens the 1.11 pages.
    """
    site_root = "https://gko-project.org/"
    samples_dir = Path(app.srcdir) / "code-samples"
    if not samples_dir.is_dir():
        return
    for sample in sorted(samples_dir.iterdir()):
        if not sample.is_dir():
            continue
        out_zip = samples_dir / f"{sample.name}.zip"
        with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(sample.rglob("*")):
                # Skip cmake build artefacts (sample-local `build/` tree).
                if "build" in path.relative_to(sample).parts:
                    continue
                if not path.is_file():
                    continue
                arcname = str(Path(sample.name) / path.relative_to(sample))
                if path.suffix == ".md" and app.config.html_baseurl != site_root:
                    text = path.read_text(encoding="utf-8")
                    zf.writestr(
                        zipfile.ZipInfo.from_file(path, arcname),
                        text.replace(site_root, app.config.html_baseurl),
                        zipfile.ZIP_DEFLATED,
                    )
                else:
                    zf.write(path, arcname=arcname)


def setup(app):
    app.connect("builder-inited", _zip_code_samples)
    app.connect("doctree-resolved", _split_class_bases)
    app.connect("doctree-resolved", _format_subclassed_by)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
