#!/usr/bin/env python3
"""Generate Sphinx-MyST pages for every Ginkgo example.

Reads each example's .cpp and doc/* metadata and emits an interleaved
commentary + code page, followed by a 'Full program' block with the
complete source.

Run from /home/pratik/00Sources/ginkgo/documentation:
    python3 examples/_gen_examples.py [--src GINKGO/examples] [--out DIR]

The script is idempotent — it regenerates every output file. Hand-polishing
should happen in the generated file directly *after* running; re-running
will overwrite changes. (Useful when iterating on the generator itself.)
"""
import argparse
import os
import re
import sys
from pathlib import Path

EXAMPLES_SRC = Path("/home/pratik/00Sources/ginkgo/upstream/examples")
OUT_DIR = Path("/home/pratik/00Sources/ginkgo/documentation/examples")

# Examples we deliberately skip (build glue, etc.)
SKIP = {"CMakeLists.txt", "build-setup.sh", "compare-output.py"}

SPDX_PATTERN = re.compile(r"^// SPDX-")
SECT_PATTERN = re.compile(r"^//\s*@sect[1-6]\{(.+?)\}\s*$")
# Match the boxed description block. Prefix and suffix can have any
# number of leading/trailing asterisks.
DESCRIPTION_BLOCK = re.compile(
    r"/\*+<DESCRIPTION>\*+.*?\*+<DESCRIPTION>\*+/", re.DOTALL)

# Boilerplate paragraphs found in some intro.dox files — same on every
# example, no per-example value, drop them.
BOILERPLATE_INTRO_PHRASES = [
    "About the example",
    "Each example has the following sections",
    "This gives an overview of the example",
    "The commented program: This section is intended",
    "Results: This section shows the results",
    "The plain program: This is the complete code",
]
BOILERPLATE_RESULTS_PHRASES = [
    "Comments about programming and debugging",
]


DOXY_MATH_ENV = re.compile(r"\\f\{([a-zA-Z*]+)\}\{(.*?)\\f\}", re.DOTALL)
DOXY_MATH_BRACKET = re.compile(r"(?:\\f|@f)\[(.*?)(?:\\f|@f)\]", re.DOTALL)
DOXY_MATH_DOLLAR_BLOCK = re.compile(
    r"^[ \t]*\\f\$[ \t]*\n(.*?)\n[ \t]*\\f\$[ \t]*$", re.DOTALL | re.MULTILINE)
DOXY_MATH_INLINE = re.compile(r"\\f\$(.+?)\\f\$", re.DOTALL)


def _display_math(body: str, env: str = "") -> str:
    """A MyST `$$` block. Blank lines would end the block, so drop them;
    `\\\\` line breaks need an environment for MathJax to honour them."""
    lines = [line.rstrip() for line in body.strip("\n").splitlines() if line.strip()]
    if not env and any("\\\\" in line for line in lines):
        env = "gathered"
    if env:
        lines = [f"\\begin{{{env}}}", *lines, f"\\end{{{env}}}"]
    return "\n$$\n" + "\n".join(lines) + "\n$$\n"


def doxygen_math_to_myst(text: str) -> str:
    """Doxygen math (\\f$..\\f$, \\f[..\\f], @f[..@f], \\f{env}{..\\f}) -> MyST math."""
    text = DOXY_MATH_ENV.sub(lambda m: _display_math(m.group(2), m.group(1)), text)
    text = DOXY_MATH_BRACKET.sub(lambda m: _display_math(m.group(1)), text)
    text = DOXY_MATH_DOLLAR_BLOCK.sub(lambda m: _display_math(m.group(1)), text)
    return DOXY_MATH_INLINE.sub(lambda m: "$" + m.group(1).strip() + "$", text)


def strip_html(text: str, boilerplate_phrases=()):
    """Very light HTML → plain text for intro.dox / results.dox.

    Also drops paragraphs that contain any of the boilerplate phrases — these
    are repeated verbatim across every upstream example and add no
    per-example signal."""
    text = doxygen_math_to_myst(text)
    # @code{.cpp} ... @endcode  ->  ```cpp ... ```
    text = re.sub(r"@code\{\.cpp\}", "```text", text)
    text = re.sub(r"@code", "```text", text)
    text = re.sub(r"@endcode", "```", text)
    # Drop the top-level h1 — we already have a page title.
    text = re.sub(r"<h1>.+?</h1>", "", text, flags=re.DOTALL)
    text = re.sub(r"<h2>(.+?)</h2>", r"\n\1\n", text, flags=re.DOTALL)
    text = re.sub(r"<h3>(.+?)</h3>", r"\n\1\n", text, flags=re.DOTALL)
    text = re.sub(r"<a name=\".*?\">\s*</a>", "", text)
    text = re.sub(r"</?p>", "\n", text)
    text = re.sub(r"</?em>", "*", text)
    text = re.sub(r"</?strong>", "**", text)
    text = re.sub(r"</?br\s*/?>", "  \n", text)
    # Drop any remaining HTML
    text = re.sub(r"<[^>]+>", "", text)
    # Strip @ref tokens that survived
    text = re.sub(r"@ref\s+", "", text)
    # Drop paragraphs containing boilerplate phrases
    if boilerplate_phrases:
        paragraphs = re.split(r"\n\s*\n", text)
        kept = [p for p in paragraphs
                if not any(phrase in p for phrase in boilerplate_phrases)]
        text = "\n\n".join(kept)
    # Collapse 3+ blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_cpp(path: Path):
    """Return list of (kind, payload) where kind in {section, prose, code}."""
    raw = path.read_text()
    # Strip the boxed /***<DESCRIPTION> ... <DESCRIPTION>***/ blocks; their
    # content is usually echoed in intro.dox.
    raw = DESCRIPTION_BLOCK.sub("", raw)

    lines = raw.splitlines()
    # Skip SPDX header (first few // SPDX lines + any blank line immediately
    # after) and any *only* `//` lines between them.
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if SPDX_PATTERN.match(s) or s == "//" or s == "":
            i += 1
        else:
            break

    blocks = []
    prose_buf = []
    code_buf = []

    def flush_prose():
        if prose_buf:
            blocks.append(("prose", "\n".join(prose_buf).strip()))
            prose_buf.clear()

    def flush_code():
        if code_buf:
            # Trim leading/trailing blank lines but preserve interior structure
            while code_buf and code_buf[0].strip() == "":
                code_buf.pop(0)
            while code_buf and code_buf[-1].strip() == "":
                code_buf.pop()
            if code_buf:
                blocks.append(("code", "\n".join(code_buf)))
            code_buf.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        m = SECT_PATTERN.match(stripped)
        if m:
            flush_prose()
            flush_code()
            blocks.append(("section", m.group(1).strip()))
            i += 1
            continue
        # Comment line (starts with //) — accumulate into prose
        if stripped.startswith("//"):
            flush_code()
            # Strip leading "// " or "//"
            text = stripped[2:]
            if text.startswith(" "):
                text = text[1:]
            # Light Doxygen normalisation
            text = re.sub(r"@note\s+", "**Note:** ", text)
            text = re.sub(r"@ref\s+", "", text)
            prose_buf.append(text)
            i += 1
            continue
        # Blank line — separator
        if stripped == "":
            if prose_buf:
                # Paragraph break: keep going (don't flush yet, allow multi-paragraph
                # prose blocks separated by blank comment lines... but with no '//'
                # continuation here, this is the actual end of a prose run)
                flush_prose()
            else:
                # Blank inside code — preserve
                if code_buf:
                    code_buf.append("")
            i += 1
            continue
        # Code line
        flush_prose()
        code_buf.append(line)
        i += 1

    flush_prose()
    flush_code()
    return blocks


def render_page(name: str, src_root: Path) -> str:
    cpp_path = src_root / f"{name}.cpp"
    if not cpp_path.exists():
        return ""

    doc_dir = src_root / "doc"
    short_intro = (doc_dir / "short-intro").read_text().strip() \
        if (doc_dir / "short-intro").exists() else ""
    kind = (doc_dir / "kind").read_text().strip() \
        if (doc_dir / "kind").exists() else ""
    intro = (doc_dir / "intro.dox").read_text() \
        if (doc_dir / "intro.dox").exists() else ""
    results = (doc_dir / "results.dox").read_text() \
        if (doc_dir / "results.dox").exists() else ""
    builds_on_raw = (doc_dir / "builds-on").read_text().strip() \
        if (doc_dir / "builds-on").exists() else ""
    builds_on = [b.strip() for b in builds_on_raw.split() if b.strip()]

    # Title: use the example dir name in Title Case (with a few exceptions)
    title = name.replace("-", " ").replace("_", " ")
    # Capitalise words but leave acronyms upper
    def cap(w):
        if w.upper() in {"CG", "ILU", "ILUT", "IR", "GMRES", "CUDA", "DPC++",
                         "MPI", "AMG", "BICGSTAB", "PAPI", "JSON", "YAML",
                         "GINKGO", "SPMV", "BLAS", "CB"}:
            return w.upper()
        return w[:1].upper() + w[1:]
    title = " ".join(cap(w) for w in title.split())

    lines = [f"# {title}\n"]
    if short_intro:
        lines.append(short_intro + "\n")

    # Metadata box
    meta_bits = []
    if kind:
        meta_bits.append(f"**Kind:** `{kind}`")
    if builds_on:
        link_list = ", ".join(f"[`{b}`]({b}.md)" for b in builds_on)
        meta_bits.append(f"**Builds on:** {link_list}")
    meta_bits.append(f"**Upstream source:** "
                     f"`examples/{name}/{name}.cpp` in the Ginkgo repository.")
    lines.append("  \n".join(meta_bits) + "\n")

    # Introduction (from intro.dox)
    if intro.strip():
        cleaned = strip_html(intro, BOILERPLATE_INTRO_PHRASES)
        if cleaned:
            lines.append("## Introduction\n")
            lines.append(cleaned + "\n")

    # The commented program
    lines.append("## The commented program\n")
    blocks = parse_cpp(cpp_path)
    for kind_, payload in blocks:
        if kind_ == "section":
            lines.append(f"### {payload}\n")
        elif kind_ == "prose":
            lines.append(payload + "\n")
        elif kind_ == "code":
            lines.append("```cpp")
            lines.append(payload)
            lines.append("```\n")

    # Results
    if results.strip():
        cleaned = strip_html(results, BOILERPLATE_RESULTS_PHRASES)
        if cleaned:
            lines.append("## Results\n")
            lines.append(cleaned + "\n")

    # Full source
    full_src = cpp_path.read_text()
    lines.append("## The plain program\n")
    lines.append("```{literalinclude} "
                 f"/../upstream/examples/{name}/{name}.cpp")
    lines.append(":language: cpp")
    lines.append("```\n")

    return "\n".join(lines)


def main(argv=None):
    global EXAMPLES_SRC, OUT_DIR
    ap = argparse.ArgumentParser(description="Generate the example pages.")
    ap.add_argument("--src", type=Path, default=EXAMPLES_SRC,
                    help="Ginkgo's examples/ directory")
    ap.add_argument("--out", type=Path, default=OUT_DIR,
                    help="directory the pages are written to")
    args = ap.parse_args(argv)
    EXAMPLES_SRC, OUT_DIR = args.src, args.out
    OUT_DIR.mkdir(exist_ok=True)
    names = []
    for d in sorted(EXAMPLES_SRC.iterdir()):
        if not d.is_dir() or d.name in SKIP:
            continue
        # Find the canonical .cpp file
        cpp_files = list(d.glob("*.cpp"))
        if not cpp_files:
            continue
        # Pick the cpp file whose name matches the directory (canonical case)
        cpp = next((c for c in cpp_files if c.stem == d.name), cpp_files[0])
        name = cpp.stem
        page = render_page(name, d)
        if not page:
            continue
        out = OUT_DIR / f"{name}.md"
        # Respect hand-polished pages: if the first 1KB contains the sentinel
        # "<!-- hand-polished -->", leave the file alone.
        if out.exists():
            head = out.read_text()[:1024]
            if "<!-- hand-polished -->" in head:
                names.append((name, d))
                print(f"kept (hand-polished) {out.relative_to(OUT_DIR.parent)}")
                continue
        out.write_text(page)
        names.append((name, d))
        print(f"wrote {out.relative_to(OUT_DIR.parent)}")

    # Build an index from the generated pages
    by_kind = {}
    for name, srcdir in names:
        kind = "uncategorised"
        if (srcdir / "doc" / "kind").exists():
            kind = (srcdir / "doc" / "kind").read_text().strip()
        intro = ""
        if (srcdir / "doc" / "short-intro").exists():
            intro = (srcdir / "doc" / "short-intro").read_text().strip()
        by_kind.setdefault(kind, []).append((name, intro))

    # Order of kinds in the index
    kind_order = ["basic", "preconditioners", "stopping-criteria",
                  "mixed-precision", "logging", "techniques",
                  "batched-functionality", "distributed", "uncategorised"]
    kind_titles = {
        "basic":                 "Basic solver setups",
        "preconditioners":       "Preconditioners",
        "stopping-criteria":     "Stopping criteria",
        "mixed-precision":       "Mixed precision",
        "logging":               "Logging and observability",
        "techniques":            "Techniques and integrations",
        "batched-functionality": "Batched functionality",
        "distributed":           "Distributed solvers",
        "uncategorised":         "Other examples",
    }

    # ---------------- Navigation graph ----------------
    # Build the builds-on graph as a mermaid block. Each example becomes a
    # clickable node; edges flow from the prerequisite (simpler) example to
    # the dependent (more complex) one, so following the arrows takes the
    # reader from "first thing you should look at" toward the more advanced
    # samples.
    KIND_CLASS = {
        "basic":                 "basic",
        "preconditioners":       "precond",
        "stopping-criteria":     "stop",
        "mixed-precision":       "mixed",
        "logging":               "logging",
        "techniques":            "tech",
        "batched-functionality": "batched",
        "distributed":           "distributed",
    }

    def mermaid_id(slug: str) -> str:
        # Mermaid node IDs prefer no leading digits / dashes. Underscore the
        # dashes; the display label stays human-readable.
        return slug.replace("-", "_")

    edges = []  # list of (prereq_slug, dependent_slug)
    for name, srcdir in names:
        bf = srcdir / "doc" / "builds-on"
        if not bf.exists():
            continue
        for prereq in bf.read_text().split():
            prereq = prereq.strip()
            if prereq:
                edges.append((prereq, name))

    # Flatter layout: drop subgraph wrappers (they force tall stacked layout
    # when there are many small clusters). Keep colour-coding by kind via
    # classDef so the categories are still visually distinct. TB direction
    # reads "simple at top, more complex below" — matching the prerequisite
    # arrow direction.
    mermaid = [
        "```{mermaid}",
        "%%{init: {'theme': 'base', "
        "'themeVariables': {'fontSize': '20px', 'fontFamily': 'system-ui'}, "
        "'flowchart': {'nodeSpacing': 40, 'rankSpacing': 60, "
        "'useMaxWidth': false, 'htmlLabels': true, 'padding': 16}}}%%",
        "graph TB",
    ]
    # Declare every node (so isolated ones still show up)
    for name, _ in names:
        mermaid.append(f"  {mermaid_id(name)}[{name}]")
    # Edges
    for prereq, dep in edges:
        mermaid.append(f"  {mermaid_id(prereq)} --> {mermaid_id(dep)}")
    # Click handlers
    for name, _ in names:
        mermaid.append(
            f'  click {mermaid_id(name)} "{name}.html" "Open {name}"')
    # Style classes (mermaid: classDef declares a class, class assigns it)
    style_defs = {
        "basic":       "fill:#dcfce7,stroke:#16a34a,color:#14532d",
        "precond":     "fill:#dbeafe,stroke:#2563eb,color:#1e3a8a",
        "stop":        "fill:#fef3c7,stroke:#d97706,color:#78350f",
        "mixed":       "fill:#fce7f3,stroke:#db2777,color:#831843",
        "logging":     "fill:#ede9fe,stroke:#7c3aed,color:#4c1d95",
        "tech":        "fill:#f1f5f9,stroke:#475569,color:#1e293b",
        "batched":     "fill:#ffe4e6,stroke:#e11d48,color:#881337",
        "distributed": "fill:#ffedd5,stroke:#ea580c,color:#7c2d12",
    }
    for cls, style in style_defs.items():
        mermaid.append(f"  classDef {cls} {style}")
    for kind, items in by_kind.items():
        cls = KIND_CLASS.get(kind)
        if not cls:
            continue
        for name, _ in items:
            mermaid.append(f"  class {mermaid_id(name)} {cls}")
    mermaid.append("```")

    idx = ["# Examples\n",
           ("Annotated walkthroughs of every example shipped under "
            "`examples/` in the Ginkgo source tree. Each page interleaves "
            "the source with explanatory commentary, lists the prerequisite "
            "examples it builds on, and reproduces the expected output. The "
            "full program is included at the bottom so you can copy it "
            "verbatim.\n"),
           ("## Navigation graph\n"),
           ("Edges flow from each example to the ones that build on it — "
            "follow the arrows to go from the simplest setups toward more "
            "advanced material. Click any node to jump to its page.\n"),
           "\n".join(mermaid) + "\n",
           ("## Browse by category\n")]

    for kind in kind_order:
        if kind not in by_kind:
            continue
        idx.append(f"### {kind_titles[kind]}\n")
        for name, intro in sorted(by_kind[kind]):
            tail = f" — {intro}" if intro else ""
            idx.append(f"- [{name}]({name}.md){tail}")
        idx.append("")

    # Hidden toctree (alphabetical, for sidebar nav)
    idx.append("```{toctree}")
    idx.append(":maxdepth: 1")
    idx.append(":hidden:")
    idx.append("")
    for name, _ in sorted(names):
        idx.append(name)
    idx.append("```")

    (OUT_DIR / "index.md").write_text("\n".join(idx) + "\n")
    print(f"wrote examples/index.md ({len(names)} pages)")


if __name__ == "__main__":
    main()
