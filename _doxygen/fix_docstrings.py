#!/usr/bin/env python3
"""Doxygen INPUT_FILTER: repair known-broken doc comments in released headers.

Ginkgo 1.10 and 1.11 ship a few doc comments with an unbalanced backtick.
Doxygen reads the stray backtick as the start of an inline code span that
runs past the end of the comment, and the documented class (or the rest of
the file) disappears from the XML. Ginkgo develop fixed the text (#2095);
released tags cannot change, so this filter applies the same fixes while
Doxygen reads the headers.

Each fix matches only the broken text, so on corrected headers the filter
changes nothing. No fix adds or removes a line, keeping Doxygen's line
numbers intact.

Usage (Doxygen invokes it): fix_docstrings.py <input-file>
"""
import sys

FIXES = {
    "ginkgo/core/solver/ir.hpp": [
        (" * lambda - 1| < 1).", " * lambda - 1| < 1`)."),
    ],
    "ginkgo/core/base/polymorphic_object.hpp": [
        (
            "`PolymorphicObject to `AbstractObject`",
            "`PolymorphicObject` to `AbstractObject`",
        ),
        (
            " * ConvertibleTo\\<V\\>::convert_to(V*)` to do",
            " * `ConvertibleTo\\<V\\>::convert_to(V*)` to do",
        ),
    ],
    "ginkgo/core/base/range.hpp": [
        (
            "     * `Returns a reference to the accessor.",
            "     * Returns a reference to the accessor.",
        ),
    ],
}


def main():
    path = sys.argv[1]
    with open(path, encoding="utf-8") as f:
        text = f.read()
    normalized = path.replace("\\", "/")
    for suffix, fixes in FIXES.items():
        if normalized.endswith(suffix):
            for old, new in fixes:
                text = text.replace(old, new)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
