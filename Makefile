PYTHON := .venv/bin/python
PIP    := .venv/bin/pip
SPHINX := .venv/bin/sphinx-build

# Ginkgo source tree this docs branch documents. Example pages include
# their full programs from `../upstream/examples/`, so Doxygen reads the
# same sibling checkout by default. Check it out at the ref pinned in
# `.ginkgo-ref`; `make doxygen-xml` warns when it is not.
GINKGO_SOURCE_DIR ?= $(abspath ../upstream)
GINKGO_REF := $(shell cat .ginkgo-ref 2>/dev/null)

.PHONY: venv install html strict linkcheck clean check-upstream doxygen-xml doxygen-xml-clean

venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -r requirements.txt

# The version shown in the switcher comes from `.ginkgo-ref`; set
# GINKGO_DOC_VERSION to override it.
html:
	$(SPHINX) -b html . _build/html

# Same flags as the PR check.
strict:
	$(SPHINX) -W --keep-going -b html . _build/html

linkcheck:
	$(SPHINX) -b linkcheck . _build/linkcheck

clean:
	rm -rf _build

check-upstream:
	@scripts/check_upstream.sh "$(GINKGO_SOURCE_DIR)" "$(GINKGO_REF)"

# Generate Doxygen XML from a Ginkgo source checkout. Required before any
# Sphinx build that includes Breathe directives (`{doxygenclass}` etc.).
# Without this step, those pages emit warnings but the rest of the site
# still builds.
doxygen-xml: check-upstream
	@if [ ! -d "$(GINKGO_SOURCE_DIR)/include" ]; then \
		echo "GINKGO_SOURCE_DIR=$(GINKGO_SOURCE_DIR) has no include/ — point it at a Ginkgo source checkout."; \
		exit 1; \
	fi
	@command -v doxygen >/dev/null || { echo "doxygen not found in PATH — install it first."; exit 1; }
	cd _doxygen && GINKGO_SOURCE_DIR=$(GINKGO_SOURCE_DIR) doxygen Doxyfile

doxygen-xml-clean:
	rm -rf _doxygen/xml
