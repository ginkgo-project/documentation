PYTHON := .venv/bin/python
PIP    := .venv/bin/pip
SPHINX := .venv/bin/sphinx-build

# Default Ginkgo source location for the Breathe → Sphinx XML build. Override
# on the command line, e.g. `make doxygen-xml GINKGO_SOURCE_DIR=/path/to/ginkgo`.
GINKGO_SOURCE_DIR ?= $(abspath ../upstream)

.PHONY: venv install html stable develop clean linkcheck strict doxygen-xml doxygen-xml-clean

venv:
	python3 -m venv .venv
	$(PIP) install --upgrade pip

install: venv
	$(PIP) install -r requirements.txt

html:
	$(SPHINX) -b html . _build/html

stable:
	GINKGO_DOC_VERSION=stable $(SPHINX) -b html . _build/html

develop:
	GINKGO_DOC_VERSION=develop $(SPHINX) -b html . _build/html/develop

strict:
	$(SPHINX) -W -b html . _build/html

linkcheck:
	$(SPHINX) -b linkcheck . _build/linkcheck

clean:
	rm -rf _build

# Generate Doxygen XML from a Ginkgo source checkout. Required before any
# Sphinx build that includes Breathe directives (`{doxygenclass}` etc.).
# Without this step, those pages emit warnings but the rest of the site
# still builds.
doxygen-xml:
	@if [ ! -d "$(GINKGO_SOURCE_DIR)/include" ]; then \
		echo "GINKGO_SOURCE_DIR=$(GINKGO_SOURCE_DIR) has no include/ — point it at a Ginkgo source checkout."; \
		exit 1; \
	fi
	@command -v doxygen >/dev/null || { echo "doxygen not found in PATH — install it first."; exit 1; }
	cd _doxygen && GINKGO_SOURCE_DIR=$(GINKGO_SOURCE_DIR) doxygen Doxyfile

doxygen-xml-clean:
	rm -rf _doxygen/xml
