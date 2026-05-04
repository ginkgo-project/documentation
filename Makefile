PYTHON := .venv/bin/python
PIP    := .venv/bin/pip
SPHINX := .venv/bin/sphinx-build

.PHONY: venv install html stable develop clean linkcheck strict

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
