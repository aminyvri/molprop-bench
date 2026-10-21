.PHONY: install test download audit run report all quick

install:
	pip install -e ".[dev]"

test:
	pytest -q

download:
	molprop download

audit:
	molprop audit

run:
	molprop run

report:
	molprop report

# full pipeline (about an hour on a laptop CPU, mostly the random forests)
all: download audit run report

# 2 seeds and smaller forests, for a quick sanity check
quick:
	molprop download
	molprop audit
	molprop run --quick
	molprop report
