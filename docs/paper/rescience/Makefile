# ReScience C submission build. The upstream template uses latexmk+xelatex;
# this box has tectonic (an xetex engine) and biber, which cover the same
# ground. metadata.tex is generated from metadata.yaml, and generated/ plus
# references.bib are copied in from the parent paper so both articles input
# identical evidence.

VENV := $(shell git rev-parse --show-toplevel)/.venv

.PHONY: article generated clean

article: metadata.tex generated
	tectonic article.tex

metadata.tex: metadata.yaml
	$(VENV)/bin/python yaml-to-latex.py -i metadata.yaml -o metadata.tex

generated:
	$(MAKE) -C .. generated
	rm -rf generated && cp -r ../generated generated
	cp ../references.bib references.bib

clean:
	rm -rf generated references.bib metadata.tex article.pdf *.aux *.log *.bbl *.bcf *.run.xml
