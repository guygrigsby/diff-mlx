# Build the paper from committed evidence. Same discipline as the agent-go
# paper: generated/ and main.pdf are produced, never hand-edited. Tables,
# macros, and figures come from gen.py over docs/paper/data/; the PDF from
# tectonic.

VENV := $(shell git rev-parse --show-toplevel)/.venv

.PHONY: paper generated arxiv clean

paper: generated
	tectonic main.tex

generated:
	$(VENV)/bin/python gen.py

# arXiv submission bundle. arXiv compiles the TeX source itself and does
# not run BibTeX, so the bundle carries main.bbl instead of references.bib.
# Standalone compile from the tarball verifies nothing is missing.
arxiv: generated
	tectonic --keep-intermediates main.tex
	tar czf arxiv.tar.gz main.tex main.bbl generated/*.tex
	@dir=$$(mktemp -d) && tar xzf arxiv.tar.gz -C $$dir && \
		(cd $$dir && tectonic main.tex >/dev/null 2>&1) && \
		echo "arxiv.tar.gz OK: standalone compile passed" || \
		{ echo "arxiv.tar.gz FAILED standalone compile"; exit 1; }

clean:
	rm -f main.pdf main.bbl main.aux main.log arxiv.tar.gz
