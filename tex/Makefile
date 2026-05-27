PAPER = ms

.PHONY: all clean

all:
	lualatex $(PAPER)
	bibtex $(PAPER)
	lualatex $(PAPER)
	lualatex $(PAPER)

clean:
	rm -f $(shell ls $(PAPER).* | grep -Ev '\.(tex|bib|pdf)$$')
