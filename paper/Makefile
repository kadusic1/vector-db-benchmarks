MAIN = main
BUILD_DIR = build
TEX_FILES = $(wildcard *.tex)
cd := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

.PHONY: all clean clean-all watch

all: $(BUILD_DIR)/$(MAIN).pdf

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/$(MAIN).pdf: $(TEX_FILES) references.bib | $(BUILD_DIR)
	cd $(cd) && pdflatex --interaction=batchmode --output-directory=$(BUILD_DIR) $(MAIN)
	-cd $(cd) && bibtex -terse $(BUILD_DIR)/$(MAIN)
	cd $(cd) && pdflatex --interaction=batchmode --output-directory=$(BUILD_DIR) $(MAIN)
	cd $(cd) && pdflatex --interaction=batchmode --output-directory=$(BUILD_DIR) $(MAIN)

watch: all
	@echo "Pratim izmjene u .tex i .bib fajlovima... Pritisni [CTRL+C] za zaustavljanje."
	@while inotifywait -q -e modify $(TEX_FILES) references.bib; do \
		echo "Detektovana izmjena! Pokrećem ponovno kompajliranje..."; \
		$(MAKE) all; \
	done

clean:
	rm -rf $(BUILD_DIR) *.bak* indent.log

clean-all: clean
