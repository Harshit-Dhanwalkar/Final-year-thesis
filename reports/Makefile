# --- Configuration ---
TARGET    = project-update
TEMP_DIR  = temp
LATEX     = pdflatex
BIBER     = biber
FLAGS     = -interaction=nonstopmode -synctex=1 -output-directory=$(TEMP_DIR)

.PHONY: all main quick slides view clean clean-temp help

# Default: Compile main without slides
all: main

# Compile main only
main: $(TEMP_DIR)/$(TARGET).pdf
	cp $(TEMP_DIR)/$(TARGET).pdf ./

# Full compilation with everything
full: main slides

# Compilation rule - outputs to temp directory
$(TEMP_DIR)/$(TARGET).pdf: $(TARGET).tex
	mkdir -p $(TEMP_DIR)
	$(LATEX) $(FLAGS) $(TARGET).tex
	$(BIBER) --input-directory $(TEMP_DIR) --output-directory $(TEMP_DIR) $(TARGET)
	$(LATEX) $(FLAGS) $(TARGET).tex
	$(LATEX) $(FLAGS) $(TARGET).tex

# Quick compile without bibliography
quick:
	mkdir -p $(TEMP_DIR)
	$(LATEX) $(FLAGS) $(TARGET).tex
	cp $(TEMP_DIR)/$(TARGET).pdf ./

# Compile slides only
slides:
	@echo "Compiling slides from the slides/ directory..."
	@mkdir -p $(TEMP_DIR)
	@for file in slides/hov-*.tex; do \
		base=$$(basename $$file .tex); \
		$(LATEX) -interaction=nonstopmode -output-directory=$(TEMP_DIR) $$file; \
		cp $(TEMP_DIR)/$$base.pdf slides/; \
	done

# View the PDF
view:
	zathura ./$(TARGET).pdf &

# Clean everything
clean:
	rm -rf $(TEMP_DIR)
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.bbl *.blg *.bcf *.run.xml *.synctex.gz *.fdb_latexmk *.fls
	rm -f $(TARGET).pdf

# Clean only temporary files
clean-temp:
	rm -rf $(TEMP_DIR)
	rm -f *.aux *.log *.nav *.out *.snm *.toc *.bbl *.blg *.bcf *.run.xml *.synctex.gz *.fdb_latexmk *.fls

help:
	@echo "Usage:"
	@echo "  make         or make main : Compile main only"
	@echo "  make full                 : Compile everything (main + slides)"
	@echo "  make quick                : Quick compile (pdflatex only)"
	@echo "  make slides               : Compile slides only"
	@echo "  make view                 : Opens the main PDF in Zathura"
	@echo "  make clean                : Deletes everything including PDF"
	@echo "  make clean-temp           : Deletes only temporary files (keeps PDF)"
