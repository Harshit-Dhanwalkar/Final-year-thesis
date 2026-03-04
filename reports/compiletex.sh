#!/usr/bin/env bash

set -e  # Exit on error

AUX_DIR="aux"
MAIN_FILE="main.tex"
ONLY_MAIN=false
VERBOSE=false
CLEANUP=true
DRY_RUN=false

# List of auxiliary extensions to move to aux directory
EXTS=(
    "aux" "fdb_latexmk" "fls" "log" "nav" "out" "snm" "synctex.gz"
    "toc" "lof" "lot" "lol" "bcf" "bbl" "blg" "run.xml"
    "xdv" "dvi" "ps" "bcf" "mtc" "mtc0" "maf"
    "idx" "ilg" "ind" "acn" "acr" "alg" "glg" "glo" "gls"
    "ist" "slg" "syg" "syi" "synctex"
)

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]   ${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]   ${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]  ${NC} $1"; }
log_debug() { [ "$VERBOSE" = true ] && echo -e "${YELLOW}[DEBUG]  ${NC} $1"; }

debug_list_files() {
    local dir=$1
    local base=$2
    log_debug "Checking files for: $dir/$base"
    for ext in "${EXTS[@]}"; do
        local aux_file="$dir/$base.$ext"
        if [ -f "$aux_file" ]; then
            log_debug "  Found: $base.$ext"
        fi
    done
}

# Usage Information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

OPTIONS:
    -m              Compile main file only (skip slides)
    -v              Verbose output
    -n              No cleanup (keep aux files in place)
    -d              Dry run (show what would happen, don't compile)
    -h              Show this help message

EXAMPLES:
    $0              # Compile both main and slides, clean up aux files
    $0 -m           # Compile main only
    $0 -d -m        # Dry run main compilation

EOF
    exit 0
}

while getopts "mfvndh" opt; do
    case $opt in
        m) ONLY_MAIN=true ;;
        v) VERBOSE=true ;;
        n) CLEANUP=false ;;
        d) DRY_RUN=true ;;
        h) usage ;;
        *)
            log_error "Invalid option: -$OPTARG"
            usage
            ;;
    esac
done

echo ""
echo "╔═════════════════════════════════════════════════════════╗"
echo "║   Monsoon Depression LaTeX Compilation Script           ║"
echo "╚═════════════════════════════════════════════════════════╝"

# Setup
# Verify main file exists
if [ ! -f "$MAIN_FILE" ]; then
    log_error "Main file not found : $MAIN_FILE"
    exit 1
fi

log_info "Main file      : $MAIN_FILE"
# log_info "Using pdflatex compiler"

if [ "$DRY_RUN" = true ]; then
    log_warn "DRY RUN MODE - No actual compilation will occur"
fi

# Create aux directory if cleanup enabled
if [ "$CLEANUP" = true ]; then
    if [ "$DRY_RUN" = false ]; then
        mkdir -p "$AUX_DIR"
    fi
    log_info "Aux files will be moved to: $AUX_DIR/"
else
    log_warn "Cleanup disabled - aux files will remain in place"
fi

# Cleanup Function to Move Auxiliary Files
cleanup_aux() {
    local dir=$1
    local base=$2
    local moved_count=0

    if [ "$CLEANUP" = false ]; then
        log_debug "Cleanup disabled for $base"
        return
    fi

    log_debug "Starting cleanup for: $base in directory: $dir"
    debug_list_files "$dir" "$base"

    if [ "$DRY_RUN" = false ]; then
        mkdir -p "$AUX_DIR"
    fi

    for ext in "${EXTS[@]}"; do
        local aux_file="$dir/$base.$ext"
        local clean_dir="${dir#./}"

        if [ "$clean_dir" = "." ] || [ -z "$clean_dir" ]; then
            # Files in current directory
            local dest_file="$AUX_DIR/$base.$ext"
        else
            # Files in subdirectories
            local prefix="${clean_dir//\//_}"
            local dest_file="$AUX_DIR/${prefix}_$base.$ext"
        fi

        if [ -f "$aux_file" ]; then
            if [ "$DRY_RUN" = false ]; then
                mv "$aux_file" "$dest_file"
                moved_count=$((moved_count + 1))
                log_debug "  Moved: $base.$ext -> $(basename $dest_file)"
            else
                log_debug "  [DRY RUN] Would move: $base.$ext"
            fi
        fi
    done

    # Move additional auxiliary files (including from included PDFs)
    if [ "$dir" = "." ]; then
        for ext in "${EXTS[@]}"; do
            for file in *.$ext; do
                if [ -f "$file" ]; then
                    if [ "$file" != "$base.$ext" ] && [ ! -f "$AUX_DIR/$file" ]; then
                        if [ "$DRY_RUN" = false ]; then
                            mv "$file" "$AUX_DIR/"
                            moved_count=$((moved_count + 1))
                            log_debug "  Moved extra: $file"
                        else
                            log_debug "  [DRY RUN] Would move extra: $file"
                        fi
                    fi
                fi
            done
        done

        # Look for files without extensions that might be auxiliary
        for file in *; do
            if [ -f "$file" ] && [[ "$file" == *.* ]]; then
                # Already handled by extension loop
                continue
            elif [ -f "$file" ] && [ ! -x "$file" ]; then
                if [[ "$file" == *.aux ]] || [[ "$file" == *.log ]] || [[ "$file" == *.bcf ]]; then
                    if [ ! -f "$AUX_DIR/$file" ]; then
                        if [ "$DRY_RUN" = false ]; then
                            mv "$file" "$AUX_DIR/"
                            moved_count=$((moved_count + 1))
                            log_debug "  Moved extra: $file"
                        else
                            log_debug "  [DRY RUN] Would move extra: $file"
                        fi
                    fi
                fi
            fi
        done
    fi

    if [ "$DRY_RUN" = false ] && [ $moved_count -gt 0 ]; then
        log_debug "  Total moved : $moved_count auxiliary files for $base"
    fi
}

# Compilation
compile_tex() {
    local status=0
    local file=$1
    local dir=$(dirname "$file")
    local base=$(basename "$file" .tex)
    local output_dir="${dir:-.}"

    if [ ! -f "$file" ]; then
        log_warn "File not found, skipping: $file"
        return 1
    fi
    log_info "Compiling     : $file"
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would run : pdflatex -interaction=nonstopmode -output-directory=\"$dir\" \"$file\""
        return 0
    fi

    # Pass 1: Generate the .bcf file for Biber
    log_info "-> Pass 1     : pdflatex"
    if ! pdflatex -interaction=nonstopmode -output-directory="$output_dir" "$file" > /dev/null 2>&1; then
        log_error "pdflatex pass 1 failed for $file"
        status=1
    fi

    # Pass 2: Run Biber
    log_info "-> Pass 2     : biber"
    local biber_status=0
    if [ "$VERBOSE" = true ]; then
        biber --input-directory "$output_dir" --output-directory "$output_dir" "$base"
        biber_status=$?
    else
        biber --input-directory "$output_dir" --output-directory "$output_dir" "$base" > /dev/null 2>&1
        biber_status=$?
    fi
    if [ $biber_status -ne 0 ]; then
        log_error "Biber failed for $file. Check .bib keys!"
        status=1
    fi

    # Link citations and fix labels
    log_info "-> Pass 3 & 4 : pdflatex finalization"
    if ! pdflatex -interaction=nonstopmode -output-directory="$output_dir" "$file" > /dev/null 2>&1; then
        log_error "pdflatex pass 3 failed for $file"
        status=1
    fi
    if ! pdflatex -interaction=nonstopmode -output-directory="$output_dir" "$file" > /dev/null 2>&1; then
        log_error "pdflatex pass 4 failed for $file"
        status=1
    fi

    # Cleanup auxiliary files
    cleanup_aux "$output_dir" "$base"

    # Check if PDF was generated
    if [ -f "${dir:-.}/$base.pdf" ]; then
        if [ $status -eq 0 ]; then
            # log_success "Generated : ${dir:-.}/$base.pdf"
            : # success, null command
        else
            log_warn "Generated PDF with errors : ${dir:-.}/$base.pdf"
        fi
    else
        log_error "No PDF generated for $file"
        status=1
    fi

    return $status
}

# Main Execution
START_TIME=$(date +%s)
TOTAL_FILES=0
COMPILED_FILES=0
FAILED_FILES=0

# Compile Slides
if [ "$ONLY_MAIN" = false ]; then
    echo ""
    log_info "======== Compiling Slides ========"

    if [ -d "slides" ]; then
        hov_files=(slides/hov-*.tex)
        if [ ${#hov_files[@]} -gt 0 ] && [ -f "${hov_files[0]}" ]; then
            for slide in "${hov_files[@]}"; do
                if [ -f "$slide" ]; then
                    TOTAL_FILES=$((TOTAL_FILES + 1))
                    if compile_tex "$slide"; then
                        COMPILED_FILES=$((COMPILED_FILES + 1))
                        log_success "Compiled      : $slide"
                    else
                        FAILED_FILES=$((FAILED_FILES + 1))
                        log_error "Failed to compile: $slide"
                    fi
                fi
            done

            if [ $FAILED_FILES -eq 0 ]; then
                log_success "Completed slides compilation - all successful!"
            else
                log_warn "Completed slides compilation with $FAILED_FILES failure(s)"
            fi
        else
            log_error "No slides found in slides/ directory"
        fi
    else
        log_error "slides/ directory not found, skipping"
    fi
else
    log_warn "Skipping slides (main only mode)"
fi

# Compile Main Project Update
echo ""
log_info "====== Compiling Main Project Update ======="

TOTAL_FILES=$((TOTAL_FILES + 1))

if compile_tex "$MAIN_FILE"; then
    COMPILED_FILES=$((COMPILED_FILES + 1))
    log_success "Main project update compiled successfully"
else
    FAILED_FILES=$((FAILED_FILES + 1))
    log_error "Failed to compile main project update"
fi

# Summary Report
END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))
MINS=$((ELAPSED_TIME / 60))
SECS=$((ELAPSED_TIME % 60))

echo ""
echo "╔═════════════════════════════════════════════════════════╗"
echo "║                   Compilation Summary                   ║"
echo "╚═════════════════════════════════════════════════════════╝"

log_info "Total files processed : $TOTAL_FILES"
log_success "Successfully compiled : $COMPILED_FILES"

if [ $FAILED_FILES -gt 0 ]; then
    log_error "Failed compilations : $FAILED_FILES"
fi

if [ "$CLEANUP" = true ] && [ -d "$AUX_DIR" ]; then
    AUX_COUNT=$(find "$AUX_DIR" -type f 2>/dev/null | wc -l)
    log_info "Auxiliary files moved : $AUX_DIR/ ($AUX_COUNT files)"
fi

if [ "$DRY_RUN" = true ]; then
    log_warn "This was a DRY RUN - no actual compilation occurred"
fi

echo ""
log_info "Compilation time: ${MINS}m ${SECS}s"

if [ $FAILED_FILES -eq 0 ]; then
    log_success "All compilations completed successfully!"
    exit 0
else
    log_error "Some compilations failed - please check logs above"
    exit 1
fi
