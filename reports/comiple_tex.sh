#!/usr/bin/env bash

START_TIME=$(date +%s)

TEMP_DIR="temp"
mkdir -p "$TEMP_DIR"

ONLY_MAIN=false
while getopts "m" opt; do
  case $opt in
    m) ONLY_MAIN=true ;;
    *) echo "Usage: $0 [-m (main only)]"; exit 1 ;;
  esac
done

# List of auxiliary extensions to move
EXTS=("aux" "fdb_latexmk" "fls" "log" "nav" "out" "snm" "synctex.gz" "toc")

cleanup_aux() {
    local dir=$1
    local base=$2
    for ext in "${EXTS[@]}"; do
        local aux_file="$dir/$base.$ext"
        if [ -f "$aux_file" ]; then
            local clean_dir=${dir#./}
            local prefix=${clean_dir//\//_}
            if [ -z "$prefix" ] || [ "$prefix" == "." ]; then
                mv "$aux_file" "$TEMP_DIR/$base.$ext"
            else
                mv "$aux_file" "$TEMP_DIR/${prefix}_$base.$ext"
            fi
        fi
    done
}

compile_tex() {
    local file=$1
    local dir=$(dirname "$file")
    local base=$(basename "$file" .tex)

    echo ">  [Compiling] $file"
    pdflatex -interaction=nonstopmode -output-directory="$dir" "$file" > /dev/null
    pdflatex -interaction=nonstopmode -output-directory="$dir" "$file" > /dev/null

    cleanup_aux "$dir" "$base"
}

if [ "$ONLY_MAIN" = false ]; then
    echo "--- Compiling Hovmoller Slides ---"
    for slide in hovmoller-slides/hov-*.tex; do
        if [ -f "$slide" ]; then
            compile_tex "$slide"
        fi
    done
else
    echo "--- Skipping Hovmoller Slides ---"
fi

echo "--- Compiling Main Project Update ---"
if [ -f "project-update.tex" ]; then
    compile_tex "project-update.tex"
fi

END_TIME=$(date +%s)
ELAPSED_TIME=$((END_TIME - START_TIME))

MINS=$((ELAPSED_TIME / 60))
SECS=$((ELAPSED_TIME % 60))

echo -e "\n Done! PDFs updated. Aux files moved to ./$TEMP_DIR"
echo "Total Compilation Time: ${MINS}m ${SECS}s"
