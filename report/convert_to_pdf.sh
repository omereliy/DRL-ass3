#!/bin/bash
#
# Convert Markdown Report to PDF
# DRL Assignment 3 - Meta and Transfer Learning
#
# Usage: ./convert_to_pdf.sh [input.md] [output.pdf]
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT="${1:-$SCRIPT_DIR/DRL_Assignment3_Report.md}"
OUTPUT="${2:-$SCRIPT_DIR/DRL_Assignment3_Report.pdf}"

echo "============================================"
echo "DRL Assignment 3 - PDF Conversion"
echo "============================================"
echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo ""

# Check if input file exists
if [ ! -f "$INPUT" ]; then
    echo "Error: Input file not found: $INPUT"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Try pandoc first (preferred method)
if command_exists pandoc; then
    echo "Using pandoc for conversion..."

    # Check for PDF engine
    if command_exists pdflatex; then
        PDF_ENGINE="pdflatex"
    elif command_exists xelatex; then
        PDF_ENGINE="xelatex"
    elif command_exists lualatex; then
        PDF_ENGINE="lualatex"
    else
        echo "Warning: No LaTeX engine found. Trying wkhtmltopdf..."
        PDF_ENGINE=""
    fi

    if [ -n "$PDF_ENGINE" ]; then
        echo "PDF Engine: $PDF_ENGINE"
        pandoc "$INPUT" \
            -o "$OUTPUT" \
            --pdf-engine="$PDF_ENGINE" \
            -V geometry:margin=1in \
            -V fontsize=11pt \
            -V documentclass=article \
            -V colorlinks=true \
            -V linkcolor=blue \
            -V urlcolor=blue \
            --highlight-style=tango \
            --toc \
            --toc-depth=2 \
            -H <(echo '\usepackage{float}') \
            -H <(echo '\let\origfigure\figure') \
            -H <(echo '\let\endorigfigure\endfigure') \
            -H <(echo '\renewenvironment{figure}[1][2]{\expandafter\origfigure\expandafter[H]}{\endorigfigure}') \
            2>/dev/null || pandoc "$INPUT" -o "$OUTPUT" --pdf-engine="$PDF_ENGINE" -V geometry:margin=1in
        echo "Success! PDF created: $OUTPUT"
        exit 0
    fi
fi

# Try wkhtmltopdf as fallback
if command_exists wkhtmltopdf; then
    echo "Using wkhtmltopdf for conversion..."

    # First convert MD to HTML
    if command_exists pandoc; then
        TEMP_HTML=$(mktemp).html
        pandoc "$INPUT" -o "$TEMP_HTML" --standalone --metadata title="DRL Assignment 3"
        wkhtmltopdf --enable-local-file-access "$TEMP_HTML" "$OUTPUT"
        rm -f "$TEMP_HTML"
        echo "Success! PDF created: $OUTPUT"
        exit 0
    fi
fi

# Try Python markdown-pdf as another fallback
if command_exists python3; then
    echo "Attempting Python-based conversion..."

    python3 << 'PYTHON_SCRIPT'
import sys
import subprocess

# Try to use markdown and weasyprint
try:
    import markdown
    from weasyprint import HTML, CSS

    with open(sys.argv[1] if len(sys.argv) > 1 else 'DRL_Assignment3_Report.md', 'r') as f:
        md_content = f.read()

    html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            h1 {{ color: #333; border-bottom: 2px solid #333; }}
            h2 {{ color: #444; border-bottom: 1px solid #ccc; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f4f4f4; }}
            code {{ background-color: #f4f4f4; padding: 2px 5px; }}
            pre {{ background-color: #f4f4f4; padding: 10px; overflow-x: auto; }}
            img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>{html_content}</body>
    </html>
    """

    output = sys.argv[2] if len(sys.argv) > 2 else 'DRL_Assignment3_Report.pdf'
    HTML(string=full_html).write_pdf(output)
    print(f"Success! PDF created: {output}")
    sys.exit(0)

except ImportError as e:
    print(f"Python packages not available: {e}")
    sys.exit(1)
PYTHON_SCRIPT

    if [ $? -eq 0 ]; then
        exit 0
    fi
fi

# If all methods fail, provide instructions
echo ""
echo "============================================"
echo "ERROR: No PDF conversion tool available"
echo "============================================"
echo ""
echo "Please install one of the following:"
echo ""
echo "Option 1 - Pandoc with LaTeX (Recommended):"
echo "  macOS:   brew install pandoc && brew install --cask mactex-no-gui"
echo "  Ubuntu:  sudo apt-get install pandoc texlive-xetex"
echo ""
echo "Option 2 - Pandoc with wkhtmltopdf:"
echo "  macOS:   brew install pandoc wkhtmltopdf"
echo "  Ubuntu:  sudo apt-get install pandoc wkhtmltopdf"
echo ""
echo "Option 3 - Python packages:"
echo "  pip install markdown weasyprint"
echo ""
echo "Option 4 - Use online converter:"
echo "  https://www.markdowntopdf.com/"
echo "  https://pandoc.org/try/"
echo ""
exit 1
