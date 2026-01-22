#!/usr/bin/env python3
"""
Convert Markdown Report to PDF
DRL Assignment 3 - Meta and Transfer Learning

Usage:
    python convert_to_pdf.py [input.md] [output.pdf]

Requirements:
    pip install markdown-pdf
"""

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DEFAULT_INPUT = SCRIPT_DIR / "DRL_Assignment3_Report.md"
DEFAULT_OUTPUT = SCRIPT_DIR / "DRL_Assignment3_Report.pdf"


def convert_with_markdown_pdf(input_path: Path, output_path: Path) -> bool:
    """Convert using markdown-pdf (PyMuPDF based, no external dependencies)."""
    try:
        from markdown_pdf import MarkdownPdf, Section

        print("Using markdown-pdf for conversion...")

        # Read markdown content
        md_content = input_path.read_text(encoding='utf-8')

        # Change to report directory for relative image paths
        original_dir = os.getcwd()
        os.chdir(input_path.parent)

        try:
            # Create PDF
            pdf = MarkdownPdf(toc_level=2)
            pdf.add_section(Section(md_content))
            pdf.meta["title"] = "DRL Assignment 3 - Meta and Transfer Learning"
            pdf.meta["author"] = "Omer Eliyahu"
            pdf.save(str(output_path.name))
        finally:
            os.chdir(original_dir)

        # Move file if needed
        temp_output = input_path.parent / output_path.name
        if temp_output != output_path and temp_output.exists():
            import shutil
            shutil.move(str(temp_output), str(output_path))

        return True

    except ImportError as e:
        print(f"markdown-pdf not available: {e}")
        print("Install with: pip install markdown-pdf")
        return False
    except Exception as e:
        print(f"markdown-pdf conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_with_pandoc(input_path: Path, output_path: Path) -> bool:
    """Convert using pandoc (external tool)."""
    try:
        import subprocess

        print("Using pandoc for conversion...")

        # Check if pandoc is available
        result = subprocess.run(['pandoc', '--version'], capture_output=True)
        if result.returncode != 0:
            return False

        # Try with different PDF engines
        for engine in ['xelatex', 'pdflatex', 'lualatex', 'wkhtmltopdf']:
            try:
                cmd = [
                    'pandoc', str(input_path),
                    '-o', str(output_path),
                    '-V', 'geometry:margin=1in',
                    '-V', 'fontsize=11pt',
                    '--highlight-style=tango'
                ]
                if engine != 'wkhtmltopdf':
                    cmd.extend([f'--pdf-engine={engine}'])

                subprocess.run(cmd, check=True, capture_output=True)
                return True
            except subprocess.CalledProcessError:
                continue

        return False

    except FileNotFoundError:
        print("pandoc not found")
        return False
    except Exception as e:
        print(f"pandoc conversion failed: {e}")
        return False


def main():
    # Parse arguments
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_INPUT
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT

    # Make paths absolute
    input_path = input_path.resolve()
    output_path = output_path.resolve()

    print("=" * 50)
    print("DRL Assignment 3 - PDF Conversion")
    print("=" * 50)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print()

    # Check input exists
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    # Try conversion methods in order
    converters = [
        ("markdown-pdf", convert_with_markdown_pdf),
        ("pandoc", convert_with_pandoc),
    ]

    for name, converter in converters:
        if converter(input_path, output_path):
            print()
            print(f"Success! PDF created: {output_path}")

            # Check page count
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(str(output_path))
                page_count = len(doc)
                doc.close()
                print(f"Page count: {page_count} (max allowed: 6)")
                if page_count > 6:
                    print("WARNING: Report exceeds 6 page limit!")
            except:
                pass

            sys.exit(0)

    # If all methods fail
    print()
    print("=" * 50)
    print("ERROR: No conversion method available")
    print("=" * 50)
    print()
    print("Please install markdown-pdf:")
    print("  pip install markdown-pdf")
    print()
    print("Or use pandoc:")
    print("  brew install pandoc mactex  # macOS")
    print()
    sys.exit(1)


if __name__ == "__main__":
    main()
