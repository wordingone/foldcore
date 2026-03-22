#!/usr/bin/env python3
"""Build publication-ready LaTeX from PAPER.md.

Usage: python build-paper.py [--pdf]
  Outputs: paper.tex (always), paper.pdf (if --pdf and xelatex/pdflatex available)
"""
import sys
import os
import subprocess

def md_to_latex(md_path: str, tex_path: str):
    """Convert PAPER.md to LaTeX via pypandoc."""
    import pypandoc

    extra_args = [
        "--standalone",
        "--from=markdown+yaml_metadata_block",
        "--template", os.path.join(os.path.dirname(__file__), "paper-template.tex"),
        "--citeproc" if os.path.exists("references.bib") else "--no-highlight",
        "--wrap=none",
        "-V", "geometry:margin=1in",
        "-V", "fontsize=11pt",
        "-V", "documentclass=article",
    ]
    # Filter out invalid args
    extra_args = [a for a in extra_args if a != "--no-highlight"]

    pypandoc.convert_file(
        md_path,
        "latex",
        outputfile=tex_path,
        extra_args=extra_args,
    )
    print(f"  {tex_path} written.")


def simple_convert(md_path: str, tex_path: str):
    """Fallback: simple markdown to LaTeX without template."""
    import pypandoc
    pypandoc.convert_file(
        md_path, "latex", outputfile=tex_path,
        extra_args=["--standalone", "-V", "geometry:margin=1in", "-V", "fontsize=11pt"]
    )
    print(f"  {tex_path} written.")


def compile_pdf(tex_path: str):
    """Compile .tex to .pdf."""
    # Try pytinytex first, then system compilers
    compilers = []
    try:
        import pytinytex
        compilers.append(pytinytex.get_pdflatex_engine())
    except Exception:
        pass
    compilers.extend(["xelatex", "pdflatex"])

    for compiler in compilers:
        try:
            result = subprocess.run(
                [compiler, "-interaction=nonstopmode", tex_path],
                capture_output=True, text=True, timeout=120,
                cwd=os.path.dirname(tex_path) or "."
            )
            if result.returncode == 0:
                print(f"  PDF compiled with {compiler}.")
                return True
            # pdflatex returns non-zero on warnings but still produces PDF
            pdf_path = tex_path.replace(".tex", ".pdf")
            if os.path.exists(os.path.join(os.path.dirname(tex_path) or ".", pdf_path)):
                print(f"  PDF compiled with {compiler} (with warnings).")
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
    print("  No LaTeX compiler found. Install texlive, miktex, or pip install pypandoc[tinytex].")
    return False


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    md = "PAPER.md"
    tex = "paper.tex"

    if not os.path.exists(md):
        print(f"Error: {md} not found.")
        sys.exit(1)

    try:
        md_to_latex(md, tex)
    except Exception as e:
        print(f"Template conversion failed ({e}), using simple conversion.")
        simple_convert(md, tex)

    if "--pdf" in sys.argv:
        compile_pdf(tex)
