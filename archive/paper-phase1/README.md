# LaTeX Paper Compiler

This directory contains tools for compiling the `.knowledge` system into a complete research paper.

## Structure

```
paper/
├── compile_paper.py        # Main compiler script
├── paper.tex               # Generated LaTeX document
├── sections/               # Section content files (written by researchers)
│   ├── intro_constitution_architecture.tex
│   └── experiments_discoveries_discussion.tex
└── README.md               # This file
```

## Usage

### Generate the paper

```bash
python paper/compile_paper.py
```

This reads:
- `.knowledge/state.md` (project state)
- `.knowledge/constraints.json` (all constraints)
- `.knowledge/entries/*.json` (all experiment/discovery/decision entries)
- `CLAUDE.md` (the constitution)

And generates:
- `paper/paper.tex` (complete LaTeX document)

### Compile to PDF

```bash
cd paper
pdflatex paper.tex
```

You may need to run `pdflatex` twice to resolve references.

## What Gets Generated

### Auto-generated content:
- **Abstract**: Extracted from `state.md` current status
- **Methodology**: Extracted from entries tagged `methodology` or `protocol`
- **Constraints table**: All 19 active constraints from `constraints.json`
- **Frozen frame table**: Current 7 frozen elements
- **Bibliography**: 6 key references (Flow-Lenia, Neural CAs, Lenia, POET, Schmidhuber)

### Section files (written by researchers):
- `sections/intro_constitution_architecture.tex` - Introduction, constitution principles, architecture overview
- `sections/experiments_discoveries_discussion.tex` - Experimental results, key discoveries, discussion

If section files don't exist, placeholders are inserted with comments showing where to add `\input{}` commands.

## Recompiling

The script is **idempotent** - you can run it multiple times as the knowledge base updates:

```bash
# After adding new entries or updating state.md
python .knowledge/ingest.py
python .knowledge/compile.py
python paper/compile_paper.py

# Recompile PDF
cd paper
pdflatex paper.tex
```

## Section File Format

Researchers should write standalone `.tex` files that will be included via `\input{}`. Do NOT include `\documentclass`, `\begin{document}`, or preamble - just the section content.

Example `sections/intro_constitution_architecture.tex`:

```latex
\section{Introduction}

This research program explores...

\section{The Constitution}

The framework consists of five principles...

\subsection{Principle I: Computation Must Exist Without External Objectives}
...

\section{Architecture}

The Living Seed implementation...
```

## Dependencies

- Python 3.7+
- LaTeX distribution with `pdflatex` (TeX Live, MiKTeX, etc.)
- LaTeX packages: `geometry`, `amsmath`, `amssymb`, `booktabs`, `hyperref`, `graphicx`
