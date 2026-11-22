#!/usr/bin/env bash
# Compile the report (and slides) from their LaTeX sources.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v pdflatex >/dev/null 2>&1; then
  echo "pdflatex is required to rebuild report.pdf and slides.pdf" >&2
  exit 1
fi

pdflatex -interaction=nonstopmode -halt-on-error report.tex >/tmp/hybrid_report_pass1.log
pdflatex -interaction=nonstopmode -halt-on-error report.tex >/tmp/hybrid_report_pass2.log
if [ -f slides.tex ]; then
  pdflatex -interaction=nonstopmode -halt-on-error slides.tex >/tmp/hybrid_slides_pass1.log
  pdflatex -interaction=nonstopmode -halt-on-error slides.tex >/tmp/hybrid_slides_pass2.log
fi
echo "PDF BUILD COMPLETE"
