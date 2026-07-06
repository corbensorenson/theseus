# Build instructions

The manuscript was built with Pandoc citation processing and rendered through LibreOffice.

## Requirements

- Pandoc with citeproc support
- An IEEE CSL file
- LibreOffice Writer for DOCX-to-PDF conversion
- Python 3 with Matplotlib to regenerate the figures

## Regenerate figures

```bash
python scripts/make_figures.py
```

## Build the DOCX

Set `CSL` to an IEEE CSL file on your system, then run:

```bash
CSL=/path/to/ieee.csl ./build.sh
```

The script writes outputs into `build/`. Page breaks can vary slightly across office-suite versions, so the checked-in files under `paper/` remain the authoritative Version 1.0 renderings.
