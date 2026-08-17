# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the repository for "FastSpecFit: Spectrophotometric Modeling of Extragalactic Targets from the DESI Early Data Release and Data Release 1" — a scientific paper (targeting *The Astronomical Journal*) by John Moustakas (Siena University). The paper documents [FastSpecFit](https://fastspecfit.readthedocs.io), an open-source Python code for modeling DESI spectra and broadband photometry using stellar continuum and emission-line templates.

**Data release context:** The primary analysis targets DESI DR2 (to be made public in early 2027), but the public NERSC directory structure is already in place. Comparisons to EDR and DR1 FastSpecFit VACs will appear in the paper's appendix.

## Repository Structure

```
fastpaper1/
├── tex/
│   ├── ms.tex          # main manuscript (AASTeX 6.3.1, submitting to AJ)
│   ├── refs.bib        # BibTeX bibliography (managed with BibDesk)
│   ├── figures/        # version-controlled final figures (committed after generation)
│   ├── tables/         # version-controlled final LaTeX table files
│   ├── Makefile         # `make` runs the full lualatex+bibtex build
│   ├── aastex701.cls   # AAS journal class
│   └── aasjournalv7.bst
├── code/                    # figure-generation and QA scripts (see code/README.md)
│   ├── build-figures.py     # primary figure-generation entry point
│   ├── example-qa.py        # Section 4 illustrative-example QA figure
│   ├── linemasker-qa.py     # line-masking/patch-fitting QA figure (Section 3.3)
│   ├── prepare-external.py  # cross-match external comparison catalogs
│   └── util.py               # shared VAC-reading utilities
├── external/            # prepared external comparison catalogs (README tracked; FITS gitignored, see external/README.md)
├── data/                # small reference files for Zenodo (large data lives on NERSC)
├── environment.yml      # conda environment for reproducibility
├── README.md
└── CLAUDE.md
```

## Building the Paper

```bash
cd tex
make            # runs lualatex, bibtex, lualatex, lualatex (see tex/Makefile)
```

The bibliography file is `refs.bib`; `ms.tex` references it as `\bibliography{refs}`.

## Running the Analysis Scripts

Figure-generation and QA scripts run on NERSC (Perlmutter) and require the DESI software environment and data paths (`$DESI_ROOT`, `/pscratch/`, `/global/cfs/cdirs/desi/`). Each script documents its exact invocation in its module docstring; see `code/README.md` for the full flag reference.

```bash
# Figure generation (code/build-figures.py) — run from repo root
python code/build-figures.py --compare-mstar [--specprod loa] [--verbose]
python code/build-figures.py --compare-mstar --split-contours [--specprod loa]
python code/build-figures.py --compare-mstar --main [--specprod loa]

# Illustrative-example QA figure (Section 4) — single target, standalone fit, no completed catalog needed
python code/example-qa.py --redrockfile data/redrock-main-bright-15344.fits --outfile tex/figures/example-bgs.pdf

# Line-masking / patch-fitting QA figure (Section 3.3)
python code/linemasker-qa.py --redrockfile data/redrock-main-bright-17366.fits --outfile tex/figures/linemasker.pdf

# Prepare external comparison catalogs (cross-matched to the Loa FastSpecFit VAC)
python code/prepare-external.py --zouhu [--specprod loa|iron] [--verbose]
```

See `external/README.md` for details on each external comparison catalog (Zou/CIGALE, Siudek/CIGALE-AGN, Salim/GSWLC-X2, Weaver/COSMOS2020, Ross/fundamental-plane).

## Environment Setup

```bash
conda env create -f environment.yml
conda activate fastpaper1
```

`python-fsps` requires a special compile flag for the C3K stellar library:
```bash
FFLAGS="-DMILES=0 -DC3K=1" pip install fsps --no-binary fsps
```

## Key Python Dependencies

- `fastspecfit` — the code being documented; provides `fastspec` CLI and `fastspecfit.{io,util,mpi}`
- `desitarget` — DESI target selection and masking (`bgs_mask`, `geomask`, etc.)
- `fitsio`, `astropy` — FITS I/O and table handling
- `numpy`, `scipy`, `matplotlib`, `seaborn` — scientific computing stack
- `prospect` + `sedpy` — Prospector SED fitting (appendix comparison only)

## FastSpecFit Domain Context

- Two fitting modes: `fastspec` (spectra + photometry) and `fastphot` (photometry only)
- Measures: stellar velocity dispersions, emission-line fluxes/EWs for 40+ lines, K-corrections, rest-frame magnitudes, stellar masses (`LOGMSTAR`), SFRs, light-weighted ages, dust attenuation (`TAUV`)
- SPS models: FSPS with C3K stellar library and MIST isochrones; Chabrier IMF; non-parametric SFH with 5 age bins (0–30 Myr, 30–100 Myr, 0.1–1.1 Gyr, 1.1–11.6 Gyr, 11.6–13.7 Gyr); 8 dust values → 40 templates total
- DESI data releases in scope: DR2 (primary), EDR and DR1 (appendix comparisons)
- NERSC specprod names: "fuji" = EDR, "iron" = DR1, "loa" = DR2 (confirmed in `code/util.py`, `DEFAULT_SPECPROD = 'loa'`)
