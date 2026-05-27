# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the repository for "FastSpecFit: Spectrophotometric Modeling of Extragalactic Targets from the DESI Early Data Release and Data Release 1" — a scientific paper (targeting *The Astronomical Journal*) by John Moustakas (Siena College). The paper documents [FastSpecFit](https://fastspecfit.readthedocs.io), an open-source Python code for modeling DESI spectra and broadband photometry using stellar continuum and emission-line templates.

**Data release context:** The primary analysis targets DESI DR2 (to be made public in early 2027), but the public NERSC directory structure is already in place. Comparisons to EDR and DR1 FastSpecFit VACs will appear in the paper's appendix.

## Repository Structure

```
fastpaper1/
├── tex/
│   ├── ms.tex          # main manuscript (AASTeX 6.3.1, submitting to AJ)
│   ├── refs.bib        # BibTeX bibliography (managed with BibDesk)
│   ├── figures/        # version-controlled final figures (committed after generation)
│   ├── tables/         # version-controlled final LaTeX table files
│   ├── aastex701.cls   # AAS journal class
│   └── aasjournalv7.bst
├── code/               # all analysis: standalone scripts + Jupyter notebooks
│   ├── fastspec-repeats        # repeat-observation fitting and QA
│   ├── fluxivar-sims           # flux uncertainty simulations
│   ├── prospector-modeling     # Prospector SED fitting (comparison)
│   ├── compare-mstar.ipynb     # stellar mass comparison notebook
│   └── compare-vac-versions.ipynb
├── data/               # small reference files for Zenodo (large data lives on NERSC)
├── environment.yml     # conda environment for reproducibility
├── README.md
└── CLAUDE.md
```

## Building the Paper

```bash
cd tex
pdflatex ms.tex
bibtex ms
pdflatex ms.tex
pdflatex ms.tex
```

The bibliography file is `refs.bib`; `ms.tex` references it as `\bibliography{refs}`.

## Running the Analysis Scripts

All scripts run on NERSC (Perlmutter) and require the DESI software environment and data paths (`$DESI_ROOT`, `/pscratch/`, `/global/cfs/cdirs/desi/`). Each script has a docstring with the exact invocation. The scripts follow a pipeline pattern driven by `argparse` flags — run steps sequentially: build parent sample → run FastSpecFit → merge outputs → QA plots. Each script documents the exact command in its module docstring.

```bash
# Repeat-observation fitting (code/fastspec-repeats)
./code/fastspec-repeats --build-parent
./code/fastspec-repeats --fastspec-cumulative --mp 8
./code/fastspec-repeats --merge
./code/fastspec-repeats --qa

# Flux uncertainty simulations (code/fluxivar-sims)
./code/fluxivar-sims --sims --niter 5000 --snr 10
./code/fluxivar-sims --qa --snr 10

# Prospector SED fitting comparison (code/prospector-modeling)
python code/prospector-modeling --priors delayedtau --sedfit --verbose --mp 12
python code/prospector-modeling --priors continuitysfh --qaplots --verbose --mp 12
```

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
- Measures: stellar velocity dispersions, emission-line fluxes/EWs for 40+ lines, K-corrections, rest-frame magnitudes, stellar masses (`LOGMSTAR`), SFRs, light-weighted ages, dust attenuation (`AV`)
- SPS models: FSPS with C3K stellar library and MIST isochrones; Chabrier IMF; non-parametric SFH with 5 age bins (0–30 Myr, 30–100 Myr, 0.1–1.1 Gyr, 1.1–11.6 Gyr, 11.6–13.7 Gyr); 8 dust values → 40 templates total
- DESI data releases in scope: DR2 (primary), EDR and DR1 (appendix comparisons)
- NERSC specprod names: "fuji" = EDR, "iron" = DR1; DR2 name TBD
