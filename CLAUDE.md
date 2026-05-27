# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the repository for "FastSpecFit: Spectrophotometric Modeling of 3.6 Million Extragalactic Targets from the DESI Early Data Release and Data Release 1" — a scientific paper (targeting *The Astronomical Journal*) by John Moustakas (Siena College). The paper documents [FastSpecFit](https://fastspecfit.readthedocs.io), an open-source Python code for modeling DESI spectra and broadband photometry using stellar continuum and emission-line templates.

## Repository Structure

- `tex/` — LaTeX source for the paper
  - `ms.tex` — main manuscript (AASTeX 6.3.1 format, `\submitjournal{\aj}`)
  - `refs.bib` — BibTeX bibliography (note: `ms.tex` references `\bibliography{bib}`, so a symlink or rename to `bib.bib` may be needed for compilation)
  - `aastex701.cls`, `aasjournalv7.bst` — AAS journal class and bibliography style
- `code/` — standalone Python analysis scripts (executable, no `.py` extension)
  - `fastspec-repeats` — fits DESI Fuji repeat observations to assess repeatability (FastSpecFit issue #127)
  - `fluxivar-sims` — Monte Carlo simulations to characterize flux uncertainty estimation
  - `prospector-modeling` — Prospector SED fitting for comparison against FastSpecFit stellar masses
- `nb/` — Jupyter notebooks for exploratory analysis
  - `compare-mstar.ipynb` — compares FastSpecFit stellar masses against CIGALE, Prospector, and Kcorrect
  - `compare-vac-versions.ipynb` — compares VAC versions

## Building the Paper

```bash
cd tex
pdflatex ms.tex
bibtex ms
pdflatex ms.tex
pdflatex ms.tex
```

The bibliography file is `refs.bib` but `ms.tex` calls `\bibliography{bib}`, so you may need:
```bash
ln -s refs.bib bib.bib  # or rename refs.bib to bib.bib
```

## Running the Analysis Scripts

All scripts run on NERSC (Perlmutter) and require the DESI software environment and data paths (`$DESI_ROOT`, `/pscratch/`, `/global/cfs/cdirs/desi/`). Each script has a docstring with the exact invocation. Examples:

```bash
# Repeat-observation fitting
./code/fastspec-repeats --build-parent
./code/fastspec-repeats --fastspec-cumulative --mp 8

# Flux uncertainty simulations
./code/fluxivar-sims --sims --niter 5000 --snr 10
./code/fluxivar-sims --qa --snr 10

# Prospector SED fitting (comparison)
python code/prospector-modeling --priors delayedtau --sedfit --verbose --mp 12
python code/prospector-modeling --priors continuitysfh --qaplots --verbose --mp 12
```

## Key Python Dependencies

- `fastspecfit` — the code being documented; provides `fastspec` CLI, `fastspecfit.io`, `fastspecfit.util`, `fastspecfit.mpi`
- `desitarget` — DESI target selection and masking
- `fitsio`, `astropy` — FITS I/O and table handling
- `numpy`, `scipy`, `matplotlib`, `seaborn` — scientific computing stack
- `prospect` (Prospector) — independent SED fitting (comparison only)
- `sedpy` — filter curves (used with Prospector)

## Architecture of the Analysis Scripts

Each script follows a pipeline pattern driven by `argparse` flags (e.g., `--build-parent`, `--fastspec-cumulative`, `--merge`, `--qa`). Steps are typically run sequentially: build parent sample → run FastSpecFit → merge outputs → make QA plots. Multiprocessing is supported via `--mp N`.

Data products (FITS files) are written to paths under `$DESI_ROOT` or `/global/cfs/cdirs/desi/users/ioannis/fastspecfit/fastpaper1/`. Figures go to a `figures/` subdirectory within that tree. The notebooks read from the same paths.

## DESI/FastSpecFit Context

- FastSpecFit measures: stellar velocity dispersions, emission-line fluxes/EWs for 40+ lines, K-corrections, rest-frame magnitudes, stellar masses (`LOGMSTAR`), SFRs, ages, dust attenuation
- Two modes: `fastspec` (uses spectra + photometry) and `fastphot` (photometry only)
- SPS models: FSPS with C3K stellar library and MIST isochrones; Chabrier IMF; non-parametric SFH with 5 age bins
- The SPS templates must be compiled with `FFLAGS="-DMILES=0 -DC3K=1" python -m pip install fsps --no-binary fsps`
- Data releases: EDR ("Fuji" specprod) and DR1
