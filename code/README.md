# Figure Generation

All figures are written to `tex/figures/` and can be committed directly into the repo for inclusion in the paper.

All scripts run on NERSC (Perlmutter) where the DESI software environment and VAC paths are available. Scripts in this directory are imported as modules by `build-figures.py`; analysis notebooks (`*.ipynb`) are for exploration and are not part of the production figure pipeline.

## Environment

```bash
conda activate fastpaper1
```

## build-figures.py

The primary figure-generation script. Run from the repo root:

```bash
python code/build-figures.py --help
```

### Stellar mass comparison

Compares `LOGMSTAR` from the fastspec and fastphot VACs. Default uses the sv3 (SV3/one-percent) catalogs, which are compact single files. Pass `--main` to use the full main-survey catalogs (larger, split into 12 nside=1 healpix files). All masses use **h=1, Chabrier IMF (0.1–100 M☉)**.

Diagonal panels always show per-target-class colored histograms (BGS/LRG/ELG/Other). Off-diagonal panels default to the all-objects Hess diagram; use `--split-contours` to also split the contours by target class.

```bash
# default: sv3, all-objects Hess + per-class diagonal histograms
python code/build-figures.py --compare-mstar [--specprod loa] [--verbose]

# per-class contours on off-diagonal panels too
python code/build-figures.py --compare-mstar --split-contours [--specprod loa] [--verbose]

# full main survey
python code/build-figures.py --compare-mstar --main [--specprod loa] [--verbose]
```

Outputs:
- `tex/figures/compare-mstar-sv3.png` — default (per-class diagonal, all-objects Hess)
- `tex/figures/compare-mstar-sv3-split.png` — per-class contours (`--split-contours`)
- `tex/figures/compare-mstar-main.png` / `compare-mstar-main-split.png` (with `--main`)

---

## Other scripts

### fastspec-repeats

Fits repeat observations to assess redshift and spectrophotometric reproducibility (Appendix).

```bash
./code/fastspec-repeats --build-parent
./code/fastspec-repeats --fastspec-cumulative --mp 8
./code/fastspec-repeats --merge
./code/fastspec-repeats --qa
```

### fluxivar-sims

Simulations to validate flux uncertainty estimates.

```bash
./code/fluxivar-sims --sims --niter 5000 --snr 10
./code/fluxivar-sims --qa --snr 10
```

### prospector-modeling

Prospector SED fitting for comparison against FastSpecFit stellar masses (Appendix).

```bash
python code/prospector-modeling --priors delayedtau --sedfit --verbose --mp 12
python code/prospector-modeling --priors continuitysfh --qaplots --verbose --mp 12
```
