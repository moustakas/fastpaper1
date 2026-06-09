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

Compares `LOGMSTAR` from the fastspec and fastphot DR2/Loa v1.0 VACs for all main-survey targets (BGS from main-bright; LRG/ELG/QSO from main-dark). All masses use **h=1, Chabrier IMF (0.1–100 M☉)**.

```bash
python code/build-figures.py --compare-mstar [--verbose]
```

Output: `tex/figures/compare-mstar.png`

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
