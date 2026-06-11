#!/usr/bin/env python
"""Prepare external catalogs for cross-comparison with the FastSpecFit DR2 (Loa) VAC.

For each catalog a prepare_*() function reads the external file once, then loops
over (survey, program) combinations, cross-matches to the FastSpecFit reference
catalog (via util.read_fastspec), applies positional and redshift consistency
checks, standardizes units, and writes a compact prepared file to ./external/.

Output files are named:
    external/<catalog>/loa-<survey>-<program>.fits

Each output file contains the matched reference columns (TARGETID, RA, DEC, Z,
LOGMSTAR, SFR, TAUV, …) side-by-side with standardized external columns, all at
h=1 and Chabrier IMF, ready for direct comparison.

Usage (from repo root or code/):
    python code/prepare-external.py --cigale [--specprod loa] [--ntest N] [--verbose]
"""

import os, sys, argparse
import numpy as np
from astropy.table import Table
import fitsio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import read_fastspec, DEFAULT_SPECPROD

C_LIGHT       = 2.998e5  # km/s
MAX_DV_KMS    = 1000.0   # redshift-consistency threshold [km/s]
MAX_SEP_ARCSEC = 1.5     # positional-consistency threshold [arcsec]

SURVEY_PROGRAMS = [
    ('sv3',  'bright'),
    ('sv3',  'dark'),
    ('main', 'bright'),
    ('main', 'dark'),
]

REPODIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTDIR  = os.path.join(REPODIR, 'external')

# Base reference columns pulled from read_fastspec for every prepared file.
# Survey-specific targeting columns are appended per-survey; read_fastspec
# silently drops any requested column not present in the catalog.
_REF_BASE_COLS = [
    'TARGETID', 'RA', 'DEC', 'Z', 'ZWARN', 'SURVEY', 'PROGRAM', 'HEALPIX',
    'LOGMSTAR', 'SFR', 'TAUV', 'VDISP', 'DN4000',
]
_SV_TARGET_COLS = {
    'sv1':     ['SV1_DESI_TARGET', 'SV1_BGS_TARGET', 'SV1_MWS_TARGET'],
    'sv3':     ['SV3_DESI_TARGET', 'SV3_BGS_TARGET', 'SV3_MWS_TARGET'],
    'main':    ['DESI_TARGET',     'BGS_TARGET',      'MWS_TARGET'],
    'special': ['DESI_TARGET',     'BGS_TARGET',      'MWS_TARGET'],
}


def _ref_columns(survey):
    return _REF_BASE_COLS + _SV_TARGET_COLS.get(survey, [])


def _decode_str_col(col):
    """Return a stripped numpy str array from a fitsio fixed-length bytes column."""
    return np.array([
        v.decode('utf-8').strip() if isinstance(v, (bytes, np.bytes_)) else str(v).strip()
        for v in col
    ])


def cross_match(ref, ext, ext_z_col='Z', ext_ra_col='RA', ext_dec_col='DEC',
                verbose=False):
    """Match ref and ext on TARGETID, then check redshift and sky-position consistency.

    Parameters
    ----------
    ref : astropy.table.Table
        Reference FastSpecFit catalog; must have TARGETID, Z, RA, DEC.
    ext : astropy.table.Table
        External catalog; must have TARGETID and a redshift column.
    ext_z_col : str
        Redshift column name in ext.
    ext_ra_col, ext_dec_col : str
        RA/Dec column names in ext. Positional check is skipped if these are
        absent from ext.
    verbose : bool

    Returns
    -------
    i_ref, i_ext : ndarray of int
        Indices into ref and ext of matched, consistency-checked pairs.
    """
    ref_map = {int(tid): i for i, tid in enumerate(ref['TARGETID'])}

    i_ref_list, i_ext_list = [], []
    for j, tid in enumerate(ext['TARGETID']):
        k = ref_map.get(int(tid))
        if k is not None:
            i_ref_list.append(k)
            i_ext_list.append(j)

    if not i_ref_list:
        return np.array([], dtype=int), np.array([], dtype=int)

    i_ref = np.array(i_ref_list)
    i_ext = np.array(i_ext_list)

    # Redshift consistency: |Δv| < MAX_DV_KMS
    dv   = np.abs(ref['Z'][i_ref] - ext[ext_z_col][i_ext]) * C_LIGHT
    z_ok = dv < MAX_DV_KMS

    # Positional consistency: sep < MAX_SEP_ARCSEC (skip if ext lacks RA/DEC)
    has_radec = (ext_ra_col in ext.colnames) and (ext_dec_col in ext.colnames)
    if has_radec:
        from astropy.coordinates import SkyCoord
        import astropy.units as u_
        c_ref = SkyCoord(ref['RA' ][i_ref] * u_.deg, ref['DEC'][i_ref] * u_.deg)
        c_ext = SkyCoord(ext[ext_ra_col][i_ext] * u_.deg,
                         ext[ext_dec_col][i_ext] * u_.deg)
        sep    = c_ref.separation(c_ext).arcsec
        pos_ok = sep < MAX_SEP_ARCSEC
    else:
        pos_ok = np.ones(len(i_ref), dtype=bool)
        if verbose:
            print('    (external catalog has no RA/DEC; positional check skipped)')

    keep       = z_ok & pos_ok
    n_fail_z   = int((~z_ok ).sum())
    n_fail_pos = int((~pos_ok).sum())
    if verbose or n_fail_z or n_fail_pos:
        print(f'    TARGETID pairs: {len(i_ref):,}  →  after cuts: {keep.sum():,}'
              f'  (Δv failures: {n_fail_z}, position failures: {n_fail_pos})')

    return i_ref[keep], i_ext[keep]


# ---------------------------------------------------------------------------
# CIGALE  (Zou et al. DR2)
# ---------------------------------------------------------------------------

def prepare_cigale(ntest=None, survey=None, specprod=DEFAULT_SPECPROD, verbose=False):
    """Prepare the CIGALE SED-fitting catalog (Zou et al. DR2).

    Source:
        /global/cfs/cdirs/desicollab/users/zouhu/vac/dr2/
        dr2_galaxy_sedfitting_v1.0.fits

    IMF: Chabrier (same as FastSpecFit — no IMF correction needed).
    Cosmology: H0=70 km/s/Mpc (h=0.7); FastSpecFit stores values at h=1.

    Unit conversions applied before writing:
        LOGMSTAR  [log10, h=1] = MASS_CG_*    + 2·log10(0.7)  (≈ −0.309 dex)
        SFR       [M⊙/yr, h=1] = SFR_CG_*     × 0.7²          (× 0.49)
        TAUV                   = AV_CG_*       / 1.086

    Uncertainty columns:
        log-mass errors (dex) are unchanged by the additive h correction.
        SFR errors scale the same as SFR (multiplicative).
        TAUV errors = AV errors / 1.086.

    Assumption: MASS_CG_15 / MASS_CG_5 are log10(M_star / M_sun). If they
    are instead linear stellar masses, the h-correction logic must be revised.
    """
    cigale_path = (
        '/global/cfs/cdirs/desicollab/users/zouhu/vac/dr2/'
        'dr2_galaxy_sedfitting_v1.0.fits'
    )
    if not os.path.exists(cigale_path):
        raise FileNotFoundError(f'CIGALE catalog not found: {cigale_path}')

    outdir = os.path.join(EXTDIR, 'cigale', specprod)
    os.makedirs(outdir, exist_ok=True)

    h_cigale = 0.7
    dlogm    = 2.0 * np.log10(h_cigale)  # ≈ −0.309 dex; additive to log10 mass
    h2       = h_cigale ** 2             # = 0.49; multiplicative to linear SFR

    # Read only the index columns once to enable cheap row selection per survey/program
    if verbose:
        print(f'Reading index columns from {cigale_path} ...')
    with fitsio.FITS(cigale_path) as fits:
        idx = fits[1].read(columns=['TARGETID', 'SURVEY', 'PROGRAM'])
    idx_survey  = _decode_str_col(idx['SURVEY'])
    idx_program = _decode_str_col(idx['PROGRAM'])
    if verbose:
        print(f'  {len(idx):,} total rows')

    for surv, program in SURVEY_PROGRAMS:
        if survey is not None and surv != survey:
            continue
        outfile = os.path.join(outdir, f'{surv}-{program}.fits')

        # Row indices in the CIGALE file for this survey+program combination
        rows = np.where((idx_survey == surv) & (idx_program == program))[0]
        if len(rows) == 0:
            print(f'  {surv}/{program}: no CIGALE rows found, skipping')
            continue
        if verbose:
            print(f'\n  {surv}/{program}: {len(rows):,} CIGALE rows')

        # --ntest: random subsample of the external catalog
        if ntest is not None:
            rng  = np.random.default_rng(42)
            rows = rng.choice(rows, min(ntest, len(rows)), replace=False)
            if verbose:
                print(f'    ntest subsample: {len(rows):,} CIGALE rows')

        # Read the selected rows from disk
        with fitsio.FITS(cigale_path) as fits:
            ext = Table(fits[1].read(rows=rows))

        # Read the reference FastSpecFit catalog for this survey+program
        try:
            ref = read_fastspec(surv, program, specprod=specprod,
                                columns=_ref_columns(surv), verbose=verbose)
        except (FileNotFoundError, ValueError) as exc:
            print(f'  {surv}/{program}: cannot read reference — {exc}, skipping')
            continue

        # --ntest: random subsample of the reference too
        if ntest is not None:
            rng = np.random.default_rng(42)
            ref = ref[rng.choice(len(ref), min(ntest, len(ref)), replace=False)]
            if verbose:
                print(f'    ntest subsample: {len(ref):,} reference rows')

        i_ref, i_ext = cross_match(ref, ext, verbose=verbose)
        if len(i_ref) == 0:
            print(f'  {surv}/{program}: no matches after consistency checks, skipping')
            continue
        print(f'  {surv}/{program}: {len(i_ref):,} matched → {outfile}')

        out   = ref[i_ref].copy()
        ext_m = ext[i_ext]

        # --- stellar mass [log10(M/M_sun), h=1] ---
        # MASS_CG_* is linear M_star / M_sun; log and apply h correction.
        # Propagate linear error to dex: sigma_log = sigma_lin / (M * ln10).
        # Guard against non-positive masses (bad fits → NaN).
        for suffix in ('CG_15', 'CG_5'):
            mass  = ext_m[f'MASS_{suffix}'].astype(float).copy()
            merr  = ext_m[f'MASSERR_{suffix}'].astype(float).copy()
            bad   = mass <= 0
            mass[bad] = np.nan
            merr[bad] = np.nan
            out[f'LOGMSTAR_{suffix}']  = np.log10(mass) + dlogm
            out[f'LOGMSTARE_{suffix}'] = merr / (mass * np.log(10))

        # --- SFR [M_sun/yr, linear, h=1] ---
        out['SFR_CG_15']  = ext_m['SFR_CG_15']    * h2
        out['SFRE_CG_15'] = ext_m['SFRERR_CG_15'] * h2
        out['SFR_CG_5']   = ext_m['SFR_CG_5']     * h2
        out['SFRE_CG_5']  = ext_m['SFRERR_CG_5']  * h2

        # --- dust attenuation: AV → TAUV = AV / 1.086 ---
        out['TAUV_CG_15']    = ext_m['AV_CG_15']    / 1.086
        out['TAUVERR_CG_15'] = ext_m['AVERR_CG_15'] / 1.086
        out['TAUV_CG_5']     = ext_m['AV_CG_5']     / 1.086
        out['TAUVERR_CG_5']  = ext_m['AVERR_CG_5']  / 1.086

        # --- aperture correction factor (dimensionless, no conversion) ---
        out['FLUX_SCALE'] = ext_m['FLUX_SCALE']

        out.write(outfile, overwrite=True)

    print('Done.')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument('--cigale', action='store_true',
                        help='Prepare the CIGALE catalog (Zou et al. DR2).')
    parser.add_argument('--specprod', default=DEFAULT_SPECPROD,
                        help='Spectroscopic production name.')
    parser.add_argument('--ntest', type=int, default=None, metavar='N',
                        help='Random subsample of N rows from each catalog '
                             '(for testing; note: independent subsamples may yield '
                             'few or no cross-matches).')
    parser.add_argument('--survey', default=None, choices=['sv3', 'main'],
                        help='Restrict preparation to this survey. '
                             'Defaults to sv3 when --ntest is set, all surveys otherwise.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress while reading and matching.')
    args = parser.parse_args()

    # When testing, default to sv3 (smaller catalogs) unless the user says otherwise.
    survey = args.survey
    if args.ntest is not None and survey is None:
        survey = 'sv3'

    if args.cigale:
        prepare_cigale(ntest=args.ntest, survey=survey,
                       specprod=args.specprod, verbose=args.verbose)


if __name__ == '__main__':
    main()
