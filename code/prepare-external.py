#!/usr/bin/env python
"""Prepare external catalogs for cross-comparison with the FastSpecFit DR2 (Loa) VAC.

For each catalog a prepare_*() function reads the external file once, then loops
over (survey, program) combinations, cross-matches to the FastSpecFit reference
catalog (via util.read_fastspec), applies positional and redshift consistency
checks, standardizes units, and writes a compact prepared file to ./external/.

Output files are named:
    external/{shortcat}-{specprod}-{survey}-{program}.fits

where {specprod} identifies the external catalog's data release (e.g. 'loa',
'iron') and {shortcat} is the catalog's short name (e.g. 'zouhu').  The
reference catalog is always the Loa FastSpecFit VAC.

Each output file contains the matched reference columns (TARGETID, RA, DEC, Z,
LOGMSTAR, SFR, TAUV, …) side-by-side with standardized external columns, all
converted to h=1 and Chabrier IMF, ready for direct comparison.

Usage (from repo root or code/):
    python code/prepare-external.py --zouhu [--specprod loa|iron] [--ntest N] [--verbose]

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
    'LOGMSTAR', 'LOGMSTAR_IVAR', 'SFR', 'SFR_IVAR', 'TAUV', 'TAUV_IVAR',
    'VDISP', 'VDISP_IVAR', #'DN4000', 'DN4000_IVAR',
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
# Zou et al. (Iron, Loa - CIGALE)
# ---------------------------------------------------------------------------

def prepare_zouhu(ntest=None, survey=None, specprod=DEFAULT_SPECPROD, verbose=False):
    """Prepare the Zou et al. (CIGALE) SED-fitting catalogs.

    Loa (DR2) Source:
        /dvs_ro/cfs/cdirs/desicollab/users/zouhu/vac/dr2/
        dr2_galaxy_sedfitting_v1.0.fits
    Iron (DR1) Source:
        /dvs_ro/cfs/cdirs/desi/public/dr1/vac/dr1/stellar-mass-emline/

    IMF: Chabrier (same as FastSpecFit — no IMF correction needed).
    Cosmology: H0=70 km/s/Mpc (h=0.7); FastSpecFit stores values at h=1.

    Unit conversions applied before writing:
        LOGMSTAR  [log10, h=1] = log10(MASS_CG_*) + 2·log10(0.7)  (≈ −0.309 dex)
        LOGMSTAR_ERR  [dex]    = MASSERR_CG_* / (MASS_CG_* × ln 10)
        SFR       [M⊙/yr, h=1] = SFR_CG_*  × 0.7²                 (× 0.49)
        SFR_ERR                = SFRERR_CG_* × 0.7²
        TAUV                   = AV_CG_*  / 1.086
        TAUV_ERR               = AVERR_CG_* / 1.086

    MASS_CG_* and MASSERR_CG_* are linear stellar masses in M_sun.
    Only the _CG_15 variant (5-band tractor + 10-band spectrophotometry) is
    included; the 5-band-only _CG_5 variant is omitted.

    """
    _zouhu_path = {
        'loa': '/dvs_ro/cfs/cdirs/desicollab/users/zouhu/vac/dr2/dr2_galaxy_sedfitting_v1.0.fits',
        'iron': '/dvs_ro/cfs/cdirs/desi/public/dr1/vac/dr1/stellar-mass-emline/v1.0/dr1_galaxy_stellarmass_lineinfo_v1.0.fits',
    }
    zouhu_path = _zouhu_path[specprod]
    if not os.path.exists(zouhu_path):
        raise FileNotFoundError(f'Zou et al. catalog not found: {zouhu_path}')

    shortcat = 'zouhu'
    h_zouhu = 0.7
    dlogm   = 2. * np.log10(h_zouhu)  # ≈ −0.309 dex; additive to log10 mass
    h2      = h_zouhu**2.             # = 0.49; multiplicative to linear SFR

    _columns = {
        'loa': {
            'readcols': ['TARGETID', 'TARGET_RA', 'TARGET_DEC', 'Z', 'FLUX_SCALE', 'AV_CG_15',
                         'AVERR_CG_15', 'SFR_CG_15', 'SFRERR_CG_15', 'MASS_CG_15',
                         'MASSERR_CG_15', ],
            'newcols': ['TARGETID', 'RA', 'DEC', 'Z', 'APERCORR', 'AV', 'AV_ERR', 'SFR',
                        'SFR_ERR', 'MSTAR', 'MSTAR_ERR', ],
        },
        'iron': {
            'readcols': ['TARGETID', 'TARGET_RA', 'TARGET_DEC', 'Z', 'FLUX_SCALE', 'AV_CG',
                         'AVERR_CG', 'SFR_CG', 'SFRERR_CG', 'MASS_CG', 'MASSERR_CG', ],
            'newcols': ['TARGETID', 'RA', 'DEC', 'Z', 'APERCORR', 'AV', 'AV_ERR', 'SFR',
                        'SFR_ERR', 'MSTAR', 'MSTAR_ERR', ],
        },
    }
    readcols = _columns[specprod]['readcols']
    newcols = _columns[specprod]['newcols']

    if verbose:
        print(f'Reading index columns from {zouhu_path} ...')
    with fitsio.FITS(zouhu_path) as fits:
        idx = fits[1].read(columns=['SURVEY', 'PROGRAM'])
    idx_survey  = _decode_str_col(idx['SURVEY'])
    idx_program = _decode_str_col(idx['PROGRAM'])
    if verbose:
        print(f'  ... read {len(idx):,} rows')

    for surv, program in SURVEY_PROGRAMS:
        if survey is not None and surv != survey:
            continue
        outfile = os.path.join(EXTDIR, f'{shortcat}-{specprod}-{surv}-{program}.fits')

        # Row indices for this survey+program combination
        rows = np.where((idx_survey == surv) & (idx_program == program))[0]
        if len(rows) == 0:
            print(f'  {surv}-{program}: no rows found; skipping')
            continue
        if verbose:
            print(f'\n  {surv}/{program}: {len(rows):,} {shortcat} rows')

        # --ntest: random subsample of the external catalog
        if ntest is not None:
            rng  = np.random.default_rng(42)
            rows = rng.choice(rows, min(ntest, len(rows)), replace=False)
            if verbose:
                print(f'    ntest subsample: {len(rows):,} {shortcat} rows')

        # Read the selected rows from disk
        with fitsio.FITS(zouhu_path) as fits:
            ext = Table(fits[1].read(columns=readcols, rows=rows))
        ext.rename_columns(readcols, newcols)

        # Read the reference FastSpecFit catalog for this
        # survey+program; always read from DR2/Loa!
        try:
            ref = read_fastspec(surv, program, specprod=DEFAULT_SPECPROD,
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

        # stellar mass: linear M_sun → log10(M/M_sun) at h=1
        mstar    = ext_m['MSTAR'].astype(float).copy()
        mstarerr = ext_m['MSTAR_ERR'].astype(float).copy()
        bad = (mstar <= 0.) | (mstarerr <= 0.) | np.isnan(mstar) | np.isnan(mstarerr)
        mstar[bad] = np.nan
        mstarerr[bad] = np.nan
        out[f'LOGMSTAR_{shortcat.upper()}']     = np.log10(mstar) + dlogm
        out[f'LOGMSTAR_ERR_{shortcat.upper()}'] = mstarerr / (mstar * np.log(10))

        # SFR: linear M_sun/yr → h=1
        out[f'SFR_{shortcat.upper()}']     = ext_m['SFR']     * h2
        out[f'SFR_ERR_{shortcat.upper()}'] = ext_m['SFR_ERR'] * h2

        # dust: AV [mag] → TAUV = AV / 1.086
        out[f'TAUV_{shortcat.upper()}']     = ext_m['AV']     / 1.086
        out[f'TAUV_ERR_{shortcat.upper()}'] = ext_m['AV_ERR'] / 1.086

        # aperture correction factor
        out[f'APERCORR_{shortcat.upper()}'] = ext_m['APERCORR']

        out.write(outfile, overwrite=True)

    print('Done.')


# ---------------------------------------------------------------------------
# Siudek et al. (CIGALE-AGN / Iron only)
# ---------------------------------------------------------------------------

def prepare_cigaleagn(ntest=None, survey=None, verbose=False):
    """Prepare the Siudek et al. CIGALE-AGN catalog (DR1/Iron only).

    Source:
        /dvs_ro/cfs/cdirs/desi/public/dr1/vac/dr1/cigale/iron/v1.2/
        IronPhysProp_v1.2.fits

    IMF: Chabrier (same as FastSpecFit — no IMF correction needed).
    Cosmology: WMAP7, H0=70.4 km/s/Mpc (h=0.704); FastSpecFit stores values at h=1.

    LOGM is already in log10 space; the h correction is purely additive:
        LOGMSTAR_h1  = LOGM + 2·log10(0.704)              (≈ −0.305 dex)
        LOGMSTAR_ERR unchanged (dex errors are additive-h-invariant)

    LOGSFR is converted to linear to match the FastSpecFit VAC convention:
        SFR_h1      = 10^(LOGSFR + 2·log10(0.704))        [M_sun/yr, linear]
        SFR_ERR_h1  = SFR_h1 × LOGSFR_ERR × ln(10)       (dex → linear propagation)

    Quality flags are stored in the output for use in figure code:
        Good masses:  0.2 < FLAG_MASSPDF < 5.0
        Good SFRs:    0.2 < FLAG_SFRPDF  < 5.0

    References:
        Siudek et al. (2025) — https://www.aanda.org/articles/aa/full_html/2025/08/aa55463-25/aa55463-25.html
        DR1 VAC documentation — https://data.desi.lbl.gov/doc/releases/dr1/vac/cigale/

    """
    shortcat = 'cigaleagn'
    specprod = 'iron'  # no Loa version exists

    cigaleagn_path = (
        '/dvs_ro/cfs/cdirs/desi/public/dr1/vac/dr1/cigale/iron/v1.2/'
        'IronPhysProp_v1.2.fits'
    )
    if not os.path.exists(cigaleagn_path):
        raise FileNotFoundError(f'CIGALE-AGN catalog not found: {cigaleagn_path}')

    h_cigaleagn = 0.704
    dlogm       = 2.0 * np.log10(h_cigaleagn)  # ≈ −0.305 dex

    if verbose:
        print(f'Reading index columns from {cigaleagn_path} ...')
    with fitsio.FITS(cigaleagn_path) as fits:
        idx = fits[1].read(columns=['SURVEY', 'PROGRAM'])
    idx_survey  = _decode_str_col(idx['SURVEY'])
    idx_program = _decode_str_col(idx['PROGRAM'])
    if verbose:
        print(f'  ... read {len(idx):,} rows')

    for surv, program in SURVEY_PROGRAMS:
        if survey is not None and surv != survey:
            continue
        outfile = os.path.join(EXTDIR, f'{shortcat}-{specprod}-{surv}-{program}.fits')

        rows = np.where((idx_survey == surv) & (idx_program == program))[0]
        if len(rows) == 0:
            print(f'  {surv}-{program}: no rows found; skipping')
            continue
        if verbose:
            print(f'\n  {surv}/{program}: {len(rows):,} {shortcat} rows')

        if ntest is not None:
            rng  = np.random.default_rng(42)
            rows = rng.choice(rows, min(ntest, len(rows)), replace=False)
            if verbose:
                print(f'    ntest subsample: {len(rows):,} {shortcat} rows')

        readcols = ['TARGETID', 'RA', 'DEC', 'Z',
                    'LOGM', 'LOGM_ERR', 'LOGSFR', 'LOGSFR_ERR',
                    'FLAG_MASSPDF', 'FLAG_SFRPDF', 'AGNFRAC']
        with fitsio.FITS(cigaleagn_path) as fits:
            ext = Table(fits[1].read(columns=readcols, rows=rows))
        ext.rename_columns(['LOGM', 'LOGM_ERR', 'FLAG_MASSPDF', 'FLAG_SFRPDF'],
                           ['LOGMSTAR', 'LOGMSTAR_ERR', 'FLAG_LOGMSTAR', 'FLAG_LOGSFR'])

        try:
            ref = read_fastspec(surv, program, specprod=DEFAULT_SPECPROD,
                                columns=_ref_columns(surv), verbose=verbose)
        except (FileNotFoundError, ValueError) as exc:
            print(f'  {surv}/{program}: cannot read reference — {exc}, skipping')
            continue

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

        # stellar mass [log10(M/M_sun), h=1]: already log, additive h correction only
        out[f'LOGMSTAR_{shortcat.upper()}']     = ext_m['LOGMSTAR']     + dlogm
        out[f'LOGMSTAR_ERR_{shortcat.upper()}'] = ext_m['LOGMSTAR_ERR']  # dex unchanged

        # SFR: log10(M_sun/yr) at h=0.704 → linear M_sun/yr at h=1
        sfr_h1 = np.power(10., ext_m['LOGSFR'] + dlogm)
        out[f'SFR_{shortcat.upper()}']     = sfr_h1
        out[f'SFR_ERR_{shortcat.upper()}'] = sfr_h1 * ext_m['LOGSFR_ERR'] * np.log(10.)

        # quality flags (cuts applied in figure code, not here)
        out[f'FLAG_LOGMSTAR_{shortcat.upper()}'] = ext_m['FLAG_LOGMSTAR']
        out[f'FLAG_LOGSFR_{shortcat.upper()}']   = ext_m['FLAG_LOGSFR']

        # AGN fraction
        out[f'AGNFRAC_{shortcat.upper()}'] = ext_m['AGNFRAC']

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
    parser.add_argument('--zouhu', action='store_true',
                        help='Prepare the Zou et al. CIGALE catalog (DR2/loa or DR1/iron).')
    parser.add_argument('--cigaleagn', action='store_true',
                        help='Prepare the Siudek et al. CIGALE-AGN catalog (DR1/iron only).')
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

    if args.zouhu:
        prepare_zouhu(ntest=args.ntest, survey=survey,
                      specprod=args.specprod, verbose=args.verbose)

    if args.cigaleagn:
        prepare_cigaleagn(ntest=args.ntest, survey=survey, verbose=args.verbose)


if __name__ == '__main__':
    main()
