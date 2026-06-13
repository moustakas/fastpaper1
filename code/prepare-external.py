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
    python code/prepare-external.py --zouhu [--specprod loa|iron] [--verbose]

"""
import os, sys, argparse
import numpy as np
from astropy.table import Table
import fitsio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import read_fastspec, DEFAULT_SPECPROD

C_LIGHT        = 2.998e5  # km/s
MAX_DV_KMS     = 1000.0   # redshift-consistency threshold [km/s]
MAX_SEP_ARCSEC = 1.5      # positional-consistency threshold [arcsec]

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
    'COADD_FIBERSTATUS', 'DELTACHI2',
    'LOGMSTAR', 'LOGMSTAR_IVAR', 'SFR', 'SFR_IVAR', 'TAUV', 'TAUV_IVAR',
    'VDISP', 'VDISP_IVAR', 'AGE', 'AGE_IVAR', 'SNR_B', 'SNR_R', 'SNR_Z',
    'APERCORR', 'DN4000_MODEL', 'DN4000_MODEL_IVAR', 'KCORR01_SDSS_U',
    'KCORR01_SDSS_G', 'KCORR01_SDSS_R', 'KCORR01_SDSS_I', 'KCORR01_SDSS_Z',
    'ABSMAG01_SDSS_U', 'ABSMAG01_SDSS_G', 'ABSMAG01_SDSS_R', 'ABSMAG01_SDSS_I',
    'ABSMAG01_SDSS_Z', 'ABSMAG01_IVAR_SDSS_U', 'ABSMAG01_IVAR_SDSS_G',
    'ABSMAG01_IVAR_SDSS_R', 'ABSMAG01_IVAR_SDSS_I', 'ABSMAG01_IVAR_SDSS_Z',
    'OII_3726_FLUX', 'OII_3729_FLUX', 'OII_3726_FLUX_IVAR', 'OII_3729_FLUX_IVAR',
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


def cross_match_radec(ref, ext, ext_z_col='Z', ext_ra_col='RA', ext_dec_col='DEC',
                      photo_z_tol=None, verbose=False):
    """Match ref and ext on sky position, then check redshift consistency.

    Used for external (non-DESI) catalogs that have no TARGETID. For each ref
    object the nearest ext neighbor is found; pairs are kept only if the
    separation is < MAX_SEP_ARCSEC and the redshift criterion is satisfied.

    Parameters
    ----------
    ref : astropy.table.Table
        Reference FastSpecFit catalog; must have RA, DEC, Z.
    ext : astropy.table.Table
        External catalog; must have positional and redshift columns.
    ext_z_col, ext_ra_col, ext_dec_col : str
        Column names in ext.
    photo_z_tol : float or None
        If set, use the photometric-redshift criterion |Δz|/(1+z_ref) < photo_z_tol
        instead of the default spectroscopic criterion |Δv| < MAX_DV_KMS.
    verbose : bool

    Returns
    -------
    i_ref, i_ext : ndarray of int
        Indices into ref and ext of matched, consistency-checked pairs.
    """
    from astropy.coordinates import SkyCoord
    import astropy.units as u_

    c_ref = SkyCoord(ref['RA'] * u_.deg, ref['DEC'] * u_.deg)
    c_ext = SkyCoord(ext[ext_ra_col] * u_.deg, ext[ext_dec_col] * u_.deg)

    idx_ext, sep, _ = c_ref.match_to_catalog_sky(c_ext)
    pos_ok = sep.arcsec < MAX_SEP_ARCSEC

    if photo_z_tol is not None:
        dz   = np.abs(ref['Z'] - ext[ext_z_col][idx_ext]) / (1. + ref['Z'])
        z_ok = dz < photo_z_tol
    else:
        dv   = np.abs(ref['Z'] - ext[ext_z_col][idx_ext]) * C_LIGHT
        z_ok = dv < MAX_DV_KMS

    keep  = pos_ok & z_ok
    i_ref = np.where(keep)[0]
    i_ext = idx_ext[keep]

    n_fail_pos = int((~pos_ok).sum())
    n_fail_z   = int((pos_ok & ~z_ok).sum())
    if verbose or n_fail_z or n_fail_pos:
        print(f'    Position matches: {pos_ok.sum():,}  →  after z cut: {keep.sum():,}'
              f'  (z failures: {n_fail_z}, position failures: {n_fail_pos})')

    return i_ref, i_ext


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

def prepare_zouhu(survey=None, specprod=DEFAULT_SPECPROD, verbose=False):
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
    newcols  = _columns[specprod]['newcols']

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

        rows = np.where((idx_survey == surv) & (idx_program == program))[0]
        if len(rows) == 0:
            print(f'  {surv}-{program}: no rows found; skipping')
            continue
        if verbose:
            print(f'\n  {surv}/{program}: {len(rows):,} {shortcat} rows')

        with fitsio.FITS(zouhu_path) as fits:
            ext = Table(fits[1].read(columns=readcols, rows=rows))
        ext.rename_columns(readcols, newcols)

        try:
            ref = read_fastspec(surv, program, specprod=DEFAULT_SPECPROD,
                                columns=_ref_columns(surv), verbose=verbose)
        except (FileNotFoundError, ValueError) as exc:
            print(f'  {surv}/{program}: cannot read reference — {exc}, skipping')
            continue

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

def prepare_cigaleagn(survey=None, verbose=False):
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
# Salim et al. (GSWLC-X2 / SDSS-based, no TARGETID)
# ---------------------------------------------------------------------------

def prepare_gswlcx2(survey=None, verbose=False):
    """Prepare the Salim et al. GALEX-SDSS-WISE Legacy Catalog (GSWLC-X2).

    Source:
        /dvs_ro/cfs/cdirs/desicollab/users/ioannis/fastspecfit/external/
        GSWLC-X2.dat

    This is an SDSS-based catalog with no TARGETID and no SURVEY/PROGRAM
    columns. Matching to the FastSpecFit reference is done purely by sky
    position (< 1.5 arcsec) and redshift (|Δv| < 1000 km/s) using
    cross_match_radec().  The full catalog is read once before the
    survey/program loop.

    IMF: Chabrier (same as FastSpecFit — no IMF correction needed).
    Cosmology: WMAP7, H0=70.4 km/s/Mpc (h=0.704); FastSpecFit stores values at h=1.

    Unit conversions applied:
        LOGMSTAR_h1  = LOGMSTAR + 2·log10(0.704)         (≈ −0.305 dex; additive)
        LOGMSTAR_ERR unchanged (dex errors are additive-h-invariant)
        SFR_h1       = 10^(LOGSFR + 2·log10(0.704))      [M_sun/yr, linear]
        SFR_ERR_h1   = SFR_h1 × LOGSFR_ERR × ln(10)     (dex → linear propagation)
        TAUV         = AV / 1.086
        TAUV_ERR     = AV_ERR / 1.086

    References:
        Salim et al. (2016) — https://iopscience.iop.org/article/10.3847/0067-0049/227/1/2
        Salim et al. (2018) — https://iopscience.iop.org/article/10.3847/1538-4357/aabf3c
        Catalog homepage   — https://salims.pages.iu.edu/gswlc/

    """
    from astropy.table import Table as ATable

    shortcat = 'gswlcx2'
    gswlcx2_path = (
        '/dvs_ro/cfs/cdirs/desicollab/users/ioannis/fastspecfit/external/'
        'GSWLC-X2.dat'
    )
    if not os.path.exists(gswlcx2_path):
        raise FileNotFoundError(f'GSWLC-X2 catalog not found: {gswlcx2_path}')

    h_gswlc = 0.704
    dlogm   = 2.0 * np.log10(h_gswlc)  # ≈ −0.305 dex

    if verbose:
        print(f'Reading {gswlcx2_path} ...')
    ext_full = ATable.read(
        gswlcx2_path, format='ascii',
        names=['OBJID', 'GLXID', 'PLATE', 'MJD', 'FIBERID', 'RA', 'DEC', 'Z',
               'RCHI2', 'LOGMSTAR', 'LOGMSTAR_ERR', 'LOGSFR', 'LOGSFR_ERR',
               'AFUV', 'AFUV_ERR', 'AB', 'AB_ERR', 'AV', 'AV_ERR',
               'FLAG_SED', 'UVSURVEY', 'FLAG_UV', 'FLAG_MIDIR', 'FLAG_MGS'],
    )
    ext_full = ext_full['OBJID', 'RA', 'DEC', 'Z',
                        'LOGMSTAR', 'LOGMSTAR_ERR', 'LOGSFR', 'LOGSFR_ERR',
                        'AV', 'AV_ERR']
    if verbose:
        print(f'  ... read {len(ext_full):,} rows')

    for surv, program in SURVEY_PROGRAMS:
        if survey is not None and surv != survey:
            continue
        outfile = os.path.join(EXTDIR, f'{shortcat}-{surv}-{program}.fits')

        try:
            ref = read_fastspec(surv, program, specprod=DEFAULT_SPECPROD,
                                columns=_ref_columns(surv), verbose=verbose)
        except (FileNotFoundError, ValueError) as exc:
            print(f'  {surv}/{program}: cannot read reference — {exc}, skipping')
            continue

        i_ref, i_ext = cross_match_radec(ref, ext_full, verbose=verbose)
        if len(i_ref) == 0:
            print(f'  {surv}/{program}: no matches after consistency checks, skipping')
            continue
        print(f'  {surv}/{program}: {len(i_ref):,} matched → {outfile}')

        out   = ref[i_ref].copy()
        ext_m = ext_full[i_ext]

        # stellar mass [log10(M/M_sun), h=1]: already log, additive h correction
        out[f'LOGMSTAR_{shortcat.upper()}']     = ext_m['LOGMSTAR']     + dlogm
        out[f'LOGMSTAR_ERR_{shortcat.upper()}'] = ext_m['LOGMSTAR_ERR']  # dex unchanged

        # SFR: log10(M_sun/yr) at h=0.704 → linear M_sun/yr at h=1
        sfr_h1 = np.power(10., ext_m['LOGSFR'] + dlogm)
        out[f'SFR_{shortcat.upper()}']     = sfr_h1
        out[f'SFR_ERR_{shortcat.upper()}'] = sfr_h1 * ext_m['LOGSFR_ERR'] * np.log(10.)

        # dust: AV [mag] → TAUV = AV / 1.086
        out[f'TAUV_{shortcat.upper()}']     = ext_m['AV']     / 1.086
        out[f'TAUV_ERR_{shortcat.upper()}'] = ext_m['AV_ERR'] / 1.086

        out.write(outfile, overwrite=True)

    print('Done.')


# ---------------------------------------------------------------------------
# Weaver et al. (COSMOS2020 / photometric, no TARGETID or SURVEY/PROGRAM)
# ---------------------------------------------------------------------------

def prepare_cosmos2020(survey=None, verbose=False):
    """Prepare the Weaver et al. COSMOS2020 photometric catalog.

    Source:
        /dvs_ro/homes/i/ioannis/ioannis/fastspecfit/laelbg-templates/
        COSMOS2020_FARMER_R1_v2.1_p3.fits

    No TARGETID or SURVEY/PROGRAM columns. Matched to FastSpecFit reference by
    sky position (< 1.5 arcsec). Because redshifts are photometric, the
    consistency check uses |Δz|/(1+z_ref) < 0.2. The full catalog is read
    once, immediately trimmed of non-finite lp_zBEST and lp_mass_best rows,
    then matched per survey/program.

    IMF: Chabrier (same as FastSpecFit — no IMF correction needed).
    Cosmology: h=0.7, OmegaM=0.3, OmegaL=0.7; FastSpecFit stores values at h=1.

    Unit conversions applied:
        LOGMSTAR_h1 = lp_mass_best + 2·log10(0.7)      (≈ −0.309 dex; additive)
        SFR_h1      = 10^(lp_SFR_best + 2·log10(0.7))  [M_sun/yr, linear]

    Note: uncertainties (lp_mass_inf/sup, lp_SFR_inf/sup) are not yet included
    and should be added in a future update.

    References:
        Weaver et al. (2022) — https://iopscience.iop.org/article/10.3847/1538-4365/ac3078
        IRSA catalog page   — https://irsa.ipac.caltech.edu/data/COSMOS/tables/cosmos2020/

    """
    shortcat    = 'cosmos2020'
    cosmos_path = (
        '/dvs_ro/cfs/cdirs/desicollab/users/ioannis/fastspecfit/external/'
        'COSMOS2020_FARMER_R1_v2.1_p3.fits'
    )
    if not os.path.exists(cosmos_path):
        raise FileNotFoundError(f'COSMOS2020 catalog not found: {cosmos_path}')

    h_cosmos = 0.7
    dlogm    = 2.0 * np.log10(h_cosmos)  # ≈ −0.309 dex

    readcols = ['ID', 'ALPHA_J2000', 'DELTA_J2000', 'lp_zBEST',
                'lp_mass_best', 'lp_SFR_best', 'FLAG_COMBINED']

    if verbose:
        print(f'Reading {cosmos_path} ...')
    with fitsio.FITS(cosmos_path) as fits:
        ext_full = Table(fits[1].read(columns=readcols))
    ext_full.rename_columns(
        ['ALPHA_J2000', 'DELTA_J2000', 'lp_zBEST', 'lp_mass_best', 'lp_SFR_best'],
        ['RA',          'DEC',          'Z',         'LOGMSTAR',     'LOGSFR'],
    )
    if verbose:
        print(f'  ... read {len(ext_full):,} rows')

    good     = np.isfinite(ext_full['Z']) & np.isfinite(ext_full['LOGMSTAR'])
    ext_full = ext_full[good]
    if verbose:
        print(f'  ... {len(ext_full):,} rows after trimming non-finite Z and LOGMSTAR')

    for surv, program in SURVEY_PROGRAMS:
        if survey is not None and surv != survey:
            continue
        outfile = os.path.join(EXTDIR, f'{shortcat}-{surv}-{program}.fits')

        try:
            ref = read_fastspec(surv, program, specprod=DEFAULT_SPECPROD,
                                columns=_ref_columns(surv), verbose=verbose)
        except (FileNotFoundError, ValueError) as exc:
            print(f'  {surv}/{program}: cannot read reference — {exc}, skipping')
            continue

        i_ref, i_ext = cross_match_radec(ref, ext_full, photo_z_tol=0.2, verbose=verbose)
        if len(i_ref) == 0:
            print(f'  {surv}/{program}: no matches after consistency checks, skipping')
            continue
        print(f'  {surv}/{program}: {len(i_ref):,} matched → {outfile}')

        out   = ref[i_ref].copy()
        ext_m = ext_full[i_ext]

        # COSMOS ID (useful for cross-referencing with other COSMOS catalogs)
        out[f'ID_{shortcat.upper()}'] = ext_m['ID']

        # stellar mass [log10(M/M_sun), h=1]: already log, additive h correction
        out[f'LOGMSTAR_{shortcat.upper()}'] = ext_m['LOGMSTAR'] + dlogm
        # (uncertainties lp_mass_inf/sup not yet included)

        # SFR: log10(M_sun/yr) at h=0.7 → linear M_sun/yr at h=1
        out[f'SFR_{shortcat.upper()}'] = np.power(10., ext_m['LOGSFR'] + dlogm)
        # (uncertainties lp_SFR_inf/sup not yet included)

        # quality flag
        out[f'FLAG_COMBINED_{shortcat.upper()}'] = ext_m['FLAG_COMBINED']

        out.write(outfile, overwrite=True)

    print('Done.')


# ---------------------------------------------------------------------------
# Ross et al. fundamental-plane catalog (Iron only, not a formal DESI VAC)
# ---------------------------------------------------------------------------

def prepare_fpcatalog(survey=None, verbose=False):
    """Prepare the Ross et al. fundamental-plane / velocity-dispersion catalog (DR1/Iron only).

    Source (not a formal DESI VAC):
        /dvs_ro/cfs/cdirs/desi/science/td/pv/VAC/DR1/peculiar-velocity/v1.0/
        fundamental-plane/FP_catalogue_v5.fits

    No IMF or cosmology corrections are needed; all columns are carried through
    as-is with a _FPCATALOG suffix.

    References:
        Ross et al. (2026, in press) — https://arxiv.org/abs/2512.03226

    """
    shortcat = 'fpcatalog'
    specprod = 'iron'  # no Loa version exists

    fpcatalog_path = (
        '/dvs_ro/cfs/cdirs/desi/science/td/pv/VAC/DR1/peculiar-velocity/v1.0/'
        'fundamental-plane/FP_catalogue_v5.fits'
    )
    if not os.path.exists(fpcatalog_path):
        raise FileNotFoundError(f'FP catalog not found: {fpcatalog_path}')

    readcols = ['TARGETID', 'RA', 'DEC', 'Z',
                'PPXF_VDISP', 'PPXF_VDISP_ERR',
                'PORTSMOUTH_SIGMA_STARS', 'PORTSMOUTH_SIGMA_STARS_ERR',
                'FPCALIBRATOR', 'PRIMARYVDISP']

    if verbose:
        print(f'Reading index columns from {fpcatalog_path} ...')
    with fitsio.FITS(fpcatalog_path) as fits:
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

        with fitsio.FITS(fpcatalog_path) as fits:
            ext = Table(fits[1].read(columns=readcols, rows=rows))

        try:
            ref = read_fastspec(surv, program, specprod=DEFAULT_SPECPROD,
                                columns=_ref_columns(surv), verbose=verbose)
        except (FileNotFoundError, ValueError) as exc:
            print(f'  {surv}/{program}: cannot read reference — {exc}, skipping')
            continue

        i_ref, i_ext = cross_match(ref, ext, verbose=verbose)
        if len(i_ref) == 0:
            print(f'  {surv}/{program}: no matches after consistency checks, skipping')
            continue
        print(f'  {surv}/{program}: {len(i_ref):,} matched → {outfile}')

        out   = ref[i_ref].copy()
        ext_m = ext[i_ext]

        for col in ['PPXF_VDISP', 'PPXF_VDISP_ERR',
                    'PORTSMOUTH_SIGMA_STARS', 'PORTSMOUTH_SIGMA_STARS_ERR',
                    'FPCALIBRATOR', 'PRIMARYVDISP']:
            out[f'{col}_{shortcat.upper()}'] = ext_m[col]

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
    parser.add_argument('--gswlcx2', action='store_true',
                        help='Prepare the Salim et al. GSWLC-X2 catalog (SDSS; matched by sky position).')
    parser.add_argument('--cosmos2020', action='store_true',
                        help='Prepare the Weaver et al. COSMOS2020 photometric catalog.')
    parser.add_argument('--fpcatalog', action='store_true',
                        help='Prepare the Ross et al. fundamental-plane catalog (DR1/iron only).')
    parser.add_argument('--specprod', default=DEFAULT_SPECPROD,
                        help='Spectroscopic production name.')
    parser.add_argument('--survey', default=None, choices=['sv3', 'main'],
                        help='Restrict preparation to this survey (default: all surveys).')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress while reading and matching.')
    args = parser.parse_args()

    if args.zouhu:
        prepare_zouhu(survey=args.survey, specprod=args.specprod, verbose=args.verbose)

    if args.cigaleagn:
        prepare_cigaleagn(survey=args.survey, verbose=args.verbose)

    if args.gswlcx2:
        prepare_gswlcx2(survey=args.survey, verbose=args.verbose)

    if args.cosmos2020:
        prepare_cosmos2020(survey=args.survey, verbose=args.verbose)

    if args.fpcatalog:
        prepare_fpcatalog(survey=args.survey, verbose=args.verbose)


if __name__ == '__main__':
    main()
