"""Utility functions for reading DESI/FastSpecFit VAC catalogs.

Each read_* function returns an astropy Table with a standardized data model
so that figure-making scripts do not need to know catalog-specific details.

Cosmology note: LOGMSTAR and SFR in the FastSpecFit VACs are stored with h=1
(i.e., M_star in units of h^{-2} M_sun). Conversion to H0=70 adds
+2*log10(1/0.7) ~ +0.31 dex; to Planck 2018 (h=0.674) adds ~+0.34 dex.
Comparisons across catalogs must use a consistent h; each read_* function
documents which convention it returns.
"""

import os
import numpy as np
from glob import glob
from astropy.table import Table, hstack, vstack
import fitsio

# DR2 / Loa v1.0 VAC roots on NERSC
_FASTSPEC_VACDIR = '/global/cfs/cdirs/desi/vac/dr2/fastspecfit/loa/v1.0/catalogs'
_FASTPHOT_VACDIR = '/global/cfs/cdirs/desi/vac/dr2/fastphot/loa/v1.0/catalogs'

# survey+program combinations split into 12 nside=1 healpix files
_SPLIT_COMBOS = {('main', 'bright'), ('main', 'dark')}


def _catfiles(vacdir, vactype, survey, program):
    """Return sorted list of catalog paths for a survey/program combination."""
    if (survey, program) in _SPLIT_COMBOS:
        pattern = os.path.join(vacdir, f'{vactype}-{survey}-{program}-nside1-hp??.fits')
        files = sorted(glob(pattern))
        if not files:
            raise FileNotFoundError(
                f'No {vactype} files found matching {pattern}')
        return files
    path = os.path.join(vacdir, f'{vactype}-{survey}-{program}.fits')
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return [path]


def _read_extensions(filepath, extensions, columns=None):
    """Read and hstack FITS extensions from one file.

    The first extension (METADATA) is always read in full. For subsequent
    extensions, `columns` limits which columns are loaded; columns already
    present from METADATA are never duplicated.

    Parameters
    ----------
    filepath : str
    extensions : list of str
        Extension names to read, e.g. ['METADATA', 'SPECPHOT', 'FASTSPEC'].
    columns : list of str or None
        Columns to read from extensions[1:]. None means read all.

    Returns
    -------
    astropy.table.Table
    """
    parts = []
    seen = set()

    with fitsio.FITS(filepath) as fits:
        for i, ext in enumerate(extensions):
            ext_cols = set(c.upper() for c in fits[ext].get_colnames())

            if i == 0:
                # METADATA: always full
                read_cols = None
            elif columns is not None:
                want = (set(c.upper() for c in columns) & ext_cols) - seen
                if not want:
                    seen.update(ext_cols)
                    continue
                read_cols = sorted(want)
            else:
                read_cols = sorted(ext_cols - seen)
                if not read_cols:
                    continue

            data = fits[ext].read(columns=read_cols)
            t = Table(data)
            parts.append(t)
            seen.update(t.colnames)

    return hstack(parts) if len(parts) > 1 else parts[0]


def _read_vac(vacdir, vactype, extensions, survey, program, columns, verbose):
    """Internal driver: read and vstack all catalog files for a survey/program."""
    files = _catfiles(vacdir, vactype, survey, program)
    chunks = []
    for f in files:
        if verbose:
            print(f'Reading {f}')
        chunks.append(_read_extensions(f, extensions, columns=columns))
    cat = vstack(chunks) if len(chunks) > 1 else chunks[0]
    if verbose:
        print(f'  {len(cat):,} rows')
    return cat


# ---------------------------------------------------------------------------
# Public read functions
# ---------------------------------------------------------------------------

def read_fastspec(survey='main', program='dark', columns=None, verbose=False):
    """Read the fastspec DR2/Loa v1.0 VAC for a given survey and program.

    Joins the METADATA, SPECPHOT, and FASTSPEC extensions into a single Table.
    For main-bright and main-dark the 12 nside=1 healpix files are stacked
    transparently.

    Parameters
    ----------
    survey : str
        Survey name: 'main', 'sv1', 'sv2', 'sv3', or 'special'.
    program : str
        Program name: 'bright', 'dark', or 'backup'.
    columns : list of str, optional
        Columns to load from SPECPHOT and FASTSPEC. METADATA is always read
        in full. If None, all columns from all three extensions are returned.
    verbose : bool
        Print file names and row counts while reading.

    Returns
    -------
    astropy.table.Table
        Merged catalog. Key columns from METADATA: TARGETID, SURVEY, PROGRAM,
        HEALPIX, RA, DEC, Z, ZWARN, SPECTYPE, DESI_TARGET, BGS_TARGET,
        MWS_TARGET. From SPECPHOT: LOGMSTAR (h=1), SFR (h=1), AGE, TAUV, VDISP,
        DN4000. From FASTSPEC: emission-line fluxes, EWs, kinematics.
    """
    return _read_vac(
        _FASTSPEC_VACDIR, 'fastspec',
        ['METADATA', 'SPECPHOT', 'FASTSPEC'],
        survey, program, columns, verbose,
    )


def plot_style(talk=True, font_scale=1.0):
    """Set seaborn plot style and return (sns, color_palette).

    Parameters
    ----------
    talk : bool
        Use 'talk' context (larger fonts) if True, else 'paper'.
    font_scale : float
        Additional font scaling factor on top of the context default.

    Returns
    -------
    (sns, colors) : (module, list)
        The seaborn module (so callers need only one import) and the
        current color palette as a list.
    """
    import seaborn as sns
    sns.set(context='talk' if talk else 'paper',
            style='whitegrid', font_scale=font_scale)
    return sns, sns.color_palette()


def read_fastphot(survey='main', program='dark', columns=None, verbose=False):
    """Read the fastphot DR2/Loa v1.0 VAC for a given survey and program.

    Joins the METADATA and SPECPHOT extensions into a single Table.
    For main-bright and main-dark the 12 nside=1 healpix files are stacked
    transparently.

    Parameters
    ----------
    survey : str
        Survey name: 'main', 'sv1', 'sv2', 'sv3', or 'special'.
    program : str
        Program name: 'bright', 'dark', or 'backup'.
    columns : list of str, optional
        Columns to load from SPECPHOT. METADATA is always read in full.
        If None, all columns are returned.
    verbose : bool
        Print file names and row counts while reading.

    Returns
    -------
    astropy.table.Table
        Merged catalog. Key columns from METADATA: TARGETID, SURVEY, PROGRAM,
        HEALPIX, RA, DEC, Z, ZWARN, SPECTYPE, DESI_TARGET, BGS_TARGET,
        MWS_TARGET. From SPECPHOT: LOGMSTAR (h=1), SFR (h=1), AGE, TAUV, VDISP,
        DN4000.
    """
    return _read_vac(
        _FASTPHOT_VACDIR, 'fastphot',
        ['METADATA', 'SPECPHOT'],
        survey, program, columns, verbose,
    )
