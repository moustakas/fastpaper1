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

_DESI_VAC = '/global/cfs/cdirs/desi/vac'

# Per-specprod catalog directory paths.  Add entries here as new specprods
# are released; set a value to None when the path is not yet known.
_SPECPROD_CONFIG = {
    'loa': {
        'fastspec': f'{_DESI_VAC}/dr2/fastspecfit/loa/v1.0/catalogs',
        'fastphot': f'{_DESI_VAC}/dr2/fastphot/loa/v1.0/catalogs',
    },
    'iron': {
        'fastspec': None,  # TBD (DR1)
        'fastphot': None,  # TBD (DR1)
    },
    'fuji': {
        'fastspec': None,  # TBD (EDR)
        'fastphot': None,  # TBD (EDR)
    },
}

DEFAULT_SPECPROD = 'loa'

# survey+program combinations split into 12 nside=1 healpix files
_SPLIT_COMBOS = {('main', 'bright'), ('main', 'dark')}


def _catfiles(specprod, vactype, survey, program):
    """Return sorted list of catalog paths for a specprod/survey/program combination."""
    cfg = _SPECPROD_CONFIG.get(specprod)
    if cfg is None:
        raise ValueError(f'Unknown specprod {specprod!r}. '
                         f'Known specprods: {list(_SPECPROD_CONFIG)}')
    vacdir = cfg[vactype]
    if vacdir is None:
        raise ValueError(f'VAC path not yet configured for specprod={specprod!r}, '
                         f'vactype={vactype!r}')

    basename = f'{vactype}-{specprod}-{survey}-{program}'
    if (survey, program) in _SPLIT_COMBOS:
        pattern = os.path.join(vacdir, f'{basename}-nside1-hp??.fits')
        files = sorted(glob(pattern))
        if not files:
            raise FileNotFoundError(f'No files found matching {pattern}')
        return files
    path = os.path.join(vacdir, f'{basename}.fits')
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


def _read_vac(specprod, vactype, extensions, survey, program, columns, verbose):
    """Internal driver: read and vstack all catalog files for a survey/program."""
    files = _catfiles(specprod, vactype, survey, program)
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

def read_fastspec(survey='main', program='dark', specprod=DEFAULT_SPECPROD,
                  columns=None, verbose=False):
    """Read the fastspec VAC for a given specprod, survey, and program.

    Joins the METADATA, SPECPHOT, and FASTSPEC extensions into a single Table.
    For main-bright and main-dark the 12 nside=1 healpix files are stacked
    transparently.

    Parameters
    ----------
    survey : str
        Survey name: 'main', 'sv1', 'sv2', 'sv3', or 'special'.
    program : str
        Program name: 'bright', 'dark', or 'backup'.
    specprod : str
        Spectroscopic production name (default: 'loa'). Must be a key in
        _SPECPROD_CONFIG.
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
        specprod, 'fastspec',
        ['METADATA', 'SPECPHOT', 'FASTSPEC'],
        survey, program, columns, verbose,
    )


def corner_plot(plotdata, labels, ranges, bins=50, truths=None, sigmas=None,
                titles=None, unity=False, diag_ylabel='N',
                figsize=None, suptitle='', subplots_adjust=None):
    """Corner-style N×N plot: histograms on the diagonal, 2D density on the lower triangle.

    Adapted from fastspecfit.qa._corner_plot for catalog-scale datasets.
    Off-diagonal panels use hexbin density maps instead of scatter plots.

    Parameters
    ----------
    plotdata : array_like, shape (nsamples, ndim)
        Data array, one column per parameter.
    labels : list of str
        Axis labels, one per parameter.
    ranges : list of (lo, hi) tuples
        Axis limits for each parameter.
    bins : int
        Number of histogram bins on the diagonal.
    truths : list of float or None
        Reference values: vertical lines on diagonal, crosshairs on off-diagonal.
        Skipped if None.
    sigmas : list of float or None
        1-sigma uncertainties drawn as dashed lines on the diagonal.
        Only used when truths is not None.
    titles : list of str or None
        Titles above each diagonal histogram. Skipped if None.
    unity : bool
        Draw a 1:1 reference line on each off-diagonal panel.
    diag_ylabel : str
        Y-axis label on the leftmost diagonal histogram.
    figsize : tuple of (float, float) or None
        Figure size in inches as (width, height). Default is (3*ndim, 3*ndim),
        minimum 6×6.
    suptitle : str
        Figure suptitle.
    subplots_adjust : dict or None
        Forwarded to fig.subplots_adjust; if None, tight_layout is used.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt

    plotdata = np.asarray(plotdata)
    ndim = plotdata.shape[1]
    if figsize is None:
        _size = max(3 * ndim, 6)
        figsize = (_size, _size)
    fig, axes = plt.subplots(ndim, ndim, figsize=figsize)
    ax = np.array(axes).reshape((ndim, ndim))

    for yi in range(ndim):
        for xi in range(ndim):
            a = ax[yi, xi]
            if xi > yi:
                a.set_visible(False)
                continue

            lo_x, hi_x = ranges[xi]
            lo_y, hi_y = ranges[yi]

            if xi == yi:
                a.hist(plotdata[:, xi], bins=bins, range=(lo_x, hi_x),
                       color='gray', alpha=0.75, edgecolor='k')
                if truths is not None:
                    a.axvline(truths[xi], color='C0', lw=2, ls='-')
                    if sigmas is not None:
                        a.axvline(truths[xi] + sigmas[xi], color='C0', lw=1, ls='--')
                        a.axvline(truths[xi] - sigmas[xi], color='C0', lw=1, ls='--')
                if titles is not None:
                    a.set_title(titles[xi], fontsize=8)
                a.set_xlim(lo_x, hi_x)
                if xi == 0:
                    a.set_ylabel(diag_ylabel)
                else:
                    a.tick_params(labelleft=False)
            else:
                a.hexbin(plotdata[:, xi], plotdata[:, yi], gridsize=50,
                         bins='log', cmap='Blues', mincnt=1,
                         extent=(lo_x, hi_x, lo_y, hi_y))
                if truths is not None:
                    a.axvline(truths[xi], color='C0', lw=1, ls='-', alpha=0.75)
                    a.axhline(truths[yi], color='C0', lw=1, ls='-', alpha=0.75)
                if unity:
                    lo = max(lo_x, lo_y)
                    hi = min(hi_x, hi_y)
                    a.plot([lo, hi], [lo, hi], color='k', lw=1, ls='--')
                a.set_xlim(lo_x, hi_x)
                a.set_ylim(lo_y, hi_y)
                if xi == 0:
                    a.set_ylabel(labels[yi])
                else:
                    a.tick_params(labelleft=False)

            if yi == ndim - 1:
                a.set_xlabel(labels[xi])
            else:
                a.tick_params(labelbottom=False)

    if suptitle:
        fig.suptitle(suptitle)
    if subplots_adjust:
        fig.subplots_adjust(**subplots_adjust)
    else:
        fig.tight_layout()

    return fig


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


def read_fastphot(survey='main', program='dark', specprod=DEFAULT_SPECPROD,
                  columns=None, verbose=False):
    """Read the fastphot VAC for a given specprod, survey, and program.

    Joins the METADATA and SPECPHOT extensions into a single Table.
    For main-bright and main-dark the 12 nside=1 healpix files are stacked
    transparently.

    Parameters
    ----------
    survey : str
        Survey name: 'main', 'sv1', 'sv2', 'sv3', or 'special'.
    program : str
        Program name: 'bright', 'dark', or 'backup'.
    specprod : str
        Spectroscopic production name (default: 'loa'). Must be a key in
        _SPECPROD_CONFIG.
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
        specprod, 'fastphot',
        ['METADATA', 'SPECPHOT'],
        survey, program, columns, verbose,
    )
