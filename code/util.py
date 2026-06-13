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

_DESI_VAC = '/dvs_ro/cfs/cdirs/desi/vac'

# Per-specprod catalog directory paths.  Add entries here as new specprods
# are released; set a value to None when the path is not yet known.
_SPECPROD_CONFIG = {
    'loa': {
        'fastspec': f'{_DESI_VAC}/dr2/fastspecfit/loa/v1.0/catalogs',
        'fastphot': f'{_DESI_VAC}/dr2/fastphot/loa/v1.0/catalogs',
    },
    'iron': {
        'fastspec': f'{_DESI_VAC}/dr1/fastspecfit/iron/v3.0/catalogs',
        'fastphot': f'{_DESI_VAC}/dr1/fastphot/loa/v1.0/catalogs',
    },
    'fuji': {
        'fastspec': f'{_DESI_VAC}/edr/fastspecfit/fuji/v3.2/catalogs',
        'fastphot': None,
    },
}

DEFAULT_SPECPROD = 'loa'

# survey+program combinations split into 12 nside=1 healpix files
_SPLIT_COMBOS = {('main', 'bright'), ('main', 'dark')}

# Columns always returned from METADATA regardless of the caller's ``columns``
# argument.  All survey-specific targeting bit names are listed here; columns
# absent from a given file are silently skipped by _read_extensions.
_DEFAULT_COLUMNS = frozenset({
    'TARGETID', 'SURVEY', 'PROGRAM', 'HEALPIX',
    'RA', 'DEC', 'Z', 'ZERR', 'ZWARN', 'DELTACHI2',
    'COADD_FIBERSTATUS',
    # OII doublet columns needed for the ELG redshift quality cut
    'OII_3726_FLUX', 'OII_3729_FLUX', 'OII_3726_FLUX_IVAR', 'OII_3729_FLUX_IVAR',
    # main-survey targeting bits
    'DESI_TARGET', 'BGS_TARGET', 'MWS_TARGET', 'SCND_TARGET', 'ETC_TARGET',
    # SV1
    'SV1_DESI_TARGET', 'SV1_BGS_TARGET', 'SV1_MWS_TARGET',
    'SV1_SCND_TARGET', 'SV1_ETC_TARGET',
    # SV2
    'SV2_DESI_TARGET', 'SV2_BGS_TARGET', 'SV2_MWS_TARGET',
    'SV2_SCND_TARGET', 'SV2_ETC_TARGET',
    # SV3
    'SV3_DESI_TARGET', 'SV3_BGS_TARGET', 'SV3_MWS_TARGET',
    'SV3_SCND_TARGET', 'SV3_ETC_TARGET',
    # CMX
    'CMX_TARGET',
})


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

            #if i == 0:
            #    # METADATA: always full
            #    read_cols = None
            if columns is not None:
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
        Merged catalog. Key columns always present: TARGETID, SURVEY, PROGRAM,
        HEALPIX, RA, DEC, Z, ZERR, ZWARN, SPECTYPE, and all survey-specific
        targeting bit columns that exist in the file (DESI_TARGET, BGS_TARGET,
        MWS_TARGET, SV3_DESI_TARGET, etc.). Additional columns from SPECPHOT
        and FASTSPEC are controlled by ``columns``.

    """
    if columns is not None:
        columns = list(set(columns) | _DEFAULT_COLUMNS)
    return _read_vac(
        specprod, 'fastspec',
        ['METADATA', 'SPECPHOT', 'FASTSPEC'],
        survey, program, columns, verbose,
    )


def corner_plot(plotdata, labels, ranges, bins=50, truths=None, sigmas=None,
                titles=None, unity=False, diag_ylabel='N',
                contour_levels=None, contour_lw=1.5, smooth=1.0,
                cmap='Blues', show_residuals=True, groups=None,
                split_contours=False,
                figsize=None, suptitle='', subplots_adjust=None):
    """Corner-style N×N plot: histograms on the diagonal, Hess+contours on the lower triangle.

    Adapted from fastspecfit.qa._corner_plot for catalog-scale datasets.
    Off-diagonal panels show a log-stretched 2D histogram (Hess diagram) with
    smoothed density contours at specified cumulative enclosed fractions overlaid.

    Parameters
    ----------
    plotdata : array_like, shape (nsamples, ndim)
        Data array, one column per parameter. Ignored when ``groups`` is provided.
    labels : list of str
        Axis labels, one per parameter.
    ranges : list of (lo, hi) tuples
        Axis limits for each parameter.
    bins : int
        Number of bins along each axis for the 2D histogram and diagonal histograms.
    truths : list of float or None
        Reference values: vertical lines on diagonal, crosshairs on off-diagonal.
        Skipped if None. Not used in groups mode.
    sigmas : list of float or None
        1-sigma uncertainties drawn as dashed lines on the diagonal.
        Only used when truths is not None.
    titles : list of str or None
        Titles above each diagonal histogram. Skipped if None.
    unity : bool
        Draw a dashed 1:1 reference line on each off-diagonal panel.
    diag_ylabel : str
        Y-axis label on the leftmost diagonal histogram.
    contour_levels : list of float or None
        Cumulative enclosed fractions at which to draw contour lines, e.g.
        [0.5, 0.75, 0.95, 0.995]. Default: [0.5, 0.75, 0.95, 0.995].
    contour_lw : float
        Line width for contours.
    smooth : float
        Gaussian smoothing sigma (in bins) applied to the 2D histogram before
        computing contour levels. Set to 0 to disable.
    cmap : str
        Colormap for the Hess diagram (single-group mode only).
    show_residuals : bool
        If True, annotate each off-diagonal panel with Δ = col_y − col_x
        statistics.  In single-group mode: one-line text annotation.  In
        groups mode: a per-group legend with colored Line2D handles.
    groups : list of dict or None
        When provided, diagonal panels show per-group step histograms instead
        of a single gray histogram.  Each dict must have keys ``'data'``
        (array shape (n, ndim)), ``'color'`` (str), and ``'label'`` (str).
        Off-diagonal behavior is controlled by ``split_contours``.
    split_contours : bool
        Only meaningful when ``groups`` is provided.  If False (default),
        off-diagonal panels show the all-data Hess diagram (requires valid
        ``plotdata``).  If True, off-diagonal panels show per-group contours
        with no Hess background.
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
    from matplotlib.colors import LogNorm
    from scipy.ndimage import gaussian_filter

    if contour_levels is None:
        contour_levels = [0.5, 0.75, 0.95, 0.995]

    use_groups = groups is not None and len(groups) > 0
    use_groups_offdiag = use_groups and split_contours
    ndim = len(labels)

    # plotdata is needed for off-diagonal Hess whenever split_contours is False
    if not use_groups or not split_contours:
        plotdata = np.asarray(plotdata)

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
                if use_groups:
                    for grp in groups:
                        a.hist(grp['data'][:, xi], bins=bins, range=(lo_x, hi_x),
                               color=grp['color'], histtype='step', lw=2)
                else:
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
                if use_groups_offdiag:
                    # Per-group contours (no Hess background)
                    for grp in groups:
                        H, xedges, yedges = np.histogram2d(
                            grp['data'][:, xi], grp['data'][:, yi],
                            bins=bins, range=[[lo_x, hi_x], [lo_y, hi_y]],
                        )
                        xc = 0.5 * (xedges[:-1] + xedges[1:])
                        yc = 0.5 * (yedges[:-1] + yedges[1:])
                        Hs = gaussian_filter(H, smooth) if smooth > 0 else H
                        flat = np.sort(Hs.flatten())[::-1]
                        cumsum = np.cumsum(flat)
                        total = cumsum[-1]
                        if total > 0 and contour_levels:
                            lvls = []
                            for frac in contour_levels:
                                idx = np.searchsorted(cumsum, frac * total)
                                lvls.append(flat[min(idx, len(flat) - 1)])
                            lvls = sorted(v for v in set(lvls) if v > 0)
                            if lvls:
                                a.contour(xc, yc, Hs.T, levels=lvls,
                                          colors=grp['color'], linewidths=contour_lw)
                    if show_residuals:
                        from matplotlib.lines import Line2D
                        handles = []
                        for grp in groups:
                            resid = grp['data'][:, yi] - grp['data'][:, xi]
                            med = np.median(resid)
                            mu  = np.mean(resid)
                            sig = np.std(resid)
                            lbl = (f"{grp['label']}: "
                                   f"$\\Delta={med:+.3f}\\,({mu:+.3f}\\pm{sig:.3f})$")
                            handles.append(Line2D([0], [0], color=grp['color'],
                                                  lw=2, label=lbl))
                        a.legend(handles=handles, loc='upper left',
                                 fontsize='x-small', framealpha=0.75,
                                 handlelength=1.5)
                else:
                    # Hess diagram: log-stretched 2D histogram
                    H, xedges, yedges = np.histogram2d(
                        plotdata[:, xi], plotdata[:, yi],
                        bins=bins, range=[[lo_x, hi_x], [lo_y, hi_y]],
                    )
                    xc = 0.5 * (xedges[:-1] + xedges[1:])
                    yc = 0.5 * (yedges[:-1] + yedges[1:])
                    a.pcolormesh(xedges, yedges, H.T, norm=LogNorm(vmin=1), cmap=cmap)
                    Hs = gaussian_filter(H, smooth) if smooth > 0 else H
                    flat = np.sort(Hs.flatten())[::-1]
                    cumsum = np.cumsum(flat)
                    total = cumsum[-1]
                    if total > 0 and contour_levels:
                        lvls = []
                        for frac in contour_levels:
                            idx = np.searchsorted(cumsum, frac * total)
                            lvls.append(flat[min(idx, len(flat) - 1)])
                        lvls = sorted(v for v in set(lvls) if v > 0)
                        if lvls:
                            a.contour(xc, yc, Hs.T, levels=lvls,
                                      colors='k', linewidths=contour_lw)
                    if truths is not None:
                        a.axvline(truths[xi], color='C0', lw=1, ls='-', alpha=0.75)
                        a.axhline(truths[yi], color='C0', lw=1, ls='-', alpha=0.75)
                    if show_residuals:
                        resid = plotdata[:, yi] - plotdata[:, xi]
                        med = np.median(resid)
                        mu  = np.mean(resid)
                        sig = np.std(resid)
                        txt = f'$\\Delta={med:+.3f}\\,({mu:+.3f}\\pm{sig:.3f})$'
                        a.text(0.04, 0.96, txt, transform=a.transAxes,
                               fontsize='x-small', va='top', ha='left',
                               bbox=dict(facecolor='white', edgecolor='none',
                                         alpha=0.75, pad=2))

                if unity:
                    lo = max(lo_x, lo_y)
                    hi = min(hi_x, hi_y)
                    a.plot([lo, hi], [lo, hi], color='k', lw=1, ls='-')

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


def hess_contours(ax, x, y, xrange, yrange, bins=50, smooth=1.0,
                  contour_levels=None, cmap='Blues', contour_lw=1.5,
                  contour_color='k', outlier_ms=2, background=True):
    """Hess diagram with smoothed cumulative contours, matching corner_plot style.

    Points that fall outside the outermost contour are drawn as individual small
    markers (size ``outlier_ms``); pass ``outlier_ms=0`` to suppress them.
    The pcolormesh background is rasterized so PDF output stays compact.
    Pass ``background=False`` to draw contours and outlier scatter only (no
    pcolormesh), which is useful when overlaying multiple samples on one axes.
    """
    from matplotlib.colors import LogNorm
    from scipy.ndimage import gaussian_filter

    if contour_levels is None:
        contour_levels = [0.5, 0.75, 0.95, 0.995]

    H, xedges, yedges = np.histogram2d(x, y, bins=bins,
                                        range=[xrange, yrange])
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    if background:
        ax.pcolormesh(xedges, yedges, H.T, norm=LogNorm(vmin=1), cmap=cmap,
                      rasterized=True, zorder=1)

    Hs = gaussian_filter(H, smooth) if smooth > 0 else H
    flat = np.sort(Hs.flatten())[::-1]
    cumsum = np.cumsum(flat)
    total = cumsum[-1]
    lvls = []
    if total > 0 and contour_levels:
        for frac in contour_levels:
            idx = np.searchsorted(cumsum, frac * total)
            lvls.append(flat[min(idx, len(flat) - 1)])
        lvls = sorted(v for v in set(lvls) if v > 0)
        if lvls:
            ax.contour(xc, yc, Hs.T, levels=lvls,
                       colors=contour_color, linewidths=contour_lw, zorder=3)

    # individual points outside the outermost contour, drawn above the Hess
    # background (zorder=2) but below contours (zorder=3)
    if lvls and outlier_ms > 0:
        in_range = ((x >= xrange[0]) & (x <= xrange[1]) &
                    (y >= yrange[0]) & (y <= yrange[1]))
        xi = np.clip(np.digitize(x[in_range], xedges) - 1, 0, len(xc) - 1)
        yi = np.clip(np.digitize(y[in_range], yedges) - 1, 0, len(yc) - 1)
        outside = Hs[xi, yi] < lvls[0]
        ax.scatter(x[in_range][outside], y[in_range][outside],
                   s=outlier_ms, c=[contour_color], alpha=0.8,
                   linewidths=0, zorder=2, rasterized=True)


def nmad(x):
    """Normalized median absolute deviation: 1.4826 * median(|x - median(x)|)."""
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def _good_fiberstatus(cat):
    """Fiber status mask: allow only RESTRICTED and VARIABLE bits."""
    from desispec.maskbits import fibermask
    okmask = fibermask.mask('RESTRICTED|VARIABLE')
    return (cat['COADD_FIBERSTATUS'] & okmask) == cat['COADD_FIBERSTATUS']


def good_redshift(cat, survey, fiberstatus_cut=True, ignore_emline=False):
    """Per-class redshift quality mask.

    Applies the standard DESI per-class cuts:
      BGS  — ZWARN==0 & DELTACHI2>40
      LRG  — ZWARN==0 & Z<1.5 & DELTACHI2>15
      ELG  — OII S/N cut: log10(OII_FLUX * sqrt(OII_FLUX_IVAR)) > 0.9 - 0.2*log10(DELTACHI2)
              (falls back to ZWARN==0 & DELTACHI2>15 if OII columns are absent or
              ``ignore_emline=True``)
      Other — ZWARN==0 & Z>0.001

    Parameters
    ----------
    cat : astropy.table.Table
        Must include ZWARN, DELTACHI2, Z, all survey-appropriate targeting bit
        columns, and COADD_FIBERSTATUS (if fiberstatus_cut=True).
    survey : str
        DESI survey flavor: 'sv1', 'sv3', 'main', or 'special'.
    fiberstatus_cut : bool
        Apply fiber status mask (RESTRICTED and VARIABLE bits allowed).
    ignore_emline : bool
        Skip the OII emission-line cut for ELGs and use DELTACHI2>15 instead.

    Returns
    -------
    numpy.ndarray of bool
    """
    if survey == 'sv3':
        from desitarget.sv3.sv3_targetmask import desi_mask
        desi_col, bgs_col = 'SV3_DESI_TARGET', 'SV3_BGS_TARGET'
    elif survey == 'sv1':
        from desitarget.sv1.sv1_targetmask import desi_mask
        desi_col, bgs_col = 'SV1_DESI_TARGET', 'SV1_BGS_TARGET'
    elif survey in ('main', 'special'):
        from desitarget.targets import desi_mask
        desi_col, bgs_col = 'DESI_TARGET', 'BGS_TARGET'
    else:
        raise ValueError(f'Unknown survey {survey!r}; expected sv1, sv3, main, or special.')

    is_bgs = cat[bgs_col] != 0
    is_lrg = (cat[desi_col] & int(desi_mask['LRG'])) != 0
    is_elg = (cat[desi_col] & int(desi_mask['ELG'])) != 0

    good_fs = _good_fiberstatus(cat) if fiberstatus_cut else np.ones(len(cat), bool)

    good_bgs = (cat['ZWARN'] == 0) & (cat['DELTACHI2'] > 40) & good_fs
    good_lrg = (cat['ZWARN'] == 0) & (cat['Z'] < 1.5) & (cat['DELTACHI2'] > 15) & good_fs

    _OII_COLS = ('OII_3726_FLUX', 'OII_3729_FLUX',
                 'OII_3726_FLUX_IVAR', 'OII_3729_FLUX_IVAR')
    has_oii = all(c in cat.colnames for c in _OII_COLS)
    if not ignore_emline and has_oii and is_elg.any():
        oii_flux = cat['OII_3726_FLUX'] + cat['OII_3729_FLUX']
        ivar_sum = cat['OII_3726_FLUX_IVAR'] + cat['OII_3729_FLUX_IVAR']
        with np.errstate(divide='ignore', invalid='ignore'):
            # harmonic sum of IVARs; zero wherever both (or either) are zero
            oii_ivar = np.where(
                ivar_sum > 0,
                cat['OII_3726_FLUX_IVAR'] * cat['OII_3729_FLUX_IVAR'] / ivar_sum,
                0.0)
            oii_snr = np.log10(oii_flux * np.sqrt(oii_ivar))
            dc2     = np.log10(np.maximum(cat['DELTACHI2'], 1e-10))
        good_elg = ((oii_flux > 0) & (oii_ivar > 0) &
                    (oii_snr > 0.9 - 0.2 * dc2) & good_fs)
    else:
        good_elg = (cat['ZWARN'] == 0) & (cat['DELTACHI2'] > 15) & good_fs

    good_other = (cat['ZWARN'] == 0) & (cat['Z'] > 0.001)

    return ((is_bgs & good_bgs) |
            (is_lrg & good_lrg) |
            (is_elg & good_elg) |
            (~(is_bgs | is_lrg | is_elg) & good_other))


def good_galaxies(cat, survey=None, fiberstatus_cut=True):
    """Boolean mask: good redshift quality and successful stellar mass fit.

    When ``survey`` is provided, applies per-class DELTACHI2 and fiber status
    cuts via ``good_redshift``; otherwise falls back to ZWARN==0 & Z>0.001.
    Always requires LOGMSTAR > 0.
    """
    if survey is not None:
        good_z = good_redshift(cat, survey, fiberstatus_cut=fiberstatus_cut)
    else:
        good_z = (cat['ZWARN'] == 0) & (cat['Z'] > 0.001)
    if 'LOGMSTAR' in cat.colnames:
        good_z = good_z & (cat['LOGMSTAR'] > 0)
    return good_z


def make_class_cmap(color, lighten=0.3):
    """White-to-lightened-color colormap for Hess diagram backgrounds.

    The endpoint is mixed ``lighten`` fraction toward white so that the
    densest bins stay visually lighter than the full class color, keeping
    the contours (drawn in the full color) clearly visible.
    """
    from matplotlib.colors import LinearSegmentedColormap, to_rgb
    rgb = np.array(to_rgb(color))
    light_end = tuple((1.0 - lighten) * rgb + lighten * np.ones(3))
    return LinearSegmentedColormap.from_list('', ['white', light_end])


def plot_style(talk=True, font_scale=1.0, palette=None):
    """Set seaborn plot style and return (sns, color_palette).

    Parameters
    ----------
    talk : bool
        Use 'talk' context (larger fonts) if True, else 'paper'.
    font_scale : float
        Additional font scaling factor on top of the context default.
    palette : str or list or None
        Color palette passed to seaborn.set_palette().  None keeps the
        seaborn default.  Pass 'colorblind' for the Okabe-Ito palette,
        which is safe for deuteranopia, protanopia, and tritanopia.

    Returns
    -------
    (sns, colors) : (module, list)
        The seaborn module (so callers need only one import) and the
        current color palette as a list.
    """
    import seaborn as sns
    sns.set(context='talk' if talk else 'paper',
            style='whitegrid', font_scale=font_scale)
    if palette is not None:
        sns.set_palette(palette)
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
        Merged catalog. Key columns always present: TARGETID, SURVEY, PROGRAM,
        HEALPIX, RA, DEC, Z, ZERR, ZWARN, SPECTYPE, and all survey-specific
        targeting bit columns that exist in the file (DESI_TARGET, BGS_TARGET,
        MWS_TARGET, SV3_DESI_TARGET, etc.). Additional columns from SPECPHOT
        are controlled by ``columns``.

    """
    if columns is not None:
        columns = list(set(columns) | _DEFAULT_COLUMNS)
    return _read_vac(
        specprod, 'fastphot',
        ['METADATA', 'SPECPHOT'],
        survey, program, columns, verbose,
    )
