#!/usr/bin/env python
"""Build publication figures for the FastSpecFit DR2 paper.

Run from the repo root or from code/:

    python code/build-figures.py --compare-mstar [--verbose]

Each flag generates one figure written to tex/figures/.

"""
import sys, os, argparse, pdb
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, join

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import (read_fastspec, read_fastphot, plot_style,
                  corner_plot, hess_contours, DEFAULT_SPECPROD,
                  nmad, good_galaxies, good_redshift, jiyan_p1p3, make_class_cmap)

REPODIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR  = os.path.join(REPODIR, 'tex', 'figures')

# axis label for log stellar mass stored at h=1
MSTAR_LABEL = r'$\log_{10}\,(\mathcal{M}_{*}\,h^{-2}\,/\,\mathcal{M}_{\odot})$'

# Colorblind-friendly (Okabe-Ito) colors for DESI target classes.
# Used for contours; pass make_class_cmap(color) to hess_contours for the background.
TARGET_CLASS_COLORS = {
    'BGS':   '#1B7837',  # forest green
    #'BGS':   '#009E73',  # bluish green
    'LRG':   '#D55E00',  # vermillion
    'ELG':   '#0072B2',  # blue
    'QSO':   '#CC79A7',  # reddish purple
    'MWS':   '#56B4E9',  # sky blue
    'Other': '#999999',  # gray
}


# ---------------------------------------------------------------------------
# sps-models
# ---------------------------------------------------------------------------

def sps_models(verbose=False):
    """SPS template library: dust-free solar-metallicity templates, normalized
    to 10^10 M_sun at z=0.1, with DESI optical range and filter curves overlaid.

    """
    import matplotlib.ticker as ticker
    from matplotlib.patches import Rectangle
    from speclite import filters as speclite_filters
    from fastspecfit.util import C_LIGHT, MASSNORM, FLUXNORM
    from fastspecfit.continuum import build_stellar_continuum
    from fastspecfit.singlecopy import sc_data

    sc_data.initialize()
    photo = sc_data.photometry
    cosmo = sc_data.cosmology
    igm = sc_data.igm
    templates = sc_data.templates

    #info = sc_data.templates.info
    #def age_label(age_yr):
    #    return f'{age_yr/1e6:.0f} Myr' if age_yr < 1e9 else f'{age_yr/1e9:.0f} Gyr'

    agebins = ['0\u201330 Myr', '30\u2013100 Myr', '0.1\u20131.1 Gyr',
               '1.1\u20131.6 Gyr', '11.6\u201313.7 Gyr']

    plot_style(talk=True, font_scale=1.1, palette='colorblind')

    @ticker.FuncFormatter
    def major_formatter(x, pos):
        if 0.01 <= x < 0.1:
            return f'{x:.2f}'
        if 0.1 <= x < 1:
            return f'{x:.1f}'
        return f'{x:.0f}'


    zref = 0.1
    tauv = 0.1
    mstar = 1e10 # [Msun]
    logmstar = np.log10(mstar)
    filt = photo.filters['S']
    bands = photo.bands

    xlim  = [0.07, 40.]
    ylim  = [29.5, 3.]     # AB mag, inverted
    wdesi = [0.36, 0.98]   # DESI optical range [µm]
    ffact = -3.            # filter depth below ylim[0]

    fig, ax = plt.subplots(figsize=(9, 6))

    # shade DESI optical range
    ax.add_artist(Rectangle((wdesi[0], ylim[1]), np.ptp(wdesi), np.ptp(ylim),
                             fill=True, color='gray', alpha=0.2))

    # one line per template
    for ii in range(templates.ntemplates):

        coeff = np.zeros(templates.ntemplates)
        coeff[ii] = mstar / MASSNORM
        sedwave, sedmodel = build_stellar_continuum(
            coeff, tauv, zref, templates, cosmo, igm,
            dust_emission=True, vdisp=None)

        # [1e-17 erg/s/cm2/A --> maggies]
        abfactor = 10.**(0.4 * 48.6) * sedwave**2. / (C_LIGHT * 1e13) / FLUXNORM
        abmag = -2.5 * np.log10(sedmodel * abfactor) # [AB mag]

        ax.plot(templates.wave / 1e4, abmag, lw=2,
                label=agebins[ii]) # age_label(info['age'][ii]))


    # filter curves anchored below the plot bottom
    for ff, band in zip(filt, bands):
        ax.plot(ff.wavelength / 1e4,
                ffact * ff.response / ff.response.max() + ylim[0],
                color='k', alpha=0.8, lw=1)
        ax.text(ff.effective_wavelength.value / 1e4, -0.7 + ffact + ylim[0],
                band, fontsize=10, va='center', ha='center')

    ax.set_xscale('log')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([0.1, 0.3, 1, 3, 10, 30])
    ax.xaxis.set_major_formatter(major_formatter)
    ax.set_xlabel(r'Rest-frame Wavelength ($\mu$m)')
    ax.set_ylabel(
        f'AB mag ($10^{{{logmstar:.0f}}}\\,M_\\odot$ at $z = {zref:.1f}$)')
    ax.legend(fontsize=10, ncols=2, loc='upper left')
    #ax.text(0.97, 0.97, f'$\\tau_{{\\mathrm{{V}}}}={tauv:.1f}$\n$Z = Z_{{\\odot}}$',
    ax.text(0.97, 0.97, f'$Z = Z_{{\\odot}}$\n$\\tau_{{\\mathrm{{V}}}}={tauv:.1f}$',
            ha='right', va='top', fontsize=15, transform=ax.transAxes)

    fig.tight_layout()

    outfile = os.path.join(FIGDIR, 'sps-models.pdf')
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Wrote {outfile}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# compare-mstar
# ---------------------------------------------------------------------------

def target_class_groups(cat, survey):
    """Return a list of corner_plot group dicts split by DESI target class.

    Loads the survey-appropriate targetmask module explicitly so that bit names
    and column names match the actual catalog columns.

    Parameters
    ----------
    cat : astropy.table.Table
        Must include the survey-appropriate targeting bitmask columns.
    survey : str
        DESI survey flavor: 'sv1', 'sv3', 'main', or 'special'.

    Returns
    -------
    list of dict with keys 'label', 'color', 'mask' (boolean array).
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

    # BGS: use the dedicated BGS_TARGET column; sv3_desi_mask has no BGS_ANY summary bit
    is_bgs   = cat[bgs_col] != 0
    is_lrg   = (cat[desi_col] & int(desi_mask['LRG'])) != 0
    is_elg   = (cat[desi_col] & int(desi_mask['ELG'])) != 0
    is_other = ~(is_bgs | is_lrg | is_elg)

    return [
        {'label': 'BGS',   'color': TARGET_CLASS_COLORS['BGS'],   'mask': is_bgs},
        {'label': 'LRG',   'color': TARGET_CLASS_COLORS['LRG'],   'mask': is_lrg},
        {'label': 'ELG',   'color': TARGET_CLASS_COLORS['ELG'],   'mask': is_elg},
        {'label': 'Other', 'color': TARGET_CLASS_COLORS['Other'], 'mask': is_other},
    ]


def mstar_corner(cat, labels, groups=None, split_contours=False,
                 mstarlim=(6, 13), figsize=(10, 8)):
    """Corner plot comparing log stellar masses from N catalogs.

    Parameters
    ----------
    cat : astropy.table.Table
        One column per catalog, in the same order as labels.
    labels : list of str
        Axis label for each mass column.
    groups : list of dict or None
        If provided (from target_class_groups), diagonal panels show per-class
        colored step histograms.  Each dict must have keys 'label', 'color',
        and 'mask' (boolean index into cat).
    split_contours : bool
        If False (default), off-diagonal panels show the all-objects Hess
        diagram regardless of ``groups``.  If True, off-diagonal panels show
        per-class colored contours.
    mstarlim : tuple
        (min, max) plot range in log10(M/Msun).
    figsize : tuple of (float, float) or None
        Figure size in inches. Default is (3*N, 3*N).

    Returns
    -------
    matplotlib.figure.Figure

    """
    import math
    from matplotlib.ticker import MaxNLocator, FuncFormatter

    plot_style(talk=True, font_scale=0.7)

    n = len(labels)
    Xdata = np.column_stack([cat[c] for c in cat.colnames])

    corner_groups = None
    if groups is not None:
        corner_groups = [
            {'label': g['label'], 'color': g['color'],
             'data': Xdata[g['mask']]}
            for g in groups
        ]

    fig = corner_plot(
        Xdata, labels=labels, ranges=[mstarlim] * n,
        bins=60, unity=True, diag_ylabel='Number of Galaxies',
        groups=corner_groups, split_contours=split_contours, figsize=figsize,
    )

    # --- y-axis normalization on diagonal panels ---
    # scale to nearest order of magnitude below the tallest bin
    max_count = max(fig.axes[ii * n + ii].get_ylim()[1] for ii in range(n))
    exp = math.floor(math.log10(max_count))
    scale = 10 ** exp
    norm_ylabel = f'Number of Galaxies ($\\times10^{{{exp}}}$)'
    fmt = FuncFormatter(lambda x, _: f'{x / scale:g}')

    for ii in range(n):
        fig.axes[ii * n + ii].yaxis.set_major_formatter(fmt)
    fig.axes[0].set_ylabel(norm_ylabel)

    # twin y-axis on last diagonal panel, mirroring the first panel's scale
    a_last = fig.axes[(n - 1) * n + (n - 1)]
    yy = a_last.twinx()
    yy.set_ylim(a_last.get_ylim())
    yy.yaxis.set_major_formatter(fmt)
    yy.set_ylabel(norm_ylabel)

    # --- aligned top x-axis on each diagonal panel ---
    # compute tick positions once so bottom and top axes are guaranteed to match
    loc = MaxNLocator(5, integer=True)
    shared_ticks = np.asarray(loc.tick_values(*mstarlim))
    shared_ticks = shared_ticks[(shared_ticks >= mstarlim[0]) & (shared_ticks <= mstarlim[1])]
    tick_labels = [f'{t:g}' for t in shared_ticks]

    for ii in range(n):
        a = fig.axes[ii * n + ii]
        a.set_xticks(shared_ticks)
        xx = a.twiny()
        xx.set_xlim(mstarlim)
        xx.set_xticks(shared_ticks)
        xx.set_xticklabels(tick_labels, rotation=45)
        xx.set_xlabel(labels[ii])

    return fig


def compare_mstar(survey='sv3', specprod=DEFAULT_SPECPROD,
                  split_contours=False, verbose=False):
    """Corner plot: fastspec vs fastphot stellar masses.

    Both VACs store LOGMSTAR with h=1 (Planck 2018 cosmology, Chabrier IMF),
    so no cosmological correction is needed for this internal comparison.

    Diagonal panels always show per-target-class colored histograms.
    Off-diagonal panels show the all-objects Hess diagram by default; pass
    ``split_contours=True`` to show per-class colored contours instead.

    Parameters
    ----------
    survey : str
        'sv3' (default, single catalog files) or 'main' (split nside=1 files,
        much larger).
    split_contours : bool
        If False (default), off-diagonal panels show the all-objects Hess
        diagram.  If True, off-diagonal panels show per-class colored contours.
    """
    mstarlim = (6, 13)

    # --- read bright (BGS) and dark (LRG/ELG/QSO) programs for the survey ---
    spec_chunks, phot_chunks = [], []
    for program in ('bright', 'dark'):
        s = read_fastspec(survey, program, specprod=specprod,
                          columns=['LOGMSTAR'], verbose=verbose)
        spec_chunks.append(s[good_galaxies(s)])

        p = read_fastphot(survey, program, specprod=specprod,
                          columns=['LOGMSTAR'], verbose=verbose)
        phot_chunks.append(p[p['LOGMSTAR'] > 0])

    cat_spec = vstack(spec_chunks)
    cat_phot = vstack(phot_chunks)

    # rename before joining so columns are unambiguous
    cat_spec.rename_column('LOGMSTAR', 'LOGMSTAR_FASTSPEC')
    cat_phot = cat_phot['TARGETID', 'LOGMSTAR']
    cat_phot.rename_column('LOGMSTAR', 'LOGMSTAR_FASTPHOT')

    cat = join(cat_spec, cat_phot, keys='TARGETID', join_type='inner')
    print(f'{len(cat):,d} galaxies with good masses in both VACs')

    labels = (
        MSTAR_LABEL + '\n [fastspec]',
        MSTAR_LABEL + '\n [fastphot]',
    )
    # pass only the mass columns to the corner function
    mass_cat = cat['LOGMSTAR_FASTSPEC', 'LOGMSTAR_FASTPHOT']

    groups = target_class_groups(cat, survey)
    for g in groups:
        print(f"  {g['label']}: {g['mask'].sum():,d} galaxies")
    #suffix = '-split' if split_contours else ''
    suffix = ''

    fig = mstar_corner(mass_cat, labels, groups=groups,
                       split_contours=split_contours, mstarlim=mstarlim)

    outfile = os.path.join(FIGDIR, f'compare-mstar-{specprod}-{survey}{suffix}.pdf')
    fig.savefig(outfile, bbox_inches='tight', dpi=150)
    print(f'Wrote {outfile}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# compare-mstar-external
# ---------------------------------------------------------------------------

def compare_mstar_external(verbose=False):
    """3×3 grid: FastSpecFit stellar masses vs external catalogs, split by target class.

    Rows: Zou+CIGALE (loa), CIGALE-AGN (iron), GSWLC-X2 (bright/BGS only).
    Columns: BGS | LRG | ELG.  GSWLC-X2 populates BGS only; other cells hidden.
    Each panel uses a class-colored Hess background with thick colored contours.
    Output: tex/figures/compare-mstar-external.pdf

    """
    import fitsio
    from desitarget.sv3.sv3_targetmask import desi_mask as sv3_mask

    extdir = os.path.join(REPODIR, 'external')
    mstarlim = [6, 13]
    all_classes = ['BGS', 'LRG', 'ELG']

    catalogs = [
        dict(
            files=['zouhu-loa-sv3-bright.fits', 'zouhu-loa-sv3-dark.fits'],
            ext_col='LOGMSTAR_ZOUHU',
            label='Zou et al. (CIGALE)',
            classes=['BGS', 'LRG', 'ELG'],
        ),
        dict(
            files=['cigaleagn-iron-sv3-bright.fits', 'cigaleagn-iron-sv3-dark.fits'],
            ext_col='LOGMSTAR_CIGALEAGN',
            flag_col='FLAG_LOGMSTAR_CIGALEAGN',
            label='Siudek et al. (CIGALE-AGN)',
            classes=['BGS', 'LRG', 'ELG'],
        ),
        dict(
            files=['gswlcx2-sv3-bright.fits'],
            ext_col='LOGMSTAR_GSWLCX2',
            label='Salim et al. (GSWLC-X2)',
            classes=['BGS'],
        ),
    ]

    # bottom visible row per column (for x-axis label placement)
    bottom_row = {}
    for ci, cls in enumerate(all_classes):
        for ri in range(len(catalogs) - 1, -1, -1):
            if cls in catalogs[ri]['classes']:
                bottom_row[ci] = ri
                break

    plot_style(talk=True, font_scale=0.85, palette='colorblind')
    fig, axes = plt.subplots(3, 3, figsize=(13, 12))
    fig.subplots_adjust(hspace=0.08, wspace=0.08)

    for ri, cat in enumerate(catalogs):
        # load and stack all files for this row
        ref_l, ext_l, flag_l, bgs_l, desi_l, goodz_l = [], [], [], [], [], []
        for fn in cat['files']:
            path = os.path.join(extdir, fn)
            if verbose:
                print(f'Reading {path}')
            d = Table(fitsio.read(path))
            ref_l.append(d['LOGMSTAR'].astype(float))
            ext_l.append(d[cat['ext_col']].astype(float))
            if 'flag_col' in cat:
                flag_l.append(d[cat['flag_col']].astype(float))
            bgs_l.append(d['SV3_BGS_TARGET'].astype(np.int64))
            desi_l.append(d['SV3_DESI_TARGET'].astype(np.int64))
            goodz_l.append(good_redshift(d, 'sv3'))

        ref   = np.concatenate(ref_l)
        ext   = np.concatenate(ext_l)
        bgs   = np.concatenate(bgs_l)
        desi  = np.concatenate(desi_l)
        goodz = np.concatenate(goodz_l)
        flag  = np.concatenate(flag_l) if flag_l else None

        base = np.isfinite(ref) & (ref > 0) & np.isfinite(ext) & (ext > 0) & goodz
        if flag is not None:
            base &= (flag > 0.2) & (flag < 5.0)

        for ci, cls in enumerate(all_classes):
            ax = axes[ri, ci]

            if cls not in cat['classes']:
                ax.set_visible(False)
                continue

            if cls == 'BGS':
                cmask = base & (bgs != 0)
            elif cls == 'LRG':
                cmask = base & ((desi & int(sv3_mask['LRG'])) != 0)
            else:  # ELG
                cmask = base & ((desi & int(sv3_mask['ELG'])) != 0)

            r, e = ref[cmask], ext[cmask]
            delta = e - r
            color = TARGET_CLASS_COLORS[cls]

            hess_contours(ax, r, e, mstarlim, mstarlim, bins=60,
                          cmap=make_class_cmap(color),
                          contour_color=color, contour_lw=2.0)
            ax.plot(mstarlim, mstarlim, 'k--', lw=1.5, zorder=5)
            ax.set_xlim(mstarlim)
            ax.set_ylim(mstarlim)

            # tick labels: left column and bottom visible row only
            ax.tick_params(
                labelleft=(ci == 0),
                labelbottom=(ri == bottom_row[ci]),
            )

            # column title on top row
            if ri == 0:
                ax.set_title(cls, color=color, fontweight='bold')

            # x-axis label on bottom visible row of this column
            if ri == bottom_row[ci]:
                ax.set_xlabel(MSTAR_LABEL + '\n[FastSpecFit]')

            ax.text(0.04, 0.96,
                    f'$N={len(r):,}$\n'
                    f'$\\Delta_{{\\rm med}}={np.median(delta):+.3f}$\n'
                    f'NMAD$={nmad(delta):.3f}$',
                    transform=ax.transAxes, fontsize='small',
                    va='top', ha='left',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=2))

        # y-axis label (catalog name) on leftmost cell of each row
        axes[ri, 0].set_ylabel(MSTAR_LABEL + f"\n[{cat['label']}]")

    outfile = os.path.join(FIGDIR, 'compare-mstar-external.pdf')
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Wrote {outfile}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# mstar-redshift
# ---------------------------------------------------------------------------

def mstar_redshift(verbose=False):
    """M* vs. redshift for BGS, LRG, and ELG from sv3 (bright + dark programs).

    Three-panel figure (one per target class) with Hess background and smoothed
    contours; panels share the y-axis (stellar mass) and use the same redshift
    range.
    Output: tex/figures/mstar-redshift.pdf

    """
    zrange   = [-0.1, 1.8]
    mstarlim = [6, 13]

    chunks = []
    for program in ('bright', 'dark'):
        cat = read_fastspec('sv3', program, specprod=DEFAULT_SPECPROD,
                            columns=['LOGMSTAR'], verbose=verbose)
        chunks.append(cat[good_galaxies(cat, survey='sv3')])
    cat = vstack(chunks)
    if verbose:
        print(f'Total after quality cuts: {len(cat):,}')

    groups = [g for g in target_class_groups(cat, 'sv3')
              if g['label'] in ('BGS', 'LRG', 'ELG')]

    plot_style(talk=True, font_scale=0.85, palette='colorblind')
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    fig.subplots_adjust(wspace=0.05)

    for ax, g in zip(axes, groups):
        sub = cat[g['mask']]
        color = g['color']
        hess_contours(ax, sub['Z'], sub['LOGMSTAR'],
                      zrange, mstarlim,
                      bins=60, smooth=1.0,
                      cmap=make_class_cmap(color),
                      contour_color=color, contour_lw=2.0,
                      outlier_ms=2, background=True)
        ax.set_xlim(zrange)
        ax.set_ylim(mstarlim)
        ax.set_xlabel('Redshift')
        ax.set_title(g['label'], color=color, fontweight='bold')
        ax.text(0.96, 0.06, f"$N={len(sub):,}$",
                transform=ax.transAxes, fontsize='small',
                va='bottom', ha='right',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=2))
        if verbose:
            print(f"  {g['label']}: {len(sub):,} galaxies")

    axes[0].set_ylabel(MSTAR_LABEL)

    outfile = os.path.join(FIGDIR, 'mstar-redshift.pdf')
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Wrote {outfile}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# ewoii-dn4000
# ---------------------------------------------------------------------------

def ewoii_dn4000(verbose=False):
    """log10 EW([OII]) vs. Dn(4000) for BGS, LRG, and ELG from sv3 (bright + dark).

    Single panel: combined grayscale Hess background for all galaxies, with
    per-class colored contours overlaid.
    Output: tex/figures/ewoii-dn4000.pdf
    """
    from matplotlib.lines import Line2D

    dn4000_range = [0.9, 2.5]
    ewoii_range  = [-1, 3]

    cols = ['OII_3726_EW', 'OII_3729_EW', 'OII_3726_EW_IVAR', 'OII_3729_EW_IVAR',
            'DN4000', 'DN4000_IVAR']

    chunks = []
    for program in ('bright', 'dark'):
        cat = read_fastspec('sv3', program, specprod=DEFAULT_SPECPROD,
                            columns=cols, verbose=verbose)
        chunks.append(cat[good_galaxies(cat, survey='sv3')])
    cat = vstack(chunks)
    if verbose:
        print(f'Total after good_galaxies: {len(cat):,}')

    # S/N > 3 on individual EW measurements and on Dn(4000)
    with np.errstate(invalid='ignore'):
        sncut = ((cat['OII_3726_EW'] * np.sqrt(cat['OII_3726_EW_IVAR']) > 3) &
                 (cat['OII_3729_EW'] * np.sqrt(cat['OII_3729_EW_IVAR']) > 3) &
                 (cat['DN4000']      * np.sqrt(cat['DN4000_IVAR'])       > 3))
    cat = cat[sncut]
    if verbose:
        print(f'  After S/N>3 cuts: {len(cat):,}')

    dn4000    = np.array(cat['DN4000'], dtype=float)
    log_ewoii = np.log10(cat['OII_3726_EW'] + cat['OII_3729_EW'])

    groups = [g for g in target_class_groups(cat, 'sv3')
              if g['label'] in ('BGS', 'LRG', 'ELG')]

    plot_style(talk=True, font_scale=0.85, palette='colorblind')
    fig, ax = plt.subplots(figsize=(7, 6))

    # grayscale Hess background: all galaxies, no contours
    hess_contours(ax, dn4000, log_ewoii, dn4000_range, ewoii_range,
                  bins=60, smooth=1.0, cmap='Greys',
                  contour_levels=[], outlier_ms=0, background=True)

    # per-class colored contours (no background)
    handles = []
    for g in groups:
        mask  = g['mask']
        color = g['color']
        hess_contours(ax, dn4000[mask], log_ewoii[mask], dn4000_range, ewoii_range,
                      bins=60, smooth=1.0, contour_color=color, contour_lw=2.0,
                      outlier_ms=2, background=False)
        handles.append(Line2D([0], [0], color=color, lw=2,
                              label=f"{g['label']} ($N={mask.sum():,}$)"))
        if verbose:
            print(f"  {g['label']}: {mask.sum():,} galaxies")

    ax.set_xlim(dn4000_range)
    ax.set_ylim(ewoii_range)
    ax.set_xlabel(r'$D_n(4000)$')
    ax.set_ylabel(r'$\log_{10}\,\mathrm{EW}([\mathrm{O\,II}])\,(\AA)$')
    ax.legend(handles=handles, loc='upper right', framealpha=0.75)

    fig.tight_layout()

    outfile = os.path.join(FIGDIR, 'ewoii-dn4000.pdf')
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Wrote {outfile}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# bpt-agn
# ---------------------------------------------------------------------------

def bpt_agn(verbose=False):
    """BPT diagram and Ji & Yan (2020) P1-P3 projection for BGS from sv3/bright.

    Left panel : [OIII] 5007/Hβ vs [NII] 6584/Hα with Kauffmann et al. (2003)
                 and Kewley et al. (2001) demarcation lines.
    Right panel: P3 vs P1 (Ji & Yan 2020).
    Both panels show the same sample: BGS galaxies passing good_galaxies and
    S/N > 3 on all six diagnostic lines.
    Output: tex/figures/bpt-agn.pdf
    """
    bpt_xrange = [-2.0, 0.8]
    bpt_yrange = [-1.2, 1.5]
    p1_range   = [-0.3, 1.5]
    p3_range   = [-1.3, 1.0]

    cols = ['OIII_5007_FLUX', 'OIII_5007_FLUX_IVAR',
            'HBETA_FLUX',     'HBETA_FLUX_IVAR',
            'NII_6584_FLUX',  'NII_6584_FLUX_IVAR',
            'HALPHA_FLUX',    'HALPHA_FLUX_IVAR',
            'SII_6716_FLUX',  'SII_6716_FLUX_IVAR',
            'SII_6731_FLUX',  'SII_6731_FLUX_IVAR']

    cat = read_fastspec('sv3', 'bright', specprod=DEFAULT_SPECPROD,
                        columns=cols, verbose=verbose)
    cat = cat[good_galaxies(cat, survey='sv3')]

    groups   = target_class_groups(cat, 'sv3')
    bgs_mask = next(g['mask'] for g in groups if g['label'] == 'BGS')
    cat      = cat[bgs_mask]
    if verbose:
        print(f'BGS after good_galaxies: {len(cat):,}')

    # uniform S/N > 3 on all six lines
    with np.errstate(invalid='ignore'):
        sncut = (
            (cat['OIII_5007_FLUX'] * np.sqrt(cat['OIII_5007_FLUX_IVAR']) > 3) &
            (cat['HBETA_FLUX']     * np.sqrt(cat['HBETA_FLUX_IVAR'])     > 3) &
            (cat['NII_6584_FLUX']  * np.sqrt(cat['NII_6584_FLUX_IVAR'])  > 3) &
            (cat['HALPHA_FLUX']    * np.sqrt(cat['HALPHA_FLUX_IVAR'])    > 3) &
            (cat['SII_6716_FLUX']  * np.sqrt(cat['SII_6716_FLUX_IVAR'])  > 3) &
            (cat['SII_6731_FLUX']  * np.sqrt(cat['SII_6731_FLUX_IVAR'])  > 3)
        )
    cat = cat[sncut]
    if verbose:
        print(f'  After S/N>3 cuts: {len(cat):,}')

    log_nii_ha  = np.log10(cat['NII_6584_FLUX']  / cat['HALPHA_FLUX'])
    log_oiii_hb = np.log10(cat['OIII_5007_FLUX'] / cat['HBETA_FLUX'])
    log_sii_ha  = np.log10((cat['SII_6716_FLUX'] + cat['SII_6731_FLUX'])
                            / cat['HALPHA_FLUX'])

    p1, p3 = jiyan_p1p3(log_nii_ha, log_sii_ha, log_oiii_hb)

    color = TARGET_CLASS_COLORS['BGS']
    cmap  = make_class_cmap(color)

    plot_style(talk=True, font_scale=0.85, palette='colorblind')
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # --- left: BPT diagram ---
    ax = axes[0]
    hess_contours(ax, log_nii_ha, log_oiii_hb, bpt_xrange, bpt_yrange,
                  bins=60, smooth=1.0, cmap=cmap, contour_color=color,
                  contour_lw=2.0, outlier_ms=2, background=True)

    x = np.linspace(bpt_xrange[0], 0.04, 300)
    y = 0.61 / (x - 0.05) + 1.3
    m = (y >= bpt_yrange[0]) & (y <= bpt_yrange[1])
    ax.plot(x[m], y[m], 'k--', lw=1.5, label='Kauffmann et al. (2003)')

    x = np.linspace(bpt_xrange[0], 0.46, 300)
    y = 0.61 / (x - 0.47) + 1.19
    m = (y >= bpt_yrange[0]) & (y <= bpt_yrange[1])
    ax.plot(x[m], y[m], 'k-', lw=1.5, label='Kewley et al. (2001)')

    ax.set_xlim(bpt_xrange)
    ax.set_ylim(bpt_yrange)
    ax.set_xlabel(r'$\log_{10}\,[\mathrm{N\,II}]\,\lambda6584\,/\,\mathrm{H}\alpha$')
    ax.set_ylabel(r'$\log_{10}\,[\mathrm{O\,III}]\,\lambda5007\,/\,\mathrm{H}\beta$')
    ax.legend(loc='lower left', fontsize='small', framealpha=0.75)
    ax.text(0.96, 0.96, f'$N={len(cat):,}$',
            transform=ax.transAxes, fontsize='small', va='top', ha='right',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=2))

    # --- right: P3 vs P1 ---
    ax = axes[1]
    hess_contours(ax, p1, p3, p1_range, p3_range,
                  bins=60, smooth=1.0, cmap=cmap, contour_color=color,
                  contour_lw=2.0, outlier_ms=2, background=True)
    ax.set_xlim(p1_range)
    ax.set_ylim(p3_range)
    ax.set_xlabel(r'$P_1$')
    ax.set_ylabel(r'$P_3$')

    fig.tight_layout()

    outfile = os.path.join(FIGDIR, 'bpt-agn.pdf')
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Wrote {outfile}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# compare-vdisp
# ---------------------------------------------------------------------------

def compare_vdisp(verbose=False):
    """FastSpecFit stellar velocity dispersions vs pPXF: scatter and S/N residuals.

    Data source: external/fpcatalog-iron-main-bright.fits (Ross et al. 2026).
    Top panel: Hess + contours of sigma_FS vs sigma_pPXF.
    Bottom panel: absolute residuals Delta-sigma vs SNR_B (log-spaced 1–100).
    Output: tex/figures/compare-vdisp.pdf

    """
    from matplotlib.gridspec import GridSpec
    import fitsio

    extdir = os.path.join(REPODIR, 'external')
    catfile = os.path.join(extdir, 'fpcatalog-iron-main-bright.fits')
    if verbose:
        print(f'Reading {catfile}')
    d = fitsio.read(catfile)

    fs   = d['VDISP'].astype(float)
    ppxf = d['PPXF_VDISP_FPCATALOG'].astype(float)
    ivar = d['VDISP_IVAR'].astype(float)
    snr  = d['SNR_B'].astype(float)

    good = ((ivar > 0) & np.isfinite(fs) & (fs > 0)
            & np.isfinite(ppxf) & (ppxf > 0) & (snr > 0))
    fs_p, ppxf_p, snr_p = fs[good], ppxf[good], snr[good]
    dpp = ppxf_p - fs_p

    sigrange = [30, 400]
    resrange = [-150, 150]
    snrrange = [0., 2.]          # log10(SNR_B): 1 to 100
    snr_ticks = [1, 3, 10, 30, 100]

    color = '#E69F00'  # Okabe-Ito amber; neutral (not class-specific)
    #color = '#009E73'  # teal alternative
    cmap  = make_class_cmap(color)

    plot_style(talk=True, font_scale=0.85, palette='colorblind')

    fig = plt.figure(figsize=(7, 7))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[3, 1], hspace=0.35)

    ax_pp  = fig.add_subplot(gs[0])
    ax_rpp = fig.add_subplot(gs[1])

    # ---- top: FS vs pPXF ----
    hess_contours(ax_pp, fs_p, ppxf_p, sigrange, sigrange, bins=60,
                  cmap=cmap, contour_color=color, contour_lw=2.0)
    ax_pp.plot(sigrange, sigrange, color='k', lw=1.5, ls='--', zorder=5)
    ax_pp.set_xlim(sigrange)
    ax_pp.set_ylim(sigrange)
    ax_pp.set_xlabel(r'$\sigma$ [FastSpecFit] (km s$^{-1}$)')
    ax_pp.set_ylabel(r'$\sigma$ [pPXF] (km s$^{-1}$)')
    ax_pp.text(0.04, 0.96,
               f'$N={len(fs_p):,}$ [main/bright]\n'
               f'$\\Delta_{{\\rm med}}={np.median(dpp):+.1f}$ km s$^{{-1}}$\n'
               f'NMAD$={nmad(dpp):.1f}$ km s$^{{-1}}$',
               transform=ax_pp.transAxes, fontsize='small',
               va='top', ha='left',
               bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=2))

    # ---- bottom: residuals vs SNR_B (log x via log10 transform) ----
    hess_contours(ax_rpp, np.log10(snr_p), dpp, snrrange, resrange, bins=[50, 40],
                  cmap=cmap, contour_color=color, contour_lw=2.0)
    ax_rpp.axhline(0, color='k', lw=1.5, ls='--', zorder=5)
    ax_rpp.set_xlim(snrrange)
    ax_rpp.set_ylim(resrange)
    ax_rpp.set_xticks([np.log10(t) for t in snr_ticks])
    ax_rpp.set_xticklabels([str(t) for t in snr_ticks])
    ax_rpp.set_xlabel(r'$S/N_b$ (pixel$^{-1}$)')
    ax_rpp.set_ylabel(r'$\Delta\sigma$ (km s$^{-1}$)')

    outfile = os.path.join(FIGDIR, 'compare-vdisp.pdf')
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Wrote {outfile}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument('--sps-models', action='store_true',
                        help='SPS template library figure.')
    parser.add_argument('--compare-mstar', action='store_true',
                        help='Stellar mass comparison: fastspec vs fastphot.')
    parser.add_argument('--compare-mstar-external', action='store_true',
                        help='Stellar mass comparison: FastSpecFit vs external catalogs.')
    parser.add_argument('--mstar-redshift', action='store_true',
                        help='M* vs. redshift for BGS, LRG, ELG (sv3).')
    parser.add_argument('--ewoii-dn4000', action='store_true',
                        help='log EW([OII]) vs. Dn(4000) for BGS, LRG, ELG (sv3).')
    parser.add_argument('--bpt-agn', action='store_true',
                        help='BPT and Ji & Yan (2020) P1-P3 diagram for BGS (sv3/bright).')
    parser.add_argument('--compare-vdisp', action='store_true',
                        help='Velocity dispersion comparison: FastSpecFit vs pPXF.')
    parser.add_argument('--specprod', default=DEFAULT_SPECPROD,
                        help='Spectroscopic production name.')
    parser.add_argument('--main', action='store_true',
                        help='Use main-survey catalogs instead of sv3 (default).')
    parser.add_argument('--no-split-contours', dest='split_contours', action='store_false',
                        help='Do not split off-diagonal contours by target class.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress while reading catalogs.')
    args = parser.parse_args()

    survey = 'main' if args.main else 'sv3'
    os.makedirs(FIGDIR, exist_ok=True)

    if args.sps_models:
        sps_models(verbose=args.verbose)

    if args.compare_mstar:
        compare_mstar(survey=survey, specprod=args.specprod,
                      split_contours=args.split_contours, verbose=args.verbose)

    if args.compare_mstar_external:
        compare_mstar_external(verbose=args.verbose)

    if args.mstar_redshift:
        mstar_redshift(verbose=args.verbose)

    if args.ewoii_dn4000:
        ewoii_dn4000(verbose=args.verbose)

    if args.bpt_agn:
        bpt_agn(verbose=args.verbose)

    if args.compare_vdisp:
        compare_vdisp(verbose=args.verbose)


if __name__ == '__main__':
    main()
