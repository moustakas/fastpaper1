#!/usr/bin/env python
"""Build publication figures for the FastSpecFit DR2 paper.

Run from the repo root or from code/:

    python code/build-figures.py --compare-mstar [--verbose]

Each flag generates one figure written to tex/figures/.

"""
import sys, os, argparse, pdb
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import vstack, join

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import (read_fastspec, read_fastphot, plot_style,
                  corner_plot, hess_contours, DEFAULT_SPECPROD)

REPODIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR  = os.path.join(REPODIR, 'tex', 'figures')

# axis label for log stellar mass stored at h=1
MSTAR_LABEL = r'$\log_{10}\,(\mathcal{M}_{*}\,h^{-2}\,/\,\mathcal{M}_{\odot})$'


# ---------------------------------------------------------------------------
# Quality cuts
# ---------------------------------------------------------------------------

def good_galaxies(cat):
    """Standard boolean mask: good redshift and successful mass fit."""
    return (cat['ZWARN'] == 0) & (cat['Z'] > 0.001) & (cat['LOGMSTAR'] > 0)


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

    outfile = os.path.join(FIGDIR, 'sps-models.png')
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
        {'label': 'BGS',   'color': 'darkgreen', 'mask': is_bgs},
        {'label': 'LRG',   'color': 'darkred',   'mask': is_lrg},
        {'label': 'ELG',   'color': 'darkblue',  'mask': is_elg},
        {'label': 'Other', 'color': 'black',      'mask': is_other},
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

    outfile = os.path.join(FIGDIR, f'compare-mstar-{specprod}-{survey}{suffix}.png')
    fig.savefig(outfile, bbox_inches='tight', dpi=150)
    print(f'Wrote {outfile}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# compare-vdisp
# ---------------------------------------------------------------------------

def compare_vdisp(verbose=False):
    """2×2 comparison of FastSpecFit stellar velocity dispersions vs pPXF and Portsmouth.

    Data source: external/fpcatalog-iron-main-bright.fits (Ross et al. 2026).
    Top row: scatter (Hess + contours); bottom row: absolute residuals vs sigma_FS.
    Output: tex/figures/compare-vdisp.png
    """
    from matplotlib.gridspec import GridSpec

    extdir = os.path.join(REPODIR, 'external')
    catfile = os.path.join(extdir, 'fpcatalog-iron-main-bright.fits')
    if verbose:
        print(f'Reading {catfile}')
    import fitsio
    d = fitsio.read(catfile)

    fs   = d['VDISP'].astype(float)
    ppxf = d['PPXF_VDISP_FPCATALOG'].astype(float)
    port = d['PORTSMOUTH_SIGMA_STARS_FPCATALOG'].astype(float)
    ivar = d['VDISP_IVAR'].astype(float)

    good_fs   = (ivar > 0) & np.isfinite(fs) & (fs > 0)
    good_ppxf = good_fs & np.isfinite(ppxf) & (ppxf > 0)
    good_port = good_fs & np.isfinite(port) & (port > 0)

    fs_p, ppxf_p = fs[good_ppxf], ppxf[good_ppxf]
    fs_q, port_q = fs[good_port], port[good_port]

    sigrange = [75, 450]
    resrange = [-150, 150]
    bins_main = 60
    bins_res  = [60, 40]

    def _nmad(x):
        return 1.4826 * np.median(np.abs(x - np.median(x)))

    plot_style(talk=True, font_scale=0.85, palette='colorblind')

    fig = plt.figure(figsize=(11, 8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1],
                  hspace=0.05, wspace=0.08)

    ax_pp  = fig.add_subplot(gs[0, 0])
    ax_pt  = fig.add_subplot(gs[0, 1])
    ax_rpp = fig.add_subplot(gs[1, 0], sharex=ax_pp)
    ax_rpt = fig.add_subplot(gs[1, 1], sharex=ax_pt)

    # ---- top-left: FS vs pPXF ----
    hess_contours(ax_pp, fs_p, ppxf_p, sigrange, sigrange, bins=bins_main)
    ax_pp.plot(sigrange, sigrange, color='k', lw=1, ls='--')
    ax_pp.set_xlim(sigrange)
    ax_pp.set_ylim(sigrange)
    ax_pp.set_ylabel(r'$\sigma_\star$ [pPXF] (km s$^{-1}$)')
    ax_pp.tick_params(labelbottom=False)
    dpp = ppxf_p - fs_p
    ax_pp.text(0.04, 0.96,
               f'$N={len(fs_p):,}$\n'
               f'$\\Delta_{{\\rm med}}={np.median(dpp):+.1f}$ km s$^{{-1}}$\n'
               f'NMAD$={_nmad(dpp):.1f}$ km s$^{{-1}}$',
               transform=ax_pp.transAxes, fontsize='small',
               va='top', ha='left',
               bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=2))

    # ---- top-right: FS vs Portsmouth ----
    hess_contours(ax_pt, fs_q, port_q, sigrange, sigrange, bins=bins_main)
    ax_pt.plot(sigrange, sigrange, color='k', lw=1, ls='--')
    ax_pt.set_xlim(sigrange)
    ax_pt.set_ylim(sigrange)
    ax_pt.set_ylabel(r'$\sigma_\star$ [Portsmouth] (km s$^{-1}$)')
    ax_pt.yaxis.set_label_position('right')
    ax_pt.yaxis.tick_right()
    ax_pt.tick_params(labelbottom=False)
    dpt = port_q - fs_q
    ax_pt.text(0.04, 0.96,
               f'$N={len(fs_q):,}$\n'
               f'$\\Delta_{{\\rm med}}={np.median(dpt):+.1f}$ km s$^{{-1}}$\n'
               f'NMAD$={_nmad(dpt):.1f}$ km s$^{{-1}}$',
               transform=ax_pt.transAxes, fontsize='small',
               va='top', ha='left',
               bbox=dict(facecolor='white', edgecolor='none', alpha=0.75, pad=2))

    # ---- bottom-left: residuals pPXF ----
    hess_contours(ax_rpp, fs_p, dpp, sigrange, resrange, bins=bins_res)
    ax_rpp.axhline(0, color='k', lw=1, ls='--')
    ax_rpp.set_xlim(sigrange)
    ax_rpp.set_ylim(resrange)
    ax_rpp.set_xlabel(r'$\sigma_\star$ [FastSpecFit] (km s$^{-1}$)')
    ax_rpp.set_ylabel(r'$\Delta\sigma_\star$ (km s$^{-1}$)')

    # ---- bottom-right: residuals Portsmouth ----
    hess_contours(ax_rpt, fs_q, dpt, sigrange, resrange, bins=bins_res)
    ax_rpt.axhline(0, color='k', lw=1, ls='--')
    ax_rpt.set_xlim(sigrange)
    ax_rpt.set_ylim(resrange)
    ax_rpt.set_xlabel(r'$\sigma_\star$ [FastSpecFit] (km s$^{-1}$)')
    ax_rpt.yaxis.set_label_position('right')
    ax_rpt.yaxis.tick_right()
    ax_rpt.set_ylabel(r'$\Delta\sigma_\star$ (km s$^{-1}$)')

    fig.tight_layout()

    outfile = os.path.join(FIGDIR, 'compare-vdisp.png')
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
    parser.add_argument('--compare-vdisp', action='store_true',
                        help='Velocity dispersion comparison: FastSpecFit vs pPXF and Portsmouth.')
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

    if args.compare_vdisp:
        compare_vdisp(verbose=args.verbose)


if __name__ == '__main__':
    main()
