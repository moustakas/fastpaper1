#!/usr/bin/env python
"""Build publication figures for the FastSpecFit DR2 paper.

Run from the repo root or from code/:

    python code/build-figures.py --compare-mstar [--verbose]

Each flag generates one figure written to tex/figures/.

"""
import os, argparse, pdb
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import vstack, join

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import read_fastspec, read_fastphot, plot_style, corner_plot, DEFAULT_SPECPROD

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
# compare-mstar
# ---------------------------------------------------------------------------

def mstar_corner(cat, labels, mstarlim=(6, 13), figsize=(10, 6)):
    """Corner plot comparing log stellar masses from N catalogs.

    Parameters
    ----------
    cat : astropy.table.Table
        One column per catalog, in the same order as labels.
    labels : list of str
        Axis label for each mass column.
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

    fig = corner_plot(
        Xdata, labels=labels, ranges=[mstarlim] * n,
        bins=60, unity=True, diag_ylabel='Number of Galaxies',
        figsize=figsize,
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


def compare_mstar(survey='sv3', specprod=DEFAULT_SPECPROD, verbose=False):
    """Corner plot: fastspec vs fastphot stellar masses, all targets.

    Both VACs store LOGMSTAR with h=1 (Planck 2018 cosmology, Chabrier IMF),
    so no cosmological correction is needed for this internal comparison.

    Parameters
    ----------
    survey : str
        'sv3' (default, single catalog files) or 'main' (split nside=1 files,
        much larger).
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

    fig = mstar_corner(mass_cat, labels, mstarlim=mstarlim)

    outfile = os.path.join(FIGDIR, f'compare-mstar-{survey}.png')
    fig.savefig(outfile, bbox_inches='tight', dpi=150)
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
    parser.add_argument('--compare-mstar', action='store_true',
                        help='Stellar mass comparison: fastspec vs fastphot.')
    parser.add_argument('--specprod', default=DEFAULT_SPECPROD,
                        help='Spectroscopic production name.')
    parser.add_argument('--main', action='store_true',
                        help='Use main-survey catalogs instead of sv3 (default).')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress while reading catalogs.')
    args = parser.parse_args()

    survey = 'main' if args.main else 'sv3'
    os.makedirs(FIGDIR, exist_ok=True)

    if args.compare_mstar:
        compare_mstar(survey=survey, specprod=args.specprod, verbose=args.verbose)


if __name__ == '__main__':
    main()
