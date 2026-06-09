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
from util import read_fastspec, read_fastphot, plot_style, DEFAULT_SPECPROD

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

def mstar_corner(cat, labels, mstarlim=(6, 13)):
    """Corner plot comparing log stellar masses from N catalogs.

    Parameters
    ----------
    cat : astropy.table.Table
        Must contain one column per label; column names are derived from
        labels via the caller (see compare_mstar).
    labels : list of str
        Axis label for each mass column, in the same order as the data columns.
    mstarlim : tuple
        (min, max) plot range in log10(M/Msun).

    Returns
    -------
    matplotlib.figure.Figure
    """
    import corner as cn
    from matplotlib.ticker import MaxNLocator

    sns, colors = plot_style(talk=True, font_scale=0.7)

    n = len(labels)
    lims = [mstarlim] * n
    max_n_ticks = 5

    col_names = list(cat.colnames)
    Xdata = np.column_stack([cat[c] for c in col_names])

    fig = cn.corner(
        Xdata,
        labels=labels,
        fill_contours=True,
        range=lims,
        labelpad=0.08,
        max_n_ticks=max_n_ticks,
        smooth=0.8,
        top_ticks=True,
        plot_density=True,
        levels=[0.5, 0.75, 0.95, 0.995],
        contour_kwargs={'colors': 'k'},
        show_titles=False,
        hist_kwargs={'color': 'k', 'lw': 2},
        color=colors[0],
        quiet=True,
    )

    # add top-axis ticks on each diagonal panel
    for ii in range(n):
        ax = fig.axes[ii + ii * n]
        xx = ax.twiny()
        xx.set_xlim(mstarlim)
        xx.set_xlabel(labels[ii])
        xx.xaxis.set_major_locator(MaxNLocator(max_n_ticks, prune='lower'))
        for lbl in xx.get_xticklabels() + xx.get_xticklabels(minor=True):
            lbl.set_rotation(45)

    fig.axes[0].set_ylabel('Number of Galaxies\n')

    # 1:1 line on every off-diagonal panel
    for row in range(1, n):
        for col in range(row):
            fig.axes[row * n + col].plot(mstarlim, mstarlim,
                                         color=colors[1], lw=1)

    return fig


def compare_mstar(specprod=DEFAULT_SPECPROD, verbose=False):
    """Corner plot: fastspec vs fastphot stellar masses (main survey, all targets).

    Both VACs store LOGMSTAR with h=1 (Planck 2018 cosmology, Chabrier IMF),
    so no cosmological correction is needed for this internal comparison.
    """
    mstarlim = (6, 13)

    # --- read main-bright (BGS) and main-dark (LRG/ELG/QSO) for each VAC ---
    spec_chunks, phot_chunks = [], []
    for program in ('bright', 'dark'):
        s = read_fastspec('main', program, specprod=specprod,
                          columns=['LOGMSTAR'], verbose=verbose)
        spec_chunks.append(s[good_galaxies(s)])

        p = read_fastphot('main', program, specprod=specprod,
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
        MSTAR_LABEL + '\n [FastSpec]',
        MSTAR_LABEL + '\n [FastPhot]',
    )
    # pass only the mass columns to the corner function
    mass_cat = cat['LOGMSTAR_FASTSPEC', 'LOGMSTAR_FASTPHOT']

    fig = mstar_corner(mass_cat, labels, mstarlim=mstarlim)

    outfile = os.path.join(FIGDIR, 'compare-mstar.png')
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
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress while reading catalogs.')
    args = parser.parse_args()

    os.makedirs(FIGDIR, exist_ok=True)

    if args.compare_mstar:
        compare_mstar(specprod=args.specprod, verbose=args.verbose)


if __name__ == '__main__':
    main()
