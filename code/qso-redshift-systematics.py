#!/usr/bin/env python
"""Quantify astrophysical systematics in QSO emission-line redshifts.

Standalone script (for a separate proposal, not the FastSpecFit DR2 paper)
that measures how far individual emission lines are offset in velocity from
the pipeline/Redrock redshift adopted for each fit -- the well-known QSO
phenomenology where high-ionization broad lines (e.g. CIV, Lyalpha) are
blueshifted relative to the systemic (narrow-line) redshift while
low-ionization broad lines (MgII, broad Balmer) track it much more closely.
Every line's fitted redshift is `redshift + VSHIFT/C_LIGHT`, where `redshift`
is the adopted input (pipeline) Z (fastspecfit/emlines.py), so VSHIFT is
already, by construction, the per-line residual velocity relative to the
pipeline redshift -- no separate comparison bookkeeping is needed.

Modeled closely on code/pipeline-diagnostics.py's --wavecal diagnostic (same
two-step, memory-bounded, one-healpix-file-at-a-time shape and the same
util.py helpers), but is a totally independent script with QSO-specific
sample selection, line categories, and x-axis (redshift instead of observed
wavelength).

Two-step process, as usual:

  --prepare  Build the long-format (one row per object x line) VSHIFT
             sample. Object-level sample is SPECTYPE=='QSO' (Redrock
             classification; effectively all z>1.7 objects), or
             SPECTYPE=='GALAXY' with a significant broad Balmer detection
             (Seyfert-1-type broad-line AGN not reclassified as QSO by
             Redrock). Restricted to survey=main (bright and dark
             separately, written to separate files); QSOs are
             overwhelmingly dark-time so the bright file will likely be
             small or empty. Also carries fastspecfit's own non-parametric
             MOMENT1/2/3 flux-patch moments (Angstrom / Angstrom**2 /
             Angstrom**3) for the four line groups that have them
             (CIV_1549, MGII_2800, HBETA, OIII_5007), as a
             fitting-model-independent cross-check on the Gaussian-derived
             VSHIFT values -- see emline-constraints.yaml's "moments"
             section.

  --figure   Median VSHIFT vs. redshift, one curve per line category (only
             reads the already-prepared external/qso-zsys-*.fits files, so
             it can run anywhere, not just on NERSC).

Output files:
    external/qso-zsys-{specprod}-{survey}-{program}.fits
    tex/figures/qso-redshift-systematics.pdf

Usage (from repo root or code/, on NERSC for --prepare):
    python code/qso-redshift-systematics.py --prepare [--survey main] [--verbose]
    python code/qso-redshift-systematics.py --figure [--verbose]

"""
import os, sys, argparse
import numpy as np
from astropy.table import Table, vstack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import _catfiles, _read_extensions, _DEFAULT_COLUMNS, DEFAULT_SPECPROD

REPODIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTDIR  = os.path.join(REPODIR, 'external')

PROGRAMS = ('bright', 'dark')

# Six line categories, grouped by ionization state and by whether the line
# arises in the broad- or narrow-line region. 'kind' ('narrow' or 'broad')
# controls which quality-cut tolerances apply below, since broad lines have
# much larger fitted-centroid uncertainties and a wider optimizer bound than
# narrow lines (emline-constraints.yaml: narrow vshift bounds are
# +/-500 km/s; global/UV-broad and narrow_broad's broad_balmer group are
# +/-2500 km/s).
_QSO_LINE_CATEGORIES = {
    'NLR forbidden': (['nev_3346', 'nev_3426', 'oii_3726', 'oii_3729', 'neiii_3869',
                       'oiii_4959', 'oiii_5007', 'nii_6548', 'nii_6584',
                       'sii_6716', 'sii_6731'], 'narrow'),
    'Narrow Balmer': (['halpha', 'hbeta', 'hgamma', 'hdelta'], 'narrow'),
    'Broad Balmer (low-ion BLR)': (['halpha_broad', 'hbeta_broad',
                                    'hgamma_broad', 'hdelta_broad'], 'broad'),
    'MgII (BLR)': (['mgii_2796', 'mgii_2803'], 'broad'),
    'Intermediate-ion BLR': (['ciii_1908', 'aliii_1857', 'siliii_1892',
                              'heii_1640', 'heii_4686'], 'broad'),
    'High-ion BLR': (['lyalpha', 'nv_1240', 'siliv_1396', 'civ_1549'], 'broad'),
}

# GALAXY-classified objects are only kept if they have a significant
# detection in this category (see prepare_qso_zsys).
_BROAD_BALMER_LABEL = 'Broad Balmer (low-ion BLR)'

CATEGORY_ORDER = list(_QSO_LINE_CATEGORIES)

# Non-parametric flux-weighted moments (Angstrom / Angstrom**2 / Angstrom**3
# of the line patch; see fastspecfit/data/emline-constraints.yaml's
# "moments" section) are only computed by fastspecfit for these four line
# groups -- MGII_2800 is a single joint measurement of the unresolved
# mgii_2796+mgii_2803 doublet, attached to both LINE rows below.
_MOMENT_GROUPS = {
    'CIV_1549':  ['civ_1549'],
    'MGII_2800': ['mgii_2796', 'mgii_2803'],
    'HBETA':     ['hbeta'],
    'OIII_5007': ['oiii_5007'],
}
_LINE_TO_MOMENT = {name.upper(): mcol
                   for mcol, names in _MOMENT_GROUPS.items() for name in names}
_MOMENT_COLS = ['MOMENT1', 'MOMENT2', 'MOMENT3',
               'MOMENT1_IVAR', 'MOMENT2_IVAR', 'MOMENT3_IVAR']

# Curated default for qso_zsys_figure(): a small set of strong, easy-to-explain
# lines pulled out of the full _QSO_LINE_CATEGORIES groupings (e.g. only
# OII_3729 of the OII doublet, only 3 of the 6 High/Intermediate-ion BLR
# lines, split into their own CIII]/CIV curves) so the legend stays short.
# Pass --all-lines to fall back to the full CATEGORY_ORDER grouping instead.
CURATED_LINE_GROUPS = {
    'Forbidden (NLR)':     ['OII_3729', 'OIII_5007'],
    'Narrow Balmer (NLR)': ['HALPHA', 'HBETA', 'HGAMMA'],
    'Broad Balmer (BLR)':  ['HALPHA_BROAD', 'HBETA_BROAD', 'HGAMMA_BROAD'],
    'Mg II (BLR)':         ['MGII_2796', 'MGII_2803'],
    'C III] (BLR)':        ['CIII_1908'],
    'C IV (BLR)':          ['CIV_1549'],
}

# Hard optimizer bounds on VSHIFT [km/s] per kind, from emline-constraints.yaml
# (narrow kinematic groups: +/-500; broad/UV-broad and narrow_broad's
# broad_balmer group: +/-2500). A handful of boundary-pinned fits report
# pathologically tiny formal errors that would otherwise dominate any
# inverse-variance-weighted statistic downstream (same issue documented in
# pipeline-diagnostics.py's prepare_wavecal).
_BOUND_NARROW = 499.0
_BOUND_BROAD  = 2499.0


def _line_categories(verbose=False):
    """QSO emission-line categories with runtime-looked-up restwaves.

    Restwaves are read from fastspecfit's own emlines.ecsv (not hardcoded),
    mirroring pipeline-diagnostics.py's _narrow_only_lines(), so this stays
    in sync with the line list fastspecfit actually fits.

    Returns
    -------
    categories : dict
        {category_label: (list_of_LINE_NAME_UPPER, kind)}, kind is 'narrow'
        or 'broad'.
    restwave : dict
        {LINE_NAME_UPPER: restwave [Angstrom, vacuum]}
    moment_restwave : dict
        {MOMENT_GROUP_LABEL: restwave [Angstrom, vacuum]}, the mean restwave
        fastspecfit itself uses for each moment group (see emlines.py:
        ``restwave = np.mean(self.line_table['restwave'][moment_lines])``),
        needed to convert MOMENT1 [observed-frame Angstrom] to a velocity
        offset downstream.
    """
    import fastspecfit

    fsf_data = os.path.join(os.path.dirname(fastspecfit.__file__), 'data')
    linetable = Table.read(os.path.join(fsf_data, 'emlines.ecsv'))
    restwave_lower = {row['name']: float(row['restwave']) for row in linetable}

    categories = {label: ([name.upper() for name in names], kind)
                 for label, (names, kind) in _QSO_LINE_CATEGORIES.items()}
    restwave = {name: restwave_lower[name.lower()]
               for names, _ in categories.values() for name in names}
    moment_restwave = {mcol: float(np.mean([restwave_lower[n] for n in names]))
                       for mcol, names in _MOMENT_GROUPS.items()}

    if verbose:
        for label, (names, kind) in categories.items():
            print(f'  {label} ({kind}): {names}')
        print(f'  moment groups: {moment_restwave}')

    return categories, restwave, moment_restwave


# ---------------------------------------------------------------------------
# QSO emission-line redshift-systematics diagnostic
# ---------------------------------------------------------------------------

def prepare_qso_zsys(survey='main', specprod=DEFAULT_SPECPROD, min_snr=5.0,
                     max_vshift_err_narrow=25.0, max_vshift_err_broad=300.0,
                     verbose=False):
    """Build the long-format QSO per-line VSHIFT-vs-redshift sample.

    Object-level cuts:
      - ZWARN==0 and Z>0.001
      - SPECTYPE=='QSO', or SPECTYPE=='GALAXY' with a significant broad
        Balmer detection (see below)

    Line-level cuts (per object x line), tolerance set by the category's
    'narrow'/'broad' kind (see _QSO_LINE_CATEGORIES):
      - FLUX_IVAR > 0 and FLUX*sqrt(FLUX_IVAR) > min_snr
      - VSHIFT_IVAR > 0 and 1/sqrt(VSHIFT_IVAR) < max_vshift_err_{narrow,broad}
      - |VSHIFT| below the kinematic group's hard optimizer bound

    A GALAXY-classified object is only kept if it has at least one row that
    survives the line-level cuts in the "Broad Balmer (low-ion BLR)"
    category -- broad-Balmer significance is not a separately reconstructed
    flag, it falls directly out of the same S/N cut applied to every other
    line (a fixed/zero broad-Balmer amplitude, from the narrow_only
    kinematic profile, never passes FLUX*sqrt(FLUX_IVAR) > min_snr).

    Output columns: TARGETID, SURVEY, PROGRAM, HEALPIX, SPECTYPE, CATEGORY,
    LINE, RESTWAVE, Z, VSHIFT, VSHIFT_ERR, FLUX, FLUX_IVAR, plus the
    non-parametric MOMENT1/2/3 (+ each's _IVAR) and MOMENT_RESTWAVE columns
    (Angstrom / Angstrom**2 / Angstrom**3 of the flux-weighted line patch;
    see fastspecfit/data/emline-constraints.yaml's "moments" section).
    fastspecfit only computes these for four line groups -- CIV_1549,
    MGII_2800 (the unresolved mgii_2796+mgii_2803 doublet, attached to both
    MgII LINE rows), HBETA (narrow only), and OIII_5007 -- so they are NaN
    on every other row.

    """
    categories, restwave, moment_restwave = _line_categories(verbose=verbose)

    extra_cols = []
    for names, _ in categories.values():
        for line in names:
            extra_cols += [f'{line}_FLUX', f'{line}_FLUX_IVAR',
                           f'{line}_VSHIFT', f'{line}_VSHIFT_IVAR']
    for mcol in _MOMENT_GROUPS:
        extra_cols += [f'{mcol}_{c}' for c in _MOMENT_COLS]
    columns = list(set(extra_cols) | _DEFAULT_COLUMNS)

    for program in PROGRAMS:
        files = _catfiles(specprod, 'fastspec', survey, program)
        outfile = os.path.join(EXTDIR, f'qso-zsys-{specprod}-{survey}-{program}.fits')

        chunks = []
        n_obj_total = n_obj_pass = n_rows = 0
        for f in files:
            if verbose:
                print(f'Reading {f}')
            cat = _read_extensions(f, ['METADATA', 'FASTSPEC'], columns=columns)
            n_obj_total += len(cat)

            spectype = np.asarray(cat['SPECTYPE'])
            good = ((np.asarray(cat['ZWARN']) == 0) &
                   (np.asarray(cat['Z'], dtype=float) > 0.001) &
                   ((spectype == 'QSO') | (spectype == 'GALAXY')))
            if not np.any(good):
                continue
            cat = cat[good]

            targetid = np.asarray(cat['TARGETID'])
            healpix  = np.asarray(cat['HEALPIX'])
            z        = np.asarray(cat['Z'], dtype=float)
            spectype = np.asarray(cat['SPECTYPE'])

            file_chunks = []
            broad_ok = set()
            for label, (lines, kind) in categories.items():
                max_vshift_err = max_vshift_err_narrow if kind == 'narrow' else max_vshift_err_broad
                bound = _BOUND_NARROW if kind == 'narrow' else _BOUND_BROAD

                for line in lines:
                    flux   = np.asarray(cat[f'{line}_FLUX'],       dtype=float)
                    fivar  = np.asarray(cat[f'{line}_FLUX_IVAR'],  dtype=float)
                    vshift = np.asarray(cat[f'{line}_VSHIFT'],      dtype=float)
                    vivar  = np.asarray(cat[f'{line}_VSHIFT_IVAR'], dtype=float)

                    with np.errstate(invalid='ignore', divide='ignore'):
                        snr        = flux * np.sqrt(fivar)
                        vshift_err = 1. / np.sqrt(vivar)

                    sel = ((fivar > 0) & (snr > min_snr) &
                          (vivar > 0) & (vshift_err < max_vshift_err) &
                          (np.abs(vshift) < bound) & np.isfinite(vshift))
                    n = int(np.sum(sel))
                    if n == 0:
                        continue

                    if label == _BROAD_BALMER_LABEL:
                        broad_ok.update(targetid[sel].tolist())

                    row = {
                        'TARGETID':   targetid[sel],
                        'SURVEY':     np.full(n, survey),
                        'PROGRAM':    np.full(n, program),
                        'HEALPIX':    healpix[sel],
                        'SPECTYPE':   spectype[sel],
                        'CATEGORY':   np.full(n, label),
                        'LINE':       np.full(n, line),
                        'RESTWAVE':   np.full(n, restwave[line], dtype='f4'),
                        'Z':          z[sel].astype('f4'),
                        'VSHIFT':     vshift[sel].astype('f4'),
                        'VSHIFT_ERR': vshift_err[sel].astype('f4'),
                        'FLUX':       flux[sel].astype('f4'),
                        'FLUX_IVAR':  fivar[sel].astype('f4'),
                    }

                    mcol = _LINE_TO_MOMENT.get(line)
                    if mcol is not None:
                        for c in _MOMENT_COLS:
                            row[c] = np.asarray(cat[f'{mcol}_{c}'], dtype='f4')[sel]
                        row['MOMENT_RESTWAVE'] = np.full(n, moment_restwave[mcol], dtype='f4')
                    else:
                        for c in _MOMENT_COLS + ['MOMENT_RESTWAVE']:
                            row[c] = np.full(n, np.nan, dtype='f4')

                    file_chunks.append(Table(row))

            if not file_chunks:
                continue

            file_table = vstack(file_chunks)
            broad_ok_arr = np.array(sorted(broad_ok), dtype=targetid.dtype) if broad_ok else \
                          np.array([], dtype=targetid.dtype)
            keep = ((np.asarray(file_table['SPECTYPE']) == 'QSO') |
                   np.isin(np.asarray(file_table['TARGETID']), broad_ok_arr))
            file_table = file_table[keep]
            if len(file_table) == 0:
                continue

            n_obj_pass += len(np.unique(file_table['TARGETID']))
            n_rows += len(file_table)
            chunks.append(file_table)

        if verbose:
            print(f'  {survey}/{program}: {n_obj_total:,} objects -> '
                  f'{n_obj_pass:,} objects pass cuts -> {n_rows:,} line rows')

        if not chunks:
            print(f'  {survey}/{program}: no rows passed cuts, skipping')
            continue

        out = vstack(chunks)
        out.write(outfile, overwrite=True)
        print(f'  {survey}/{program}: wrote {len(out):,} rows -> {outfile}')

    print('Done.')


def _logz_edges(nbins, xrange):
    """Bin edges in redshift, equal-width in log(1+z).

    Using log(1+z) rather than log(z) handles xrange[0]==0 for free (no
    log-of-zero issue) while still compressing the dense low-z bulk and
    giving the sparse high-z QSO tail proportionally more, better-populated
    bins than equal-width-in-z binning would.
    """
    return np.expm1(np.linspace(np.log1p(xrange[0]), np.log1p(xrange[1]), nbins + 1))


def _running_binstat_edges(x, y, edges, q_lo=25, q_hi=75, min_count=5):
    """Like util.running_binstat, but for explicit (e.g. log-spaced) bin
    edges rather than an equal-width bin count + range. Duplicated locally,
    rather than extending util.running_binstat, to keep this script totally
    independent.
    """
    from scipy.stats import binned_statistic

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    kw = dict(bins=edges)

    cnt, _, _ = binned_statistic(x, y, statistic='count', **kw)
    med, _, _ = binned_statistic(x, y, statistic='median', **kw)
    lo,  _, _ = binned_statistic(x, y, statistic=lambda v: np.percentile(v, q_lo), **kw)
    hi,  _, _ = binned_statistic(x, y, statistic=lambda v: np.percentile(v, q_hi), **kw)

    centers = 0.5 * (edges[:-1] + edges[1:])
    sparse = cnt < min_count
    for arr in (med, lo, hi):
        arr[sparse] = np.nan

    return centers, med, lo, hi


def _binned_median_sem(x, y, edges, min_count=200):
    """Running median of y vs x, with an NMAD-based standard error per bin.

    SEM_median ~ 1.2533 * NMAD(y) / sqrt(N); see pipeline-diagnostics.py's
    identically-named helper for the full derivation. Duplicated here
    (rather than imported) to keep this script totally independent.

    Returns
    -------
    centers, median, sem, count : ndarray, ndarray, ndarray, ndarray
        Bins with count < min_count have median and sem set to NaN.
    """
    from scipy.stats import binned_statistic
    from util import nmad

    kw = dict(bins=edges)
    cnt, _, _ = binned_statistic(x, y, statistic='count', **kw)
    med, _, _ = binned_statistic(x, y, statistic='median', **kw)
    mad, _, _ = binned_statistic(x, y, statistic=nmad, **kw)

    centers = 0.5 * (edges[:-1] + edges[1:])
    sem = 1.2533 * mad / np.sqrt(np.maximum(cnt, 1))
    sparse = cnt < min_count
    med[sparse] = np.nan
    sem[sparse] = np.nan

    return centers, med, sem, cnt


def qso_zsys_figure(specprod=DEFAULT_SPECPROD, survey='main', nbins=40,
                    xrange=(0., 5.), min_count=20, all_lines=False, verbose=False):
    """QSO redshift-systematics figure: median per-line-group VSHIFT vs.
    redshift, as a proxy for astrophysical (not instrumental) emission-line
    redshift systematics -- e.g. the well-known blueshift of high-ionization
    broad lines (CIV, Lyalpha) relative to the systemic (narrow-line)
    redshift.

    Reads external/qso-zsys-{specprod}-{survey}-{bright,dark}.fits (written
    by prepare_qso_zsys) and plots one running-median VSHIFT-vs-Z curve per
    line group, using _running_binstat_edges for the median + IQR shaded
    band (log(1+z)-spaced bins throughout, via _logz_edges, so the sparse
    high-z QSO tail gets wider, better-populated bins instead of thinning
    out to near-empty linear-z bins), matching the panel layout convention
    in pipeline-diagnostics.py's wavecal_residuals.

    By default (``all_lines=False``) plots the small curated set of strong
    lines in CURATED_LINE_GROUPS, chosen for a short, easy-to-explain
    legend. Pass ``all_lines=True`` to instead plot the full
    CATEGORY_ORDER grouping (every line in _QSO_LINE_CATEGORIES combined
    per category).

    With ``verbose=True``, also prints a quotable per-line-group summary to
    stdout: the median of the per-bin running medians (a single
    representative velocity offset, treating every redshift bin equally so
    it isn't dominated by whichever bin has the most objects) and the NMAD
    scatter *of those per-bin medians* as an approximate spread -- i.e. how
    much the trend itself moves with redshift, not the (much larger)
    object-to-object scatter within a bin (that's what the shaded IQR band
    shows).

    Output: tex/figures/qso-redshift-systematics.pdf

    """
    import matplotlib.pyplot as plt
    from util import plot_style, nmad

    chunks = []
    for program in PROGRAMS:
        infile = os.path.join(EXTDIR, f'qso-zsys-{specprod}-{survey}-{program}.fits')
        if not os.path.isfile(infile):
            print(f'  Missing {infile}, skipping.')
            continue
        if verbose:
            print(f'Reading {infile}')
        chunks.append(Table.read(infile))
    if not chunks:
        raise FileNotFoundError('No qso-zsys input files found; run --prepare first.')
    t = vstack(chunks)

    # Table.read() returns FITS string columns as fixed-width bytes, not
    # str, so t['CATEGORY']/t['LINE'] == label would silently match nothing.
    t['CATEGORY'] = np.asarray(t['CATEGORY']).astype(str)
    t['LINE']     = np.asarray(t['LINE']).astype(str)

    if all_lines:
        group_list = [(label, np.asarray(t['CATEGORY'] == label)) for label in CATEGORY_ORDER]
    else:
        group_list = [(label, np.isin(t['LINE'], lines))
                     for label, lines in CURATED_LINE_GROUPS.items()]

    edges = _logz_edges(nbins, xrange)

    sns, colors = plot_style(talk=True, font_scale=0.9, palette='colorblind')
    fig, (ax, axn) = plt.subplots(
        2, 1, figsize=(9, 7), sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08})

    ymax = 50.  # widened below if the IQR band needs more room
    summary_rows = []
    for icat, (label, sel) in enumerate(group_list):
        n_avail = int(np.sum(sel))
        if n_avail == 0:
            if verbose:
                print(f'  {label}: no rows, skipping')
            continue

        x = np.asarray(t['Z'][sel],      dtype=float)
        y = np.asarray(t['VSHIFT'][sel], dtype=float)
        color = colors[icat % len(colors)]

        centers, med, qlo, qhi = _running_binstat_edges(x, y, edges, min_count=min_count)
        finite = np.isfinite(med)
        # fill zorders are below every median line's zorder so lines stay on
        # top regardless of draw order (same convention as wavecal_residuals).
        ax.fill_between(centers[finite], qlo[finite], qhi[finite],
                        color=color, alpha=0.2, zorder=icat + 1)
        ax.plot(centers[finite], med[finite], color=color, lw=2.2, label=label, zorder=icat + 11)
        if np.any(finite):
            ymax = max(ymax, 1.05 * np.nanmax(np.abs(np.concatenate([qlo[finite], qhi[finite]]))))

            # "median of the medians": a single representative offset for
            # this line group, treating every redshift bin equally (so it
            # isn't dominated by whichever bin happens to hold the most
            # objects), with the NMAD scatter *across those per-bin medians*
            # as an approximate spread -- i.e. how much the trend itself
            # moves with redshift, not the (much larger) object-to-object
            # scatter within a bin (that's what the shaded IQR band shows).
            med_of_med = float(np.median(med[finite]))
            n_bins_used = int(np.sum(finite))
            spread = float(nmad(med[finite])) if n_bins_used >= 2 else np.nan
            zlo, zhi = float(centers[finite].min()), float(centers[finite].max())
            summary_rows.append((label, med_of_med, spread, n_bins_used, zlo, zhi))

        _, _, sem, count = _binned_median_sem(x, y, edges, min_count=min_count)
        if verbose:
            finite_sem = np.isfinite(sem)
            if np.any(finite_sem):
                print(f'  {label}: {n_avail:,} rows; SEM on the median ranges '
                      f'{np.nanmin(sem[finite_sem]):.2f}-{np.nanmax(sem[finite_sem]):.2f} km/s across bins')

    if verbose and summary_rows:
        width = max(len(label) for label, *_ in summary_rows)
        print('\nLine-group summary (median of the per-bin running medians, '
             '+/- NMAD spread of those per-bin medians across redshift):')
        for label, med_of_med, spread, n_bins_used, zlo, zhi in summary_rows:
            spread_str = f'{spread:.1f}' if np.isfinite(spread) else 'n/a'
            print(f'  {label:<{width}} : {med_of_med:+7.1f} km/s  (NMAD {spread_str:>5} km/s '
                  f'over {n_bins_used} bins, z = {zlo:.2f}-{zhi:.2f})')

    ax.axhline(0, color='0.4', lw=1, ls='--', zorder=10)
    ax.set_ylim(-ymax, ymax)
    ax.set_ylabel('Velocity Shift from Redrock (km/s)')
    ax.legend(loc='upper left', fontsize=13.3, framealpha=0.85, ncol=3)
    ax.set_xlim(xrange)

    # Bottom panel: unique-QSO counts per bin (not row counts, which would
    # double-count objects with more than one line surviving cuts in a
    # given group) -- this represents the overall sample's redshift
    # distribution, independent of which line group is plotted above.
    _, uniq_idx = np.unique(np.asarray(t['TARGETID']), return_index=True)
    z_uniq = np.asarray(t['Z'])[uniq_idx].astype(float)
    cnt_obj, _ = np.histogram(z_uniq, bins=edges)
    xc = 0.5 * (edges[:-1] + edges[1:])

    axn.plot(xc, cnt_obj, color='k', lw=1.2, drawstyle='steps-mid')
    axn.set_yscale('log')
    axn.set_ylabel('Number of QSOs')
    axn.set_xlabel(r'Redshift')

    outfile = os.path.join(REPODIR, 'tex', 'figures', 'qso-redshift-systematics.pdf')
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
    parser.add_argument('--prepare', action='store_true',
                        help='Build the QSO redshift-systematics diagnostic sample.')
    parser.add_argument('--figure', action='store_true',
                        help='Make the VSHIFT-vs-redshift figure from an existing sample.')
    parser.add_argument('--all-lines', action='store_true',
                        help='--figure only: plot the full CATEGORY_ORDER grouping (every line '
                             'in _QSO_LINE_CATEGORIES) instead of the default curated '
                             'CURATED_LINE_GROUPS subset.')
    parser.add_argument('--survey', default='main', choices=['sv3', 'main'],
                        help='DESI survey.')
    parser.add_argument('--specprod', default=DEFAULT_SPECPROD,
                        help='Spectroscopic production name.')
    parser.add_argument('--min-snr', type=float, default=5.0,
                        help='Minimum per-line flux S/N.')
    parser.add_argument('--max-vshift-err-narrow', type=float, default=25.0,
                        help='Maximum formal VSHIFT uncertainty [km/s] for narrow-line categories.')
    parser.add_argument('--max-vshift-err-broad', type=float, default=300.0,
                        help='Maximum formal VSHIFT uncertainty [km/s] for broad-line categories.')
    parser.add_argument('--min-count', type=int, default=20,
                        help='--figure only: minimum rows/bin for a line group\'s running-median '
                             'point to be plotted (sparser bins are skipped, leaving a gap).')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress while reading and cutting.')
    args = parser.parse_args()

    if args.prepare:
        prepare_qso_zsys(survey=args.survey, specprod=args.specprod,
                         min_snr=args.min_snr,
                         max_vshift_err_narrow=args.max_vshift_err_narrow,
                         max_vshift_err_broad=args.max_vshift_err_broad,
                         verbose=args.verbose)

    if args.figure:
        qso_zsys_figure(specprod=args.specprod, survey=args.survey,
                        min_count=args.min_count,
                        all_lines=args.all_lines, verbose=args.verbose)


if __name__ == '__main__':
    main()
