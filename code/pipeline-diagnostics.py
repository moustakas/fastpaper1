#!/usr/bin/env python
"""Build diagnostic samples for the DR2 pipeline-diagnostics section (Sec. 5.5).

Two independent diagnostics, each with its own prepare_*() function:

  --wavecal  Wavelength-calibration residuals (Sec. 5.5.1): a long-format
             (one row per object x line) table of emission-line VSHIFT values
             for objects where fastspecfit's narrow_only kinematic final pass
             was adopted, i.e. individual line VSHIFTs were decoupled from
             their shared kinematic anchor (see
             fastspecfit/data/emline-constraints.yaml). Binning VSHIFT vs.
             observed-frame wavelength (line restwave*(1+z)) across many
             different lines/redshifts is a proxy for per-camera/per-pixel
             wavelength-solution residuals, since fastspecfit does not
             actually fit the three DESI cameras separately.

  --fluxcal  Spectrophotometric-calibration residuals (Sec. 5.5.2): a table of
             synthesized-vs-observed grz photometry (and the raw ingredients
             to compute it) for computing Delta-m = m_synth - m_obs.

Both diagnostics restrict to survey=main (bright and dark separately, written
to separate files), which are each split across 12 nside=1 healpix catalog
files covering the full ~38M-object DR2 sample. To keep peak memory bounded,
each function processes one healpix file at a time -- reading only the
columns it needs via util._read_extensions() -- and accumulates only the
(much smaller) rows that pass the object- and line-level cuts, rather than
loading the full VAC via util.read_fastspec() (which vstacks all 12 files in
memory at once).

Output files are named:
    external/wavecal-{specprod}-{survey}-{program}.fits
    external/fluxcal-{specprod}-{survey}-{program}.fits

Every output row carries TARGETID, SURVEY, PROGRAM, HEALPIX so it can be
rejoined to the full VAC (via util.read_fastspec) for any additional metadata
needed at plotting time; neither diagnostic pre-splits by target class --
DESI_TARGET/BGS_TARGET are carried through uninterpreted in the fluxcal
output so class-dependence can be explored downstream (e.g. via the same
logic as build_figures.target_class_groups()).

Usage (from repo root or code/, on NERSC):
    python code/pipeline-diagnostics.py --wavecal [--survey main] [--verbose]
    python code/pipeline-diagnostics.py --fluxcal [--survey main] [--verbose]

The wavecal figure itself (median VSHIFT vs. observed wavelength) only reads
the already-prepared external/wavecal-*.fits files, so it can run anywhere,
not just on NERSC:
    python code/pipeline-diagnostics.py --wavecal-figure [--include-siii] [--verbose]

"""
import os, sys, argparse
import numpy as np
from astropy.table import Table, vstack

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from util import _catfiles, _read_extensions, _DEFAULT_COLUMNS, good_redshift, DEFAULT_SPECPROD

REPODIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXTDIR  = os.path.join(REPODIR, 'external')

PROGRAMS = ('bright', 'dark')


def _narrow_only_lines(verbose=False):
    """Lines whose VSHIFT is freed from its kinematic-group anchor during the
    narrow_only final pass, per fastspecfit/data/emline-constraints.yaml
    (loaded at runtime, not hardcoded, so this stays in sync with the
    constraint file). Only groups with final_pass.free_vshift: true qualify;
    both 'forbidden' and 'narrow_balmer' currently do (19 + 8 lines).

    Returns
    -------
    dict
        {LINE_NAME_UPPER: restwave [Angstrom, vacuum]}, e.g. {'OIII_5007': 5008.24, ...}
    """
    import yaml
    import fastspecfit

    fsf_data = os.path.join(os.path.dirname(fastspecfit.__file__), 'data')
    with open(os.path.join(fsf_data, 'emline-constraints.yaml')) as f:
        constraints = yaml.safe_load(f)
    linetable = Table.read(os.path.join(fsf_data, 'emlines.ecsv'))
    restwave = {row['name']: float(row['restwave']) for row in linetable}

    lines = {}
    for group in constraints['profiles']['narrow_only']['kinematic_groups']:
        if not group['final_pass']['free_vshift']:
            continue
        for name in [group['anchor']] + list(group.get('members', [])):
            lines[name.upper()] = restwave[name]

    if verbose:
        print(f'  {len(lines)} narrow_only lines with freed VSHIFT: {sorted(lines)}')

    return lines


# ---------------------------------------------------------------------------
# Wavelength-calibration diagnostic (Sec. 5.5.1)
# ---------------------------------------------------------------------------

def prepare_wavecal(survey='main', specprod=DEFAULT_SPECPROD, min_snr=7.0,
                    max_vshift_err=25.0, verbose=False):
    """Build the long-format VSHIFT-vs-observed-wavelength sample.

    Object-level cuts:
      - good_redshift() (per-class DELTACHI2/ZWARN/fiberstatus cuts)
      - no broad Balmer component (HALPHA_BROAD_AMP == 0), so all narrow
        lines share the narrow_only kinematic profile
      - narrow_only final pass adopted, reconstructed exactly from the
        catalog's own adoption criterion (see emlines.py):
        DELTA_KINENDOF > 0 and DELTA_KINECHI2 > DELTA_KINENDOF

    Line-level cuts (per object x line):
      - FLUX_IVAR > 0 and FLUX*sqrt(FLUX_IVAR) > min_snr
      - VSHIFT_IVAR > 0 and 1/sqrt(VSHIFT_IVAR) < max_vshift_err [km/s]

    Output columns: TARGETID, SURVEY, PROGRAM, HEALPIX, LINE, RESTWAVE,
    OBSWAVE [= RESTWAVE*(1+Z)], Z, VSHIFT, VSHIFT_ERR.

    """
    lines = _narrow_only_lines(verbose=verbose)

    extra_cols = ['DELTA_KINECHI2', 'DELTA_KINENDOF', 'HALPHA_BROAD_AMP']
    for line in lines:
        extra_cols += [f'{line}_FLUX', f'{line}_FLUX_IVAR',
                       f'{line}_VSHIFT', f'{line}_VSHIFT_IVAR']
    columns = list(set(extra_cols) | _DEFAULT_COLUMNS)

    for program in PROGRAMS:
        files = _catfiles(specprod, 'fastspec', survey, program)
        outfile = os.path.join(EXTDIR, f'wavecal-{specprod}-{survey}-{program}.fits')

        chunks = []
        n_obj_total = n_obj_pass = n_rows = 0
        for f in files:
            if verbose:
                print(f'Reading {f}')
            cat = _read_extensions(f, ['METADATA', 'FASTSPEC'], columns=columns)
            n_obj_total += len(cat)

            good = good_redshift(cat, survey) & (np.asarray(cat['HALPHA_BROAD_AMP']) == 0)

            dchi2 = np.asarray(cat['DELTA_KINECHI2'], dtype=float)
            dndof = np.asarray(cat['DELTA_KINENDOF'], dtype=float)
            good &= (dndof > 0) & (dchi2 > dndof)

            if not np.any(good):
                continue
            cat = cat[good]
            n_obj_pass += len(cat)

            targetid = np.asarray(cat['TARGETID'])
            healpix  = np.asarray(cat['HEALPIX'])
            z        = np.asarray(cat['Z'], dtype=float)

            for line, restwave in lines.items():
                flux  = np.asarray(cat[f'{line}_FLUX'],       dtype=float)
                fivar = np.asarray(cat[f'{line}_FLUX_IVAR'],  dtype=float)
                vshift = np.asarray(cat[f'{line}_VSHIFT'],       dtype=float)
                vivar  = np.asarray(cat[f'{line}_VSHIFT_IVAR'],  dtype=float)

                with np.errstate(invalid='ignore', divide='ignore'):
                    snr        = flux * np.sqrt(fivar)
                    vshift_err = 1. / np.sqrt(vivar)

                # |VSHIFT| >= 499 km/s means the fit hit the hard optimizer
                # bound (+/-500 km/s); a handful of such fits report
                # pathologically tiny formal errors (down to ~1e-8 km/s), a
                # bounded-optimizer covariance artifact that would otherwise
                # dominate any inverse-variance-weighted statistic computed
                # downstream despite being ~0.02% of rows.
                sel = ((fivar > 0) & (snr > min_snr) &
                       (vivar > 0) & (vshift_err < max_vshift_err) &
                       (np.abs(vshift) < 499.) & np.isfinite(vshift))
                n = int(np.sum(sel))
                if n == 0:
                    continue
                n_rows += n

                chunks.append(Table({
                    'TARGETID':   targetid[sel],
                    'SURVEY':     np.full(n, survey),
                    'PROGRAM':    np.full(n, program),
                    'HEALPIX':    healpix[sel],
                    'LINE':       np.full(n, line),
                    'RESTWAVE':   np.full(n, restwave, dtype='f4'),
                    'OBSWAVE':    (restwave * (1. + z[sel])).astype('f4'),
                    'Z':          z[sel].astype('f4'),
                    'VSHIFT':     vshift[sel].astype('f4'),
                    'VSHIFT_ERR': vshift_err[sel].astype('f4'),
                }))

        if verbose:
            print(f'  {survey}/{program}: {n_obj_total:,} objects -> '
                  f'{n_obj_pass:,} pass object-level cuts -> {n_rows:,} line rows')

        if not chunks:
            print(f'  {survey}/{program}: no rows passed cuts, skipping')
            continue

        out = vstack(chunks)
        out.write(outfile, overwrite=True)
        print(f'  {survey}/{program}: wrote {len(out):,} rows -> {outfile}')

    print('Done.')


# Curated line sets for the wavecal figure. Weak/rare lines dropped for excess
# scatter well beyond their formal errors (NeV 3346/3426, [NII] 5755,
# [OIII] 4363, [SIII] 6312, He I 4471 -- all N < 50,000 with std(VSHIFT) more
# than ~10x the median formal VSHIFT_ERR) or suspiciously tiny formal errors
# ([OII] 7320/7330, median VSHIFT_ERR ~0.4 km/s despite N ~34,000, an order of
# magnitude smaller than any other line -- likely a fit-degeneracy artifact
# from the tied auroral quadruplet, not a real precision gain).
#
# [SIII] 9069/9532 are deliberately excluded from the default figure: both
# show a consistent +11-12 km/s median offset, but fastspecfit's own
# emlines.ecsv carries the comment "Note that the Chianti [SIII] 9069,9532
# wavelengths are quite wrong" -- and at the same observed wavelength (>9000
# Angstrom, populated there only by high-z Halpha) the Balmer curve stays
# flat, so this looks like the known rest-wavelength bug, not a DESI
# wavelength-calibration residual. Pass include_siii=True to show them anyway
# (e.g. to double check this interpretation).
WAVECAL_FORBIDDEN_LINES = ['OII_3726', 'OII_3729', 'NEIII_3869', 'OIII_4959', 'OIII_5007',
                          'NII_6548', 'NII_6584', 'SII_6716', 'SII_6731', 'ARIII_7135',
                          'OI_6300']
WAVECAL_SIII_LINES       = ['SIII_9069', 'SIII_9532']
WAVECAL_BALMER_LINES     = ['HALPHA', 'HBETA', 'HGAMMA', 'HDELTA', 'HEI_5876']

# nominal DESI camera dichroic transition zones [Angstrom], approximate
WAVECAL_CAMERA_TRANSITIONS = [(5660., 5930.), (7470., 7720.)]


def _binned_median_sem(x, y, bins, xrange, min_count=200):
    """Running median of y vs x, with an NMAD-based standard error per bin.

    SEM_median ~ 1.2533 * NMAD(y) / sqrt(N) is the standard large-sample
    approximation for the SEM of a median under (possibly non-Gaussian)
    scatter described by a robust scale estimator; used here instead of an
    inverse-variance-weighted mean because a small fraction of catastrophic
    or underestimated per-object VSHIFT_ERR values otherwise dominate a
    naive weighted statistic (see prepare_wavecal's boundary-pinned-fit cut).

    Returns
    -------
    centers, median, sem, count : ndarray, ndarray, ndarray, ndarray
        Bins with count < min_count have median and sem set to NaN.
    """
    from scipy.stats import binned_statistic
    from util import nmad

    kw = dict(bins=bins, range=xrange)
    cnt, edges, _ = binned_statistic(x, y, statistic='count', **kw)
    med, _, _     = binned_statistic(x, y, statistic='median', **kw)
    mad, _, _     = binned_statistic(x, y, statistic=nmad, **kw)

    centers = 0.5 * (edges[:-1] + edges[1:])
    sem = 1.2533 * mad / np.sqrt(np.maximum(cnt, 1))
    sparse = cnt < min_count
    med[sparse] = np.nan
    sem[sparse] = np.nan

    return centers, med, sem, cnt


def wavecal_residuals(specprod=DEFAULT_SPECPROD, survey='main', nbins=62,
                      xrange=(3550., 9850.), min_count=200, include_siii=False,
                      verbose=False):
    """Sec. 5.5.1 figure: median emission-line VSHIFT vs. observed-frame
    wavelength, as a proxy for residual wavelength-calibration errors.

    Reads external/wavecal-{specprod}-{survey}-{bright,dark}.fits (written by
    prepare_wavecal), drops boundary-pinned fits defensively (harmless if
    prepare_wavecal has already been rerun with that cut applied), and plots
    three running-median curves -- all curated lines combined, forbidden
    lines only, Balmer lines only -- so that a genuine wavelength-calibration
    residual (which should affect any line the same way at a given observed
    wavelength) can be distinguished from a species-dependent astrophysical
    or fit-blending effect (which should not).

    The shaded band is the 25th-75th percentile spread of individual-object
    VSHIFT values in each bin (via util.running_binstat, matching the
    convention used elsewhere in build-figures.py), *not* the uncertainty on
    the plotted median -- with N/bin in the tens of thousands, the standard
    error of the median is of order 0.01-0.1 km/s (printed per curve when
    verbose=True), far too small to plot usefully; the individual-object
    scatter (tens of km/s, dominated by real velocity dispersion and
    per-object measurement noise) is what's actually visible to the eye, and
    is what the band shows.

    Output: tex/figures/wave-residuals.pdf

    """
    import matplotlib.pyplot as plt
    from util import plot_style, running_binstat

    chunks = []
    for program in PROGRAMS:
        infile = os.path.join(EXTDIR, f'wavecal-{specprod}-{survey}-{program}.fits')
        if not os.path.isfile(infile):
            print(f'  Missing {infile}, skipping.')
            continue
        if verbose:
            print(f'Reading {infile}')
        chunks.append(Table.read(infile))
    if not chunks:
        raise FileNotFoundError('No wavecal input files found; run --wavecal first.')
    t = vstack(chunks)

    # Table.read() returns FITS string columns as fixed-width bytes (|S10),
    # not str, so np.isin() against a Python str line list would silently
    # match nothing.
    t['LINE'] = np.asarray(t['LINE']).astype(str)

    t = t[np.abs(np.asarray(t['VSHIFT'], dtype=float)) < 499.]

    forbidden_lines = list(WAVECAL_FORBIDDEN_LINES)
    if include_siii:
        forbidden_lines += WAVECAL_SIII_LINES
    balmer_lines = list(WAVECAL_BALMER_LINES)
    all_lines = forbidden_lines + balmer_lines

    if verbose:
        for label, lines in [('forbidden', forbidden_lines), ('Balmer', balmer_lines)]:
            n = np.isin(t['LINE'], lines).sum()
            print(f'  {label}: {n:,} rows across {len(lines)} lines')

    sns, colors = plot_style(talk=True, font_scale=0.9, palette='colorblind')
    fig, (ax, axn) = plt.subplots(
        2, 1, figsize=(9, 6.5), sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.08})

    for lo, hi in WAVECAL_CAMERA_TRANSITIONS:
        ax.axvspan(lo, hi, color='0.85', zorder=0)
        axn.axvspan(lo, hi, color='0.85', zorder=0)

    # fill zorders (1-3) are all below every median line's zorder (11-13), so
    # the lines stay on top regardless of draw order -- equal-zorder artists
    # otherwise stack in draw order, which buried the first-drawn line under
    # later fill_between() calls.
    curves = [
        ('All lines (combined)', all_lines,       'k',        2.5, 11),
        ('Forbidden lines',      forbidden_lines,  colors[0],  2.2, 12),
        ('Balmer lines',         balmer_lines,     colors[1],  2.2, 13),
    ]
    xc = cnt = None
    ymax = 15.  # widened below if the IQR band needs more room
    for label, lines, color, lw, zorder in curves:
        sel = np.isin(t['LINE'], lines)
        x = np.asarray(t['OBSWAVE'][sel], dtype=float)
        y = np.asarray(t['VSHIFT'][sel],  dtype=float)

        centers, med, qlo, qhi = running_binstat(x, y, bins=nbins, xrange=xrange, min_count=min_count)
        finite = np.isfinite(med)
        ax.fill_between(centers[finite], qlo[finite], qhi[finite],
                        color=color, alpha=0.2, zorder=zorder - 10)
        ax.plot(centers[finite], med[finite], color=color, lw=lw, label=label, zorder=zorder)
        if np.any(finite):
            ymax = max(ymax, 1.05 * np.nanmax(np.abs(np.concatenate([qlo[finite], qhi[finite]]))))

        _, _, sem, count = _binned_median_sem(x, y, nbins, xrange, min_count=min_count)
        if verbose:
            finite_sem = np.isfinite(sem)
            print(f'  {label}: SEM on the median ranges '
                  f'{np.nanmin(sem[finite_sem]):.3f}-{np.nanmax(sem[finite_sem]):.3f} km/s across bins '
                  f'(vs. IQR half-width up to {0.5*np.nanmax(qhi[finite]-qlo[finite]):.1f} km/s)')

        if label.startswith('All'):
            xc, cnt = centers, count

    ax.axhline(0, color='0.4', lw=1, ls='--', zorder=10)
    ax.set_ylim(-ymax, ymax)
    ax.set_ylabel(r'$\Delta v$ (km s$^{-1}$)')
    ax.legend(loc='upper right', fontsize='small', framealpha=0.85)
    ax.set_xlim(xrange)

    axn.plot(xc, cnt, color='k', lw=1.2, drawstyle='steps-mid')
    axn.set_yscale('log')
    axn.set_ylabel(r'$N$/bin')
    axn.set_xlabel(r'Observed-frame Wavelength ($\mathrm{\AA}$)')

    outfile = os.path.join(REPODIR, 'tex', 'figures', 'wave-residuals.pdf')
    fig.savefig(outfile, dpi=150, bbox_inches='tight')
    print(f'Wrote {outfile}')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Spectrophotometric-calibration diagnostic (Sec. 5.5.2)
# ---------------------------------------------------------------------------

_FLUXCAL_BANDS = ('G', 'R', 'Z')


def prepare_fluxcal(survey='main', specprod=DEFAULT_SPECPROD, min_snr=5.0, verbose=False):
    """Build the synthesized-vs-observed photometry sample.

    Delta-m is *not* precomputed: raw fluxes and quality columns are stored
    so Delta-m and any aperture-bias control cuts (compactness via
    FIBERFLUX/FLUX, aperture-correction color-flatness via APERCORR_G/R/Z)
    can be applied and adjusted at plotting time, following the convention
    used elsewhere in prepare-external.py (e.g. prepare_cigaleagn) of storing
    quality flags and applying cuts in the figure code.

    Object-level cuts:
      - good_redshift()
      - exclude the APERCORR fallback sentinel (APERCORR_G == APERCORR_R ==
        APERCORR_Z == 1.0 exactly, set in continuum.py when the
        aperture-correction fit fails)
      - FLUX_IVAR_{G,R,Z} > 0 and FLUX*sqrt(FLUX_IVAR) > min_snr in all three
        bands, so the reference (Legacy Survey) photometry is itself reliable
      - FLUX_SYNTH_{G,R,Z} finite and > 0 in all three bands

    Output columns: TARGETID, SURVEY, PROGRAM, HEALPIX, RA, DEC, Z,
    DESI_TARGET, BGS_TARGET, APERCORR, APERCORR_{G,R,Z}, RCHI2, RCHI2_CONT,
    RCHI2_PHOT, FLUX_SYNTH_{G,R,Z}, FLUX_SYNTH_SPECMODEL_{G,R,Z} (for a
    data-vs-model cross-check), FIBERFLUX_{G,R,Z}, FLUX_{G,R,Z},
    FLUX_IVAR_{G,R,Z}.

    """
    extra_cols = (['APERCORR', 'RCHI2', 'RCHI2_CONT', 'RCHI2_PHOT'] +
                 [f'APERCORR_{b}' for b in _FLUXCAL_BANDS] +
                 [f'FLUX_SYNTH_{b}' for b in _FLUXCAL_BANDS] +
                 [f'FLUX_SYNTH_SPECMODEL_{b}' for b in _FLUXCAL_BANDS] +
                 [f'FIBERFLUX_{b}' for b in _FLUXCAL_BANDS] +
                 [f'FLUX_{b}' for b in _FLUXCAL_BANDS] +
                 [f'FLUX_IVAR_{b}' for b in _FLUXCAL_BANDS])
    columns = list(set(extra_cols) | _DEFAULT_COLUMNS)

    keep_cols = (['TARGETID', 'HEALPIX', 'RA', 'DEC', 'Z',
                 'DESI_TARGET', 'BGS_TARGET'] + extra_cols)

    for program in PROGRAMS:
        files = _catfiles(specprod, 'fastspec', survey, program)
        outfile = os.path.join(EXTDIR, f'fluxcal-{specprod}-{survey}-{program}.fits')

        chunks = []
        n_obj_total = n_obj_pass = 0
        for f in files:
            if verbose:
                print(f'Reading {f}')
            cat = _read_extensions(f, ['METADATA', 'SPECPHOT', 'FASTSPEC'], columns=columns)
            n_obj_total += len(cat)

            good = good_redshift(cat, survey)

            apc = np.array([np.asarray(cat[f'APERCORR_{b}'], dtype=float)
                           for b in _FLUXCAL_BANDS])
            good &= ~np.all(apc == 1.0, axis=0)

            for b in _FLUXCAL_BANDS:
                flux    = np.asarray(cat[f'FLUX_{b}'],       dtype=float)
                fivar   = np.asarray(cat[f'FLUX_IVAR_{b}'],  dtype=float)
                fsynth  = np.asarray(cat[f'FLUX_SYNTH_{b}'], dtype=float)
                with np.errstate(invalid='ignore'):
                    good &= (fivar > 0) & (flux * np.sqrt(fivar) > min_snr)
                good &= np.isfinite(fsynth) & (fsynth > 0)

            if not np.any(good):
                continue
            cat = cat[good]
            n_obj_pass += len(cat)

            sub = cat[keep_cols].copy()
            sub['SURVEY']  = survey
            sub['PROGRAM'] = program
            chunks.append(sub)

        if verbose:
            print(f'  {survey}/{program}: {n_obj_total:,} objects -> '
                  f'{n_obj_pass:,} pass cuts')

        if not chunks:
            print(f'  {survey}/{program}: no rows passed cuts, skipping')
            continue

        out = vstack(chunks)
        out.write(outfile, overwrite=True)
        print(f'  {survey}/{program}: wrote {len(out):,} rows -> {outfile}')

    print('Done.')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument('--wavecal', action='store_true',
                        help='Build the wavelength-calibration diagnostic sample (Sec 5.5.1).')
    parser.add_argument('--wavecal-figure', action='store_true',
                        help='Make the wavecal VSHIFT-vs-wavelength figure (Sec 5.5.1) from existing samples.')
    parser.add_argument('--include-siii', action='store_true',
                        help='wavecal-figure only: include SIII_9069/9532 despite the likely '
                             'emlines.ecsv rest-wavelength bug (see WAVECAL_SIII_LINES docstring).')
    parser.add_argument('--fluxcal', action='store_true',
                        help='Build the spectrophotometric-calibration diagnostic sample (Sec 5.5.2).')
    parser.add_argument('--survey', default='main', choices=['sv3', 'main'],
                        help='DESI survey.')
    parser.add_argument('--specprod', default=DEFAULT_SPECPROD,
                        help='Spectroscopic production name.')
    parser.add_argument('--min-snr', type=float, default=None,
                        help='Override the default per-function line/photometry S/N floor.')
    parser.add_argument('--max-vshift-err', type=float, default=25.0,
                        help='wavecal only: maximum formal VSHIFT uncertainty [km/s].')
    parser.add_argument('--verbose', action='store_true',
                        help='Print progress while reading and cutting.')
    args = parser.parse_args()

    if args.wavecal:
        kwargs = dict(survey=args.survey, specprod=args.specprod,
                      max_vshift_err=args.max_vshift_err, verbose=args.verbose)
        if args.min_snr is not None:
            kwargs['min_snr'] = args.min_snr
        prepare_wavecal(**kwargs)

    if args.wavecal_figure:
        wavecal_residuals(specprod=args.specprod, survey=args.survey,
                          include_siii=args.include_siii, verbose=args.verbose)

    if args.fluxcal:
        kwargs = dict(survey=args.survey, specprod=args.specprod, verbose=args.verbose)
        if args.min_snr is not None:
            kwargs['min_snr'] = args.min_snr
        prepare_fluxcal(**kwargs)


if __name__ == '__main__':
    main()
