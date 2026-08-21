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

                sel = ((fivar > 0) & (snr > min_snr) &
                       (vivar > 0) & (vshift_err < max_vshift_err) &
                       np.isfinite(vshift))
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

    if args.fluxcal:
        kwargs = dict(survey=args.survey, specprod=args.specprod, verbose=args.verbose)
        if args.min_snr is not None:
            kwargs['min_snr'] = args.min_snr
        prepare_fluxcal(**kwargs)


if __name__ == '__main__':
    main()
