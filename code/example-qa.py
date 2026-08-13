#!/usr/bin/env python
"""Illustrative-example QA figure for Section 4 ("Illustrative Examples").

Reads a single DESI target directly from a Redrock file plus its companion
coadd file, runs the full continuum + emission-line fit standalone (via
fastspecfit.fastspecfit.fastspec_one -- no completed fastspec/fastphot
catalog is required), and builds a publication figure: Legacy Survey grz
cutout, broadband SED, full observed spectrum with the best-fit
continuum/smooth-continuum/emission-line model, and a bottom row of zoom
panels on a standard set of line groups chosen from the target's redshift
(see default_groups_for_redshift) -- whichever of those fall in the
observed window are shown. Written to tex/figures/.

Panel content and per-camera colors intentionally match fastspecfit.qa's
production QA figure (fastspecfit.qa.qa_fastspec); several of its private
helpers are reused directly (cutout fetch, SED model, spectral models, line
statistics) rather than reimplemented, exactly as code/patch-emlines-qa.py
reuses fastspecfit.linemasker for the patch-fitting figure. Unlike the
production QA figure, on-figure text is kept to a target-ID/coordinate/
redshift label only; headline fit parameters belong in the caption/prose.

Example (mini specprod built with fastspecfit's build-mini-specprod):

    python code/example-qa.py \
        --redrockfile data/redrock-main-bright-15344.fits \
        --outfile tex/figures/example-bgs.pdf

--targetid is only needed if --redrockfile contains more than one target.

Tractor photometry is looked up under this repo's data/ directory by default
-- i.e. data/{region}/tractor/{brick[:3]}/tractor-{brick}.fits, matching the
{region}/tractor/... layout under build-mini-specprod's
--outdir/external/legacysurvey/dr9 tree (copy just that {region}/tractor/...
subtree into data/, alongside the redrock/coadd files). Falls back to
$FPHOTO_DIR if data/ has no such catalog; override either with --fphotodir.

"""
import os
import sys
import argparse
import numpy as np

REPODIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTFILE = os.path.join(REPODIR, 'tex', 'figures', 'example-qa.pdf')
DEFAULT_FPHOTODIR = os.path.join(REPODIR, 'data')

# Standard zoom-panel groups, left to right: each is (line_names, label),
# where line_names are values from the 'name' column of fastspecfit's
# data/emlines.ecsv (not tied to that file's own 'patch' grouping -- we
# center each panel's window on exactly the requested line(s), e.g. the mean
# wavelength for a doublet, regardless of what else LineMasker happens to
# fit in the same wavelength patch). Any '<name>_broad' counterpart that
# exists in the fit is added automatically (see _expand_with_broad), so e.g.
# 'halpha' alone is enough to pull in a detected broad Halpha component too.
# Which four-panel set is used depends on the target's redshift (see
# default_groups_for_redshift), so a bare invocation always produces four
# physically sensible panels without having to remember and pass --groups by
# hand. Groups with none of their lines in the observed window are skipped.
GROUPS_LOWZ = [    # z < 0.5 (e.g. the BGS example)
    (('oii_3726', 'oii_3729'), '[OII]'),
    (('hbeta', 'oiii_4959', 'oiii_5007'), r'H$\beta$+[OIII]'),
    (('halpha', 'nii_6548', 'nii_6584'), r'H$\alpha$+[NII]'),
    (('sii_6716', 'sii_6731'), '[SII]'),
]
GROUPS_MIDZ = [    # 0.5 <= z < 1.5 (e.g. the ELG example)
    (('oii_3726', 'oii_3729'), '[OII]'),
    (('hgamma',), r'H$\gamma$'),
    (('hbeta',), r'H$\beta$'),
    (('oiii_4959', 'oiii_5007'), '[OIII]'),
]
GROUPS_HIGHZ = [   # z >= 1.5 (e.g. a QSO) -- these UV lines are physically
                   # blended, so each group here matches a full LineMasker
                   # patch (data/emlines.ecsv 'patch' column) rather than a
                   # single line: Lya+NV (patch a), CIV (patch d),
                   # CIII]+SiIII]+AlIII (patch f), MgII doublet (patch g)
    (('lyalpha', 'nv_1240'), r'Ly$\alpha$+NV'),
    (('civ_1549',), 'CIV'),
    (('ciii_1908', 'siliii_1892', 'aliii_1857'), 'CIII]+SiIII]+AlIII'),
    (('mgii_2796', 'mgii_2803'), 'MgII'),
]
DEFAULT_GROUPS = GROUPS_LOWZ   # fallback if make_figure() is called with groups=None directly


def default_groups_for_redshift(redshift):
    """Pick the standard four-panel zoom-panel line set for this target's
    redshift, so the observed-frame line choice tracks which lines actually
    land in the DESI window without the caller having to specify --groups.
    """
    if redshift < 0.5:
        return GROUPS_LOWZ
    if redshift < 1.5:
        return GROUPS_MIDZ
    return GROUPS_HIGHZ


def default_sed_xmin_for_redshift(redshift):
    """Lower bound (observed-frame micron) for the SED panel's x-axis.

    fastspecfit.qa's own SED panel doesn't adapt this to redshift, but a
    fixed 0.1um lower bound leaves increasingly more dead space blueward of
    the Lyman break as targets move to higher redshift, so we widen it in
    step with the same z thresholds used for the zoom-panel line sets (see
    default_groups_for_redshift).
    """
    if redshift < 0.5:
        return 0.1
    if redshift < 1.5:
        return 0.15
    return 0.2


def _expand_with_broad(names, available_names):
    """Add any '<name>_broad' counterpart that exists in the fit, so a group
    named by its narrow line (e.g. 'halpha') still shows a detected broad
    component without listing it explicitly.
    """
    expanded = list(names)
    for name in names:
        broad_name = f'{name}_broad'
        if broad_name in available_names and broad_name not in expanded:
            expanded.append(broad_name)
    return expanded


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--redrockfile', required=True,
                        help='Full path to a Redrock file; the matching coadd-*.fits '
                             'file must be in the same directory.')
    parser.add_argument('--targetid', type=int, default=None,
                        help='TARGETID to process. Only needed if --redrockfile '
                             'contains more than one target.')
    parser.add_argument('--groups', nargs='+', default=None,
                        help="Comma-separated emlines.ecsv line names (the 'name' column) "
                             "to feature as zoom panels, one group each, left to right, "
                             "e.g. 'oii_3726,oii_3729 hgamma hbeta oiii_4959,oiii_5007'. "
                             "Each panel's label is auto-derived from its first line's "
                             "nicename. Default depends on the target's redshift (see "
                             'default_groups_for_redshift): the z<0.5, 0.5<=z<1.5, and '
                             'z>=1.5 sets defined by GROUPS_LOWZ/GROUPS_MIDZ/GROUPS_HIGHZ.')
    parser.add_argument('--outfile', default=DEFAULT_OUTFILE,
                        help='Output path for the figure.')
    parser.add_argument('--fphotodir', default=None,
                        help='Full path to the Legacy Surveys Tractor catalog tree '
                             f'(default: {DEFAULT_FPHOTODIR} if present, else $FPHOTO_DIR).')
    args = parser.parse_args()

    from fastspecfit.singlecopy import sc_data
    from fastspecfit.io import DESISpectra, get_output_dtype
    from fastspecfit.fastspecfit import fastspec_one

    print('Initializing FastSpecFit singleton data structures...')
    sc_data.initialize()

    fphotodir = args.fphotodir
    if fphotodir is None and os.path.isdir(DEFAULT_FPHOTODIR):
        fphotodir = DEFAULT_FPHOTODIR
        print(f'Using local Tractor catalog tree {fphotodir}')

    Spec = DESISpectra(phot=sc_data.photometry, cosmo=sc_data.cosmology, fphotodir=fphotodir)
    targetids = [args.targetid] if args.targetid is not None else None
    Spec.gather_metadata(redrockfiles=[args.redrockfile], targetids=targetids)
    data, meta = Spec.read(sc_data.photometry, fastphot=False)

    if len(data) == 0:
        raise ValueError(f'No targets found in {args.redrockfile}.')
    if len(data) > 1:
        found = [d['uniqueid'] for d in data]
        raise ValueError(f'{len(data)} targets found in {args.redrockfile}; specify '
                         f'--targetid. Candidates: {found}')

    specdata, objmeta = data[0], meta[0]
    print(f'Read TARGETID {specdata["uniqueid"]} at z={specdata["redshift"]:.4f}.')

    ncoeff = sc_data.templates.ntemplates
    fastfit_dtype, _ = get_output_dtype(
        Spec.specprod, phot=sc_data.photometry, linetable=sc_data.emlines.table,
        ncoeff=ncoeff, cameras=specdata['cameras'], fastphot=False, fitstack=False)
    specphot_dtype, _ = get_output_dtype(
        Spec.specprod, phot=sc_data.photometry, linetable=sc_data.emlines.table,
        ncoeff=ncoeff, cameras=specdata['cameras'], fastphot=False, fitstack=False,
        specphot=True)

    objmeta, specphot, fastspec, _ = fastspec_one(
        0, specdata, objmeta, fastfit_dtype, specphot_dtype, broadlinefit=True)

    print(f'  logmstar={specphot["LOGMSTAR"]:.2f}, tauv={specphot["TAUV"]:.2f}, '
          f'vdisp={specphot["VDISP"]:.0f} km/s')

    if args.groups is not None:
        groups = [(tuple(token.split(',')), None) for token in args.groups]
    else:
        groups = default_groups_for_redshift(specdata['redshift'])

    make_figure(specdata, objmeta, specphot, fastspec, Spec.coadd_type, groups, args.outfile)


def make_figure(specdata, objmeta, specphot, fastspec, coadd_type, groups, outfile):
    """Cutout + SED + full spectrum on top, with a bottom row of zoom panels
    on standard emission-line groups showing the data, the best-fit
    continuum, continuum+smooth-continuum, and the full model
    (continuum+smooth-continuum+emission lines) -- mirroring
    fastspecfit.qa.qa_fastspec's panel content and per-camera colors, but
    restyled for the paper and with on-figure text limited to a target
    label (headline numbers belong in the caption).
    """
    sys.path.insert(0, os.path.join(REPODIR, 'code'))
    from util import plot_style
    from matplotlib.patches import Circle, ConnectionPatch
    from matplotlib.lines import Line2D

    from scipy.ndimage import median_filter
    from fastspecfit.util import C_LIGHT, FLUXNORM, ivar2var
    from fastspecfit.continuum import ContinuumTools
    from fastspecfit.emlines import EMFitTools
    from fastspecfit.emline_fit import EMLine_MultiLines
    from fastspecfit.qa import (_target_label, _compute_line_stats, _build_sed_model,
                                _build_spectral_models, _fetch_cutout)
    from fastspecfit.singlecopy import sc_data

    if groups is None:
        groups = DEFAULT_GROUPS

    # same context/style/font_scale as fastspecfit.qa.qa_fastspec, so this
    # figure reads at the same weight and size as the production QA plots
    plot_style(talk=True, font_scale=1.3, palette='colorblind', style='ticks')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    @ticker.FuncFormatter
    def major_formatter(x, pos):
        if 0.01 <= x < 0.1:
            return f'{x:.2f}'
        elif 0.1 <= x < 1:
            return f'{x:.1f}'
        else:
            return f'{x:.0f}'

    # per-camera (b, r, z) data/model colors, matching fastspecfit.qa.qa_fastspec
    # and code/patch-emlines-qa.py
    CAMERA_COLORS = ['#468fcc', '#4caf81', '#e07a75']       # data
    CAMERA_COLORS_DARK = ['#003f91', '#007f5f', '#9b2226']  # model
    bbox = dict(boxstyle='round', facecolor='lightgray', alpha=0.3, edgecolor='none')

    phot = sc_data.photometry
    templates = sc_data.templates
    redshift = specdata['redshift']

    CTools = ContinuumTools(specdata, templates, phot, sc_data.igm, fastphot=False,
                            fluxnorm=FLUXNORM)
    EMFit = EMFitTools(emline_table=sc_data.emlines.table, constraints=sc_data.constraints)

    linetable, _ = _compute_line_stats(EMFit, fastspec, specdata, redshift)

    specmodels = _build_spectral_models(
        CTools, EMFit, specdata, fastspec, specphot, templates, fitstack=False,
        no_smooth_continuum=False, emline_snrmin=0.0, redshift=redshift)

    phot_wavelims = (default_sed_xmin_for_redshift(redshift), 35.)
    allfilters = phot.filters[objmeta['PHOTSYS']]
    sedwave, sedmodel, sedphot, phot_tbl = _build_sed_model(
        CTools, templates, specphot, objmeta, phot, redshift, phot_wavelims, allfilters)

    outdir = os.path.dirname(outfile)
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    layer = getattr(phot, 'viewer_layer', f'ls-{getattr(phot, "legacysurveydr", "dr9")}')
    pixscale = getattr(phot, 'viewer_pixscale', 0.262)
    img, _, _, _ = _fetch_cutout(objmeta, outdir, outfile, layer, pixscale)

    target = _target_label(objmeta, coadd_type)

    fullwave = specmodels['fullwave']
    apercorr = specmodels['apercorr']

    # which of the standard line groups actually fall in this target's
    # observed window -- decided up front so the figure width/layout below
    # is sized to what will actually be drawn. Each group's line list is
    # expanded with any detected broad counterpart, and a missing label is
    # auto-derived from the first present line's nicename (used for
    # user-supplied --groups, which don't carry a hand-written label).
    all_names = set(linetable['name'])
    active_groups = []
    for names, label in groups:
        present = [n for n in _expand_with_broad(names, all_names) if n in all_names]
        if not present:
            continue
        if label is None:
            firstrow = linetable[linetable['name'] == present[0]]
            label = firstrow['nicename'][0].replace('-', ' ')
        active_groups.append((present, label))
    ngroup = len(active_groups)

    # ------------------------------------------------------------------
    # figure layout: explicit axes rects, all three rows using the same
    # left/right extent -- the cutout is now a small inset in the lower-
    # right corner of the SED panel (no RA/Dec axes of its own), so there's
    # no longer a narrower "top row" competing for width with the spectrum
    # and zoom-panel rows below it.
    # ------------------------------------------------------------------
    L, R = 0.09, 0.97
    fig_w_in, fig_h_in = 3.6 * max(ngroup, 1), 12.5

    sed_y1, sed_h = 0.9, 0.33          # leaves room above for the rest-frame twin axis
    sed_y0 = sed_y1 - sed_h

    spec_gap, spec_h = 0.05, 0.20
    spec_y1 = sed_y0 - spec_gap
    spec_y0 = spec_y1 - spec_h

    zoom_gap, zoom_h = 0.05, 0.15
    zoom_y1 = spec_y0 - zoom_gap
    zoom_y0 = zoom_y1 - zoom_h

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))

    sedax = fig.add_axes([L, sed_y0, R - L, sed_h])
    specax = fig.add_axes([L, spec_y0, R - L, spec_h])

    # ---- SED panel ----
    factor = 10**(0.4 * 48.6) * sedwave**2 / (C_LIGHT * 1e13) / FLUXNORM / CTools.massnorm
    sedmodel_abmag = -2.5 * np.log10(sedmodel * factor)
    sedax.plot(sedwave / 1e4, sedmodel_abmag, color='0.5', alpha=0.9, zorder=1)
    sedax.scatter(sedphot['lambda_eff'] / 1e4, sedphot['abmag'], marker='D', s=300,
                  color='k', facecolor='none', linewidth=1.5, zorder=3)

    abmag_good = phot_tbl['abmag_ivar'] > 0
    abmag_lim = phot_tbl['abmag_limit'] > 0

    # y-range with headroom, faint (large mag) at bottom -- mirrors
    # qa_fastspec's sed_ymin/sed_ymax construction, including its safety
    # clamp so a handful of near-zero-flux model pixels near the wavelength
    # edges (formally ABmag -> a very large number) can't blow up the axis
    dm = 1.5
    mags = [np.atleast_1d(sedmodel_abmag)]
    if np.any(abmag_good):
        mags.append(np.atleast_1d(phot_tbl['abmag'][abmag_good]))
    if np.any(abmag_lim):
        mags.append(np.atleast_1d(phot_tbl['abmag_limit'][abmag_lim]))
    mags = np.concatenate(mags)
    mags = mags[np.isfinite(mags)]
    sed_ymin = np.nanmax(mags) + dm
    sed_ymax = np.nanmin(mags) - dm
    if sed_ymin > 30:
        sed_ymin = 30.

    sedax.set_xscale('log')
    sedax.set_xlim(*phot_wavelims)
    # set_ylim (sed_ymin > sed_ymax, so this flips the axis to put faint/large
    # mags at the bottom) *before* the lolims=True errorbar call below --
    # matplotlib decides which way the upper-limit caret points by checking
    # yaxis_inverted() at call time, so plotting the limits before the axis
    # is actually inverted draws them pointing the wrong way.
    sedax.set_ylim(sed_ymin, sed_ymax)
    sedax.set_ylabel('AB mag')

    if np.any(abmag_good):
        yerr = np.squeeze([phot_tbl['abmag_brighterr'], phot_tbl['abmag_fainterr']])
        sedax.errorbar(phot_tbl['lambda_eff'][abmag_good]/1e4, phot_tbl['abmag'][abmag_good],
                       yerr=yerr[:, abmag_good], fmt='o', markersize=14, markeredgewidth=1,
                       markeredgecolor='k', markerfacecolor='darkorange', elinewidth=1.5,
                       ecolor='darkorange', capsize=3, zorder=4)
    if np.any(abmag_lim):
        # markersize bumped up relative to the detections above -- the
        # lolims caret glyph reads visually smaller than a filled circle at
        # the same nominal markersize
        sedax.errorbar(phot_tbl['lambda_eff'][abmag_lim]/1e4, phot_tbl['abmag_limit'][abmag_lim],
                       lolims=True, yerr=0.75, fmt='o', markersize=20, markeredgewidth=1.5,
                       markeredgecolor='k', markerfacecolor='none', elinewidth=1.5,
                       ecolor='darkorange', capsize=3, alpha=0.7, zorder=4)
    sedax.xaxis.set_major_formatter(major_formatter)
    obsticks = np.array([0.1, 0.2, 0.5, 1.0, 1.5, 3.0, 5.0, 10.0, 20.0])
    obsticks = obsticks[(obsticks >= phot_wavelims[0]) & (obsticks <= phot_wavelims[1])]
    sedax.set_xticks(obsticks)
    # observed-frame x-axis label is shared with the spectrum panel directly
    # below (tied together visually by the connector lines), so it isn't
    # repeated here -- matches qa_fastspec, which only labels sedax's
    # x-axis in fastphot-only mode

    # rest-frame wavelength as a twin top axis, as in qa_fastspec
    sedax_twin = sedax.twiny()
    sedax_twin.set_xscale('log')
    sedax_twin.set_xlim(phot_wavelims[0]/(1+redshift), phot_wavelims[1]/(1+redshift))
    sedax_twin.xaxis.set_major_formatter(major_formatter)
    restticks = np.array([0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 3.0, 5.0, 10.0, 15.0, 20.0])
    restticks = restticks[(restticks >= phot_wavelims[0]/(1+redshift)) &
                          (restticks <= phot_wavelims[1]/(1+redshift))]
    sedax_twin.set_xticks(restticks)
    sedax_twin.set_xlabel(r'Rest-frame Wavelength ($\mu$m)')

    # gray bar marking the DESI spectral range plus the aperture-correction
    # factor applied to place the fiber spectrum on the photometric scale
    specwave_min, specwave_max = np.min(fullwave)/1e4, np.max(fullwave)/1e4
    sedax.plot([specwave_min, specwave_max], [sed_ymin - 1, sed_ymin - 1], lw=2, ls='-',
              color='gray', marker='s', markersize=4, zorder=2)
    sedax.text((specwave_max - specwave_min)/2 + specwave_min*0.8, sed_ymin - 1.7,
              f'DESI x {apercorr:.2f}', ha='center', va='center', fontsize=16, color='k')

    sedax.text(0.03, 0.96, '\n'.join(target), transform=sedax.transAxes, ha='left',
              va='top', fontsize=14, linespacing=1.4, bbox=bbox, zorder=6)

    # ---- cutout inset, lower-right corner of the SED panel -- a plain
    # (non-WCS) image, no RA/Dec axes, so it can't blow up the layout the
    # way the WCSAxes version did
    cut_pad = 0.012
    cut_w = 0.2
    cut_h = cut_w * fig_w_in / (1.3 * fig_h_in)   # 1.3 = cutout image's own width/height ratio
    cutax = fig.add_axes([R - cut_pad - cut_w, sed_y0 + cut_pad, cut_w, cut_h])
    cutax.imshow(img, origin='lower', aspect='auto')
    cutax.set_xticks([])
    cutax.set_yticks([])
    for spine in cutax.spines.values():
        spine.set_edgecolor('k')
        spine.set_linewidth(1.2)
    sz = img.shape
    cutax.add_artist(Circle((sz[1]/2, sz[0]/2), radius=1.5/2/pixscale,
                            facecolor='none', edgecolor='yellow', ls='-', alpha=0.9))
    cutax.add_artist(Circle((sz[1]/2, sz[0]/2), radius=10/2/pixscale,
                            facecolor='none', edgecolor='yellow', ls='--', alpha=0.9))
    handles = [Line2D([0], [0], color='yellow', lw=1.5, ls='-', label='1.5"'),
              Line2D([0], [0], color='yellow', lw=1.5, ls='--', label='10"')]
    cutax.legend(handles=handles, loc='lower left', fontsize=8, facecolor='lightgray',
                framealpha=0.6, handlelength=1.4, borderpad=0.3, labelspacing=0.2)

    # ---- full spectrum panel ----
    specax.plot(fullwave/1e4, specmodels['fullcontinuum'], color='k', lw=0.8, alpha=0.7, zorder=2)
    specax.plot(fullwave/1e4, specmodels['fullcontinuum'] + specmodels['fullsmoothcontinuum'],
               color='0.4', lw=0.8, ls='--', alpha=0.8, zorder=2)

    # robust range, following qa_fastspec's own construction: a robust sigma
    # from the interquartile spread of the data-minus-model residuals, plus
    # a positive floor set by the smoothed data and model themselves -- this
    # is what keeps the panel from being dragged down by noisy negative
    # pixels or blown up by a single spike
    spec_ymin, spec_ymax = 1e6, -1e6
    for icam in range(len(specdata['cameras'])):
        wave = specdata['wave'][icam]
        flux = specdata['flux'][icam]
        cont = specmodels['desicontinuum'][icam]
        emlines = specmodels['desiemlines'][icam]
        model = cont + specmodels['desismoothcontinuum'][icam] + emlines

        sigma, good = ivar2var(specdata['ivar'][icam], sigma=True, allmasked_ok=True, clip=0)
        wave_g, flux_g, model_g = wave[good], flux[good], model[good]

        specax.plot(wave_g/1e4, flux_g, color=CAMERA_COLORS[icam], lw=0.7, alpha=0.75,
                   drawstyle='steps-mid', zorder=3)
        specax.plot(wave_g/1e4, model_g, color=CAMERA_COLORS_DARK[icam], lw=1.3, alpha=0.95, zorder=4)

        if len(flux_g) > 0:
            filtflux = median_filter(flux_g, 51, mode='nearest')
            sigflux = np.diff(np.percentile(flux_g - model_g, [25, 75]))[0] / 1.349
            spec_ymin = min(spec_ymin, -2 * sigflux)
            spec_ymax = max(spec_ymax, 6 * sigflux, 1.25 * np.nanmax(filtflux),
                            1.25 * np.nanmax(model_g))

        # aperture-corrected DESI model (continuum + emission lines, no
        # smooth-continuum term) overplotted on the SED panel in AB mag, as
        # in qa_fastspec -- gives a preview of the spectral shape sitting
        # within the broadband SED
        desimodelspec = apercorr * (cont + emlines)
        sedgood = desimodelspec > 0
        if np.any(sedgood):
            sed_factor = 10**(0.4 * 48.6) * wave[sedgood]**2 / (C_LIGHT * 1e13) / FLUXNORM
            sedax.plot(wave[sedgood]/1e4, -2.5 * np.log10(desimodelspec[sedgood] * sed_factor),
                      color=CAMERA_COLORS_DARK[icam], lw=1.1, alpha=0.85, zorder=3)

    specax.set_xlim(np.min(fullwave)/1e4, np.max(fullwave)/1e4)
    specax.set_ylim(spec_ymin, spec_ymax)
    # x-axis label is shared across the SED/spectrum/zoom-panel rows (see
    # the single fig.text call at the bottom of the figure)

    # light-gray labels for every in-range emission-line group along the top
    # of the spectrum panel, as in qa_fastspec
    specwave_lo, specwave_hi = np.min(fullwave), np.max(fullwave)
    for pgroup in sorted(set(linetable['plotgroup'])):
        pgrows = linetable[linetable['plotgroup'] == pgroup]
        meanwave_obs = np.mean(pgrows['restwave'].value) * (1 + redshift)
        if not (specwave_lo < meanwave_obs < specwave_hi):
            continue
        linename = pgrows['nicename'][0].replace('-', ' ')
        specax.text(meanwave_obs/1e4, spec_ymax*0.97, linename, ha='center', va='top',
                   rotation=270, fontsize=10, alpha=0.5, zorder=6)

    # connector lines tying the DESI wavelength range in the SED panel to
    # the full-spectrum panel below it, as in qa_fastspec
    for xw in specax.get_xlim():
        sedax.add_artist(ConnectionPatch(
            xyA=(xw, sed_ymin), xyB=(xw, spec_ymax), coordsA='data', coordsB='data',
            axesA=sedax, axesB=specax, color='k', lw=0.8))

    # ---- zoom panels on standard line groups (continuum-subtracted) ----
    # explicit rects (not GridSpec columns), evenly spaced across [L, R]
    zoom_gap_frac = 0.05
    zoom_w = (R - L - (ngroup - 1) * zoom_gap_frac) / max(ngroup, 1)
    zoom_x0s = [L + i * (zoom_w + zoom_gap_frac) for i in range(ngroup)]

    # instantiate the individual line profiles for the converged model, used
    # only to overplot each line's own Gaussian (not just the combined
    # curve) -- mirrors fastspecfit.qa.qa_fastspec's zoom-panel code
    parameters = np.array([fastspec[param] for param in EMFit.param_table['modelname']])
    parameters[EMFit.doublet_idx] *= parameters[EMFit.doublet_src]
    lineprofiles = EMLine_MultiLines(parameters, fullwave, redshift,
                                     EMFit.line_table['restwave'].value,
                                     specdata['res'], specdata['camerapix'])

    zoomaxes = []
    for igrp, (names, label) in enumerate(active_groups):
        rows = linetable[np.isin(linetable['name'], names)]
        minwave = np.min(rows['restwave'].value)
        maxwave = np.max(rows['restwave'].value)
        deltawave = 0.5 * (maxwave - minwave)

        sigmas1 = np.array([fastspec[f'{name.upper()}_SIGMA'] for name in rows['name']])
        sigmas1 = sigmas1[sigmas1 > 0]
        plotsig = max(50., 1.5 * np.mean(sigmas1)) if len(sigmas1) > 0 else 200.

        wmin = (minwave - deltawave) * (1+redshift) - 5 * plotsig * minwave * (1+redshift) / C_LIGHT
        wmax = (maxwave + deltawave) * (1+redshift) + 5 * plotsig * maxwave * (1+redshift) / C_LIGHT

        ax = fig.add_axes([zoom_x0s[igrp], zoom_y0, zoom_w, zoom_h])

        line_ymin, line_ymax = 0., 0.
        for icam in range(len(specdata['cameras'])):
            wave = specdata['wave'][icam]
            resid = (specdata['flux'][icam] - specmodels['desicontinuum'][icam] -
                    specmodels['desismoothcontinuum'][icam])
            model = specmodels['desiemlines'][icam]

            sigma, good = ivar2var(specdata['ivar'][icam], sigma=True, allmasked_ok=True, clip=0)
            wave, resid, model = wave[good], resid[good], model[good]

            indx = np.where((wave > wmin) & (wave < wmax))[0]
            if len(indx) < 2:
                continue

            ax.plot(wave[indx]/1e4, resid[indx], color=CAMERA_COLORS[icam], lw=0.9,
                   alpha=0.75, drawstyle='steps-mid', zorder=3)

            # individual line profiles for this group, so blended narrow
            # components (e.g. narrow Halpha under a broad wing) stay visible
            for name in rows['name']:
                (s, e), oneline = lineprofiles.getLine(EMFit.line_map[name])
                if s == e:
                    continue
                plotline = np.zeros_like(fullwave)
                plotline[s:e] = oneline
                srt = np.argsort(fullwave)
                ax.plot(fullwave[srt]/1e4, plotline[srt], lw=0.9, alpha=0.7,
                       color=CAMERA_COLORS_DARK[icam], zorder=4)

            ax.plot(wave[indx]/1e4, model[indx], color=CAMERA_COLORS_DARK[icam], lw=1.6,
                   alpha=0.95, zorder=5)

            sigflux = np.std(resid[indx])
            filtflux = median_filter(resid[indx], 3, mode='nearest')
            _line_ymax = max(4 * sigflux, 1.4 * np.nanmax(model[indx]), np.nanmax(filtflux))
            _line_ymin = -1.5 * sigflux
            if np.nanmin(model[indx]) < _line_ymin:
                _line_ymin = 0.8 * np.nanmin(model[indx])
            line_ymax = max(line_ymax, _line_ymax)
            line_ymin = min(line_ymin, _line_ymin)

        ax.set_xlim(wmin/1e4, wmax/1e4)
        if line_ymax > line_ymin:
            ax.set_ylim(line_ymin, line_ymax)
        ax.text(0.05, 0.94, label, transform=ax.transAxes, ha='left', va='top',
               fontsize=14, fontweight='bold')
        ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
        ax.tick_params(axis='x', labelrotation=30)
        zoomaxes.append(ax)

    for ax in fig.axes:
        ax.grid(False)

    # single x-axis label shared by the SED, spectrum, and zoom-panel rows
    # (all observed-frame wavelength in microns), placed under the bottom-
    # most row actually present
    bottomrow = zoomaxes if zoomaxes else [specax]
    x0 = bottomrow[0].get_position().x0
    x1 = bottomrow[-1].get_position().x1
    y0 = min(ax.get_position().y0 for ax in bottomrow)
    fig.text((x0 + x1)/2, y0 - 0.06, r'Observed-frame Wavelength ($\mu$m)',
             ha='center', va='top', fontsize=plt.rcParams['axes.labelsize'])

    # shared F_lambda y-axis label spanning the spectrum panel and the
    # zoom-panel row below it (both on the same flux scale), positioned
    # just left of their tick labels
    specpos = specax.get_position()
    if zoomaxes:
        zoompos = zoomaxes[0].get_position()
        ymid = (specpos.y1 + zoompos.y0) / 2
    else:
        ymid = (specpos.y0 + specpos.y1) / 2
    fig.text(L - 0.06, ymid, r'$F_{\lambda}\ (10^{-17}~{\rm erg}~{\rm s}^{-1}~{\rm cm}^{-2}~\AA^{-1})$',
             ha='center', va='center', rotation=90, fontsize=plt.rcParams['axes.labelsize'])

    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f'Wrote {outfile}')


if __name__ == '__main__':
    main()
