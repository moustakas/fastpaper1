#!/usr/bin/env python
"""Draft QA figure for the velocity-dispersion section.

Reads a single DESI target's raw spectrum (Redrock + coadd), a *precomputed*
fastspecfit output file for that target (METADATA/SPECPHOT/FASTSPEC/MODELS,
produced by a normal `fastspec` run -- read with
fastspecfit.io.read_fastspecfit, no re-fit needed), and a companion chi2-grid
ECSV (VDISP, TAUV, CHI2 columns, written by
fastspecfit.continuum.vdisp_by_chi2scan's debug_plots=True path). Builds a
two-panel figure:

  left  -- observed spectrum + best-fit continuum/smooth-continuum model
           (reusing code/example-qa.py's full-spectrum panel machinery
           almost verbatim), with the rest-frame 3800-6000A window that
           fastspecfit.continuum.can_compute_vdisp actually uses to fit
           sigma_star (Ca H&K, G band, Hbeta, Mgb, Fe complex) shaded, plus
           a Legacy Survey grz cutout inset (as in example-qa.py's SED
           panel, just relocated since there's no SED panel here).
  right -- new: P(sigma_star) vs sigma_star, i.e. the chi2 grid converted to
           a normalized profile-likelihood curve, with the production
           VDISP/VDISP_IVAR (from SPECPHOT -- a continuous refit seeded by
           this grid, with its uncertainty from Monte Carlo repeats, *not*
           from this grid's curvature) annotated as a vertical line + band.
           The grid's own chi2-minimum can also be marked for comparison
           (see the commented-out lines in make_figure).

This is a first-cut workshop figure -- run it and see how it looks.

All three input files default to the single example this figure is being
workshopped on (see DEFAULT_* below), so a bare invocation just works:

    python code/vdisp-qa.py

Override --redrockfile/--fastspecfile/--chi2scanfile/--targetid together to
try a different target. --targetid is only needed if --redrockfile contains
more than one target.

The Legacy Survey cutout is fetched once per TARGETID and permanently cached
as data/cutout-<targetid>.jpeg (see --cutoutdir, and util.fetch_cutout_cached
-- fastspecfit.qa._fetch_cutout deletes its own temp JPEG after every call,
so we don't use it directly) -- re-running the script (even with a different
--outfile) reuses the cached image instead of hitting the network again.

"""
import os
import sys
import argparse
import numpy as np

REPODIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATADIR = os.path.join(REPODIR, 'data')
DEFAULT_OUTFILE = os.path.join(REPODIR, 'tex', 'figures', 'vdisp-qa.pdf')
DEFAULT_FPHOTODIR = DATADIR

# single example this figure is being workshopped on; override on the
# command line to try a different target
DEFAULT_TARGETID = 39627496647296130
DEFAULT_REDROCKFILE = os.path.join(DATADIR, 'redrock-main-dark-17289.fits')
DEFAULT_FASTSPECFILE = os.path.join(DATADIR, f'f-main-dark-17289-{DEFAULT_TARGETID}.fits')
DEFAULT_CHI2SCANFILE = os.path.join(DATADIR, f'qa-vdisp-chi2scan-{DEFAULT_TARGETID}.ecsv')

# rest-frame window fastspecfit.continuum.can_compute_vdisp uses to fit
# sigma_star (Ca H&K, G band, Hbeta, Mgb, Fe complex); kept in sync by hand
# since it's a plotting-only constant, not imported from fastspecfit.
VDISP_FIT_RESTRANGE = (3800., 6000.)

# reject pixels this many sigma from the model when *displaying* the
# spectrum -- a handful of bad-sky/cosmic-ray pixels are visually distracting
# in an illustrative figure; doesn't touch the actual fit
DISPLAY_CLIP_SIGMA = 6.0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--redrockfile', default=DEFAULT_REDROCKFILE,
                        help='Full path to a Redrock file; the matching coadd-*.fits '
                             'file must be in the same directory.')
    parser.add_argument('--fastspecfile', default=DEFAULT_FASTSPECFILE,
                        help='Full path to a precomputed fastspecfit output FITS file '
                             '(METADATA/SPECPHOT/FASTSPEC/MODELS) for the same target.')
    parser.add_argument('--chi2scanfile', default=DEFAULT_CHI2SCANFILE,
                        help='Full path to the companion qa-vdisp-chi2scan-*.ecsv file '
                             '(VDISP, TAUV, CHI2 columns).')
    parser.add_argument('--targetid', type=int, default=None,
                        help='TARGETID to process. Only needed if --redrockfile '
                             'contains more than one target.')
    parser.add_argument('--outfile', default=DEFAULT_OUTFILE,
                        help='Output path for the figure.')
    parser.add_argument('--cutoutdir', default=DATADIR,
                        help='Directory for the cached Legacy Survey cutout JPEG '
                             '(named from --targetid/TARGETID, independent of '
                             '--outfile, so it survives across --outfile choices '
                             'and is only ever fetched from the network once).')
    parser.add_argument('--fphotodir', default=None,
                        help='Full path to the Legacy Surveys Tractor catalog tree '
                             f'(default: {DEFAULT_FPHOTODIR} if present, else $FPHOTO_DIR). '
                             'Needed even though this figure has no SED/photometry panel -- '
                             'DESISpectra.gather_metadata always cross-matches Tractor '
                             'photometry as part of reading the metadata.')
    parser.add_argument('--pdf', action='store_true',
                        help='Show the right-hand panel as the normalized likelihood '
                             'P(sigma_star) (proportional to exp(-chi2/2)). Default is '
                             'to show delta-chi2 vs sigma_star directly -- the exponential '
                             'in the normalized likelihood suppresses essentially all '
                             'information away from the minimum, which chi2 preserves.')
    args = parser.parse_args()

    from astropy.table import Table
    from fastspecfit.singlecopy import sc_data
    from fastspecfit.io import DESISpectra, read_fastspecfit

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

    # Spec.read() only returns the raw per-camera wave0/flux0/ivar0/mask0/res0
    # arrays; one_spectrum() applies MW dust correction, pixel masking, and
    # the uncertainty floor in place to populate wave/flux/ivar/mask/res
    # (and linemask/linepix), same as fastspecfit.qa.desiqa_one does when
    # building a QA figure from an already-completed fit rather than a fresh
    # fastspec_one() call.
    from fastspecfit.io import one_spectrum
    one_spectrum(specdata, objmeta, fastphot=False)

    print(f'Read TARGETID {specdata["uniqueid"]} at z={specdata["redshift"]:.4f}.')

    filemeta, specphot, fastspec, coadd_type, fastphot = read_fastspecfit(args.fastspecfile)
    if fastphot:
        raise ValueError(f'{args.fastspecfile} is a fastphot-only file; need a fastspec file.')
    if len(specphot) != 1 or filemeta['TARGETID'][0] != specdata['uniqueid']:
        raise ValueError(f'{args.fastspecfile} does not contain a single row matching '
                         f'TARGETID {specdata["uniqueid"]}.')
    specphot, fastspec = specphot[0], fastspec[0]

    print(f'  logmstar={specphot["LOGMSTAR"]:.2f}, tauv={specphot["TAUV"]:.2f}, '
          f'vdisp={specphot["VDISP"]:.0f} km/s')

    chi2grid = Table.read(args.chi2scanfile)

    make_figure(specdata, objmeta, specphot, fastspec, chi2grid, args.outfile,
               args.cutoutdir, args.pdf)


def make_figure(specdata, objmeta, specphot, fastspec, chi2grid, outfile, cutoutdir, show_pdf):
    """Left: spectrum + continuum fit (shaded vdisp-fit window) + cutout inset.
    Right: delta-chi2 (default) or normalized P(sigma_star) (show_pdf=True)
    vs sigma_star from the chi2 grid, with the production VDISP annotated.
    """
    sys.path.insert(0, os.path.join(REPODIR, 'code'))
    from util import plot_style, target_class, fetch_cutout_cached
    from matplotlib.patches import Circle
    from matplotlib.lines import Line2D

    from scipy.ndimage import median_filter
    from fastspecfit.util import FLUXNORM, ivar2var
    from fastspecfit.continuum import ContinuumTools
    from fastspecfit.emlines import EMFitTools
    from fastspecfit.qa import _build_spectral_models
    from fastspecfit.singlecopy import sc_data

    plot_style(talk=True, font_scale=1.15, palette='colorblind', style='ticks')
    import matplotlib.pyplot as plt

    # per-camera (b, r, z) data/model colors, matching fastspecfit.qa.qa_fastspec
    # and code/example-qa.py
    CAMERA_COLORS = ['#468fcc', '#4caf81', '#e07a75']       # data
    CAMERA_COLORS_DARK = ['#003f91', '#007f5f', '#9b2226']  # model

    phot = sc_data.photometry
    templates = sc_data.templates
    redshift = specdata['redshift']

    CTools = ContinuumTools(specdata, templates, phot, sc_data.igm, fastphot=False,
                            fluxnorm=FLUXNORM)
    EMFit = EMFitTools(emline_table=sc_data.emlines.table, constraints=sc_data.constraints)

    specmodels = _build_spectral_models(
        CTools, EMFit, specdata, fastspec, specphot, templates, fitstack=False,
        no_smooth_continuum=False, emline_snrmin=0.0, redshift=redshift)

    outdir = os.path.dirname(outfile)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    os.makedirs(cutoutdir, exist_ok=True)

    # cache name keyed on TARGETID (not --outfile) so the cutout is only ever
    # fetched from the network once per target, regardless of --outfile, and
    # is kept on disk (unlike fastspecfit.qa._fetch_cutout, which deletes it)
    layer = getattr(phot, 'viewer_layer', f'ls-{getattr(phot, "legacysurveydr", "dr9")}')
    pixscale = getattr(phot, 'viewer_pixscale', 0.262)
    img = fetch_cutout_cached(objmeta, cutoutdir, layer, pixscale)

    tclass = target_class(objmeta, survey=objmeta['SURVEY'])
    title = f'{tclass} TARGETID {objmeta["TARGETID"]} ' + r'($z=' + f'{redshift:.4f}' + r'$)'

    fullwave = specmodels['fullwave']

    # ------------------------------------------------------------------
    # figure layout: two panels side by side
    # ------------------------------------------------------------------
    fig_w_in, fig_h_in = 13., 5.5
    L, R = 0.10, 0.98
    B, T = 0.16, 0.90
    gap = 0.09
    spec_w = 0.56
    like_w = R - L - gap - spec_w

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    specax = fig.add_axes([L, B, spec_w, T - B])
    likeax = fig.add_axes([L + spec_w + gap, B, like_w, T - B])

    # ---- left: full spectrum panel ----
    specax.plot(fullwave/1e4, specmodels['fullcontinuum'], color='k', lw=0.8, alpha=0.7, zorder=2)
    specax.plot(fullwave/1e4, specmodels['fullcontinuum'] + specmodels['fullsmoothcontinuum'],
               color='0.4', lw=0.8, ls='--', alpha=0.8, zorder=2)

    spec_ymin, spec_ymax = 1e6, -1e6
    for icam in range(len(specdata['cameras'])):
        wave = specdata['wave'][icam]
        flux = specdata['flux'][icam]
        cont = specmodels['desicontinuum'][icam]
        emlines = specmodels['desiemlines'][icam]
        model = cont + specmodels['desismoothcontinuum'][icam] + emlines

        sigma, good = ivar2var(specdata['ivar'][icam], sigma=True, allmasked_ok=True, clip=0)
        wave_g, flux_g, model_g = wave[good], flux[good], model[good]

        # reject a handful of badly noisy pixels for display purposes only
        # (illustrative figure -- doesn't touch the fit or the catalog). Use
        # a robust camera-wide sigma from the IQR of (data-model) residuals,
        # *not* the per-pixel formal sigma from ivar -- at e.g. the noisy red
        # end of a camera the formal sigma is itself inflated, so a per-pixel
        # cut there never triggers even for genuinely bad pixels.
        if len(flux_g) > 0:
            resid = flux_g - model_g
            sigflux = np.diff(np.percentile(resid, [25, 75]))[0] / 1.349
            not_outlier = np.abs(resid) < DISPLAY_CLIP_SIGMA * sigflux
            wave_g, flux_g, model_g = wave_g[not_outlier], flux_g[not_outlier], model_g[not_outlier]

        specax.plot(wave_g/1e4, flux_g, color=CAMERA_COLORS[icam], lw=0.7, alpha=0.75,
                   drawstyle='steps-mid', zorder=3)
        specax.plot(wave_g/1e4, model_g, color=CAMERA_COLORS_DARK[icam], lw=1.3, alpha=0.95, zorder=4)

        if len(flux_g) > 0:
            filtflux = median_filter(flux_g, 51, mode='nearest')
            sigflux = np.diff(np.percentile(flux_g - model_g, [25, 75]))[0] / 1.349
            spec_ymin = min(spec_ymin, -2 * sigflux)
            spec_ymax = max(spec_ymax, 6 * sigflux, 1.25 * np.nanmax(filtflux),
                            1.25 * np.nanmax(model_g))

    specax.set_xlim(np.min(fullwave)/1e4, np.max(fullwave)/1e4)
    specax.set_ylim(spec_ymin, spec_ymax)
    specax.set_xlabel(r'Observed-frame Wavelength ($\mu$m)')
    specax.set_ylabel(r'$F_{\lambda}\ (10^{-17}~{\rm erg}~{\rm s}^{-1}~{\rm cm}^{-2}~\AA^{-1})$')

    # shade the rest-frame window fastspecfit actually uses to fit sigma_star
    fit_obs_lo = VDISP_FIT_RESTRANGE[0] * (1 + redshift) / 1e4
    fit_obs_hi = VDISP_FIT_RESTRANGE[1] * (1 + redshift) / 1e4
    specax.axvspan(fit_obs_lo, fit_obs_hi, color='gold', alpha=0.15, zorder=1)
    specax.text(0.5 * (fit_obs_lo + fit_obs_hi), spec_ymax * 0.97,
               r'$\sigma_{\star}$ fit range', ha='center', va='top', fontsize=12,
               color='darkgoldenrod', alpha=0.9, zorder=6)

    specax.set_title(title)
    specax.grid(False)

    # ---- cutout inset, lower-right corner of the spectrum panel ----
    cut_pad = 0.012
    cut_w = 0.144
    cut_h = cut_w * fig_w_in / (1.3 * fig_h_in)   # 1.3 = cutout image's own width/height ratio
    specpos = specax.get_position()
    cutax = fig.add_axes([specpos.x1 - cut_pad - cut_w, specpos.y0 + cut_pad, cut_w, cut_h])
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
    cutax.legend(handles=handles, loc='lower left', fontsize=7, facecolor='lightgray',
                framealpha=0.6, handlelength=1.4, borderpad=0.3, labelspacing=0.2)

    # ---- right: chi2 (default) or P(sigma_star) vs sigma_star ----
    vgrid = chi2grid['VDISP'].data.astype(float)
    chi2 = chi2grid['CHI2'].data.astype(float)
    chi2min = np.min(chi2)

    if show_pdf:
        # normalized likelihood -- note the exp() suppresses essentially all
        # visual information away from the peak (that's exactly why this
        # isn't the default; see --pdf help)
        yvals = np.exp(-0.5 * (chi2 - chi2min))
        yvals /= np.trapezoid(yvals, vgrid)   # unit area -> a proper PDF over the grid
        ylabel = r'$P(\sigma_{\star})$'
    else:
        # delta-chi2: same information as the full chi2 grid, just shifted
        # to a floor of zero at the minimum; preserves the shape (and hence
        # the actual information content) at sigma_star far from the peak,
        # unlike the exponentiated/normalized likelihood above
        yvals = chi2 - chi2min
        ylabel = r'$\Delta\chi^2$'

    likeax.plot(vgrid, yvals, color='k', lw=1.5, zorder=3)
    likeax.fill_between(vgrid, yvals, color='0.85', zorder=1)
    likeax.plot(vgrid, yvals, 'o', color='0.4', ms=3, zorder=4)

    # grid's own chi2-minimum (coarse scan, used only to seed the continuous
    # refit) -- commented out for now; may show this later for comparison
    # with the production VDISP below, but it's more than we need right now
    #igrid_min = np.argmin(chi2)
    #likeax.axvline(vgrid[igrid_min], color='0.5', lw=1.2, ls=':', zorder=2)

    # production VDISP/VDISP_IVAR (continuous refit; uncertainty from Monte Carlo
    # repeats, not from this grid's curvature -- shown for comparison, not as a
    # fit to this curve)
    vdisp = specphot['VDISP']
    vdisp_ivar = specphot['VDISP_IVAR']
    likeax.axvline(vdisp, color='firebrick', lw=1.8, ls='-', zorder=5)
    if vdisp_ivar > 0:
        vdisp_sigma = 1. / np.sqrt(vdisp_ivar)
        likeax.axvspan(vdisp - vdisp_sigma, vdisp + vdisp_sigma, color='firebrick',
                      alpha=0.15, zorder=2)
        vlabel = r'$\sigma_{\star}=' + f'{vdisp:.0f}\\pm{vdisp_sigma:.0f}' + r'$ km/s'
    else:
        vlabel = r'$\sigma_{\star}=' + f'{vdisp:.0f}' + r'$ km/s'

    handles = [
        Line2D([0], [0], color='firebrick', lw=1.8, ls='-', label=vlabel),
        #Line2D([0], [0], color='0.5', lw=1.2, ls=':',
        #      label=f'grid $\\chi^2$ min = {vgrid[igrid_min]:.0f} km/s'),
    ]
    likeax.legend(handles=handles, loc='upper right', fontsize=10, framealpha=0.85)

    likeax.set_xlim(vgrid.min(), vgrid.max())
    likeax.set_ylim(0, None)
    likeax.set_xlabel(r'$\sigma_{\star}$ (km/s)')
    likeax.set_ylabel(ylabel)
    likeax.grid(False)

    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f'Wrote {outfile}')


if __name__ == '__main__':
    main()
