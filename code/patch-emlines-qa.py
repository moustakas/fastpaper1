#!/usr/bin/env python
"""Fitting-in-patches / line-masking QA figure (Section: "Fast Fitting in
Wavelength Patches").

Reads a single DESI target directly from a Redrock file plus its companion
coadd file -- no completed fastspec fit is required, since the line masker
(fastspecfit.linemasker.LineMasker) runs *before* the continuum and
emission-line models exist. Calls LineMasker.build_linemask(...,
return_patchfit=True) to recover the exact per-patch fit arrays (data,
best-fit model, local linear continuum +/- noise) that fastspecfit's own
--debug_plots QA uses, and builds a two-row publication figure: the full
per-camera spectrum on top, with a zoom panel below for each patch (default:
all patches fit for this object). Written to tex/figures/.

Requires a fastspecfit build with the `return_patchfit` option on
LineMasker.build_linemask().

Example (mini specprod built with fastspecfit's build-mini-specprod):

    python code/patch-emlines-qa.py \
        --redrockfile data/redrock-main-bright-15344.fits \
        --outfile tex/figures/example-emlines.pdf

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
DEFAULT_OUTFILE = os.path.join(REPODIR, 'tex', 'figures', 'example-emlines.pdf')
DEFAULT_FPHOTODIR = os.path.join(REPODIR, 'data')


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--redrockfile', required=True,
                        help='Full path to a Redrock file; the matching coadd-*.fits '
                             'file must be in the same directory.')
    parser.add_argument('--targetid', type=int, default=None,
                        help='TARGETID to process. Only needed if --redrockfile '
                             'contains more than one target.')
    parser.add_argument('--patches', nargs='+', default=None,
                        help="Patch IDs (the 'patch' column of data/emlines.ecsv, "
                             'possibly merged, e.g. \'tu\') to feature, one zoom panel '
                             'each, left to right. Default: all patches fit for this '
                             'object.')
    parser.add_argument('--outfile', default=DEFAULT_OUTFILE,
                        help='Output path for the figure.')
    parser.add_argument('--fphotodir', default=None,
                        help='Full path to the Legacy Surveys Tractor catalog tree '
                             f'(default: {DEFAULT_FPHOTODIR} if present, else $FPHOTO_DIR).')
    args = parser.parse_args()

    from fastspecfit.singlecopy import sc_data
    from fastspecfit.io import DESISpectra
    from fastspecfit.linemasker import LineMasker

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

    LM = LineMasker(sc_data.emlines.table, sc_data.constraints)
    out = LM.build_linemask(
        specdata['coadd_wave'], specdata['coadd_flux'], specdata['coadd_ivar'],
        specdata['coadd_res'], uniqueid=specdata['uniqueid'],
        redshift=specdata['redshift'], debug_plots=False, return_patchfit=True)

    print(f'  balmerbroad={out["balmerbroad"]}, '
          f'linesigma_balmer_broad={out["linesigma_balmer_broad"]:.0f} km/s, '
          f'linevshift_balmer_broad={out["linevshift_balmer_broad"]:.0f} km/s')

    make_figure(specdata, objmeta, Spec.coadd_type, out, args.patches, args.outfile)


def make_figure(specdata, objmeta, coadd_type, out, patchids, outfile):
    """Full per-camera spectrum on top, with a zoom panel below for each
    requested patch showing the data, the combined best-fit model
    (continuum+lines), the local linear continuum +/- noise, and the
    continuum pixels (unaffected by line emission) used for that local fit.
    Per-line S/N labels are restored as text (mirroring linemasker.py's own
    --debug_plots legends), but the individual Gaussian line profiles
    themselves are not drawn -- only the combined best-fit curve.
    """
    sys.path.insert(0, os.path.join(REPODIR, 'code'))
    from util import plot_style
    from matplotlib.patches import ConnectionPatch, Rectangle

    from fastspecfit.emlines import EMFitTools
    from fastspecfit.emline_fit import EMLine_MultiLines
    from fastspecfit.linemasker import LineMasker
    from fastspecfit.qa import format_niceline, _target_label
    from fastspecfit.singlecopy import sc_data

    pf = out['patchfit']

    plot_style(talk=True, font_scale=0.7, palette='colorblind')
    import matplotlib.pyplot as plt

    # per-camera (b, r, z) data/model colors, matching fastspecfit.qa.qa_fastspec
    CAMERA_COLORS = ['#468fcc', '#4caf81', '#e07a75']       # col1: data
    CAMERA_COLORS_DARK = ['#003f91', '#007f5f', '#9b2226']  # col2: model
    bbox = dict(boxstyle='round', facecolor='lightgray', alpha=0.3, edgecolor='none')

    coadd_wave = specdata['coadd_wave']
    coadd_flux = specdata['coadd_flux']

    # assign each coadd_wave pixel to a camera, using the midpoint of each
    # pair of adjacent cameras' native wavelength coverage as the cut
    cam_cuts = [0.5 * (np.max(specdata['wave0'][icam]) + np.min(specdata['wave0'][icam + 1]))
               for icam in range(len(specdata['cameras']) - 1)]
    camids = np.searchsorted(cam_cuts, coadd_wave)

    def plot_segments(ax, wave, flux, camids, keep, colors, **kwargs):
        """Plot (wave, flux) as separate line segments, one per camera and
        broken wherever `keep` is False, so gaps (masked-out line pixels,
        camera transitions) are not bridged by a connecting line."""
        n = len(wave)
        if n < 2:
            return
        breaks = np.where((np.diff(camids) != 0) | ~keep[1:] | ~keep[:-1])[0] + 1
        starts = np.concatenate(([0], breaks))
        ends = np.concatenate((breaks, [n]))
        for st, en in zip(starts, ends):
            if keep[st] and en - st > 1:
                ax.plot(wave[st:en], flux[st:en], color=colors[camids[st]], **kwargs)

    contfit = pf['contfit']
    allpatchids = list(contfit['patchid'])
    if patchids is None:
        # contfit's row order reflects internal processing order, not
        # wavelength order -- sort by each patch's starting pixel.
        wave_order = np.argsort(contfit['endpts'][:, 0])
        patchids = [allpatchids[i] for i in wave_order]

    missing = [p for p in patchids if p not in allpatchids]
    if missing:
        raise ValueError(f'Requested patch(es) {missing} were not fit; '
                         f'available patches: {allpatchids}')

    order = [allpatchids.index(p) for p in patchids]
    rows = contfit[order]
    npanel = len(rows)

    # used only to identify which lines in each patch are "real" (not fixed
    # to zero, e.g. a broad Balmer line when the narrow-only model was
    # adopted) -- we do not plot the individual per-line profiles it returns.
    lines = EMLine_MultiLines(pf['parameters'], coadd_wave, specdata['redshift'],
                              pf['linetable']['restwave'].value,
                              specdata['coadd_res'], pf['camerapix'])

    # Reconstruct the continuum pixels (unaffected by line emission) used
    # for each patch's local linear-continuum fit, using the same public
    # LineMasker.linepix_and_contpix() call and the converged kinematics
    # that build_linemask() itself used for the final adopted model.
    #
    # patchMap[patchid][1] ("index_inrange") indexes into the *further*
    # in-range-filtered subset of pf['linetable'] -- not pf['linetable']
    # itself, which is indexed by patchMap[patchid][2] ("index_full"). We
    # must therefore pass linepix_and_contpix() that same in-range subset
    # (and matching linesigmas/linevshifts slice), exactly mirroring
    # build_linemask()'s own internal fit_patches() call.
    EMFit = EMFitTools(emline_table=pf['linetable'], constraints=sc_data.constraints,
                       uniqueid=specdata['uniqueid'], stronglines=False)
    EMFit.compute_inrange_lines(specdata['redshift'],
                                wavelims=(np.min(coadd_wave), np.max(coadd_wave)))

    linesigmas = np.zeros(len(pf['linetable']))
    linesigmas[EMFit.isBroad] = out['linesigma_broad']
    linesigmas[EMFit.isNarrow] = out['linesigma_narrow']
    linesigmas[EMFit.isBalmerBroad] = out['linesigma_balmer_broad']

    linevshifts = np.zeros_like(linesigmas)
    linevshifts[EMFit.isBroad] = out['linevshift_broad']
    linevshifts[EMFit.isNarrow] = out['linevshift_narrow']
    linevshifts[EMFit.isBalmerBroad] = out['linevshift_balmer_broad']

    # nsigma=5 matches build_linemask()'s own nsigma_mask default, which is
    # what it actually uses (via fit_patches) to build patchMap's contpix.
    pix = LineMasker.linepix_and_contpix(
        coadd_wave, specdata['coadd_ivar'], pf['linetable'][EMFit.line_in_range],
        linesigmas[EMFit.line_in_range], linevshifts=linevshifts[EMFit.line_in_range],
        patchMap=pf['patchMap'], redshift=specdata['redshift'], nsigma=5.)
    patch_contpix = pix['patch_contpix']

    # contfit's endpts cover the fitted lines but not necessarily the full
    # extent of the continuum-pixel search window (which can reach much
    # farther from the line core, e.g. for patch 'i' / [OII] near the blue
    # edge) -- widen each patch's plotted window to include its contpix too,
    # so the continuum pixels we found are actually visible in the panel.
    patch_windows = {}
    for patchid, endpts in contfit[order].iterrows('patchid', 'endpts'):
        s, e = endpts
        contpix = patch_contpix.get(patchid)
        if contpix is not None and len(contpix) > 0:
            s = min(s, int(np.min(contpix)))
            e = max(e, int(np.max(contpix)) + 1)
        patch_windows[patchid] = (s, e)

    fig = plt.figure(figsize=(2.8 * npanel, 6.5), constrained_layout=True)
    # rect leaves room at bottom/left for the fig.text() axis labels added
    # below (plain fig.text() isn't margin-managed by constrained_layout,
    # unlike supxlabel/supylabel, so we reserve the space by hand)
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.12, wspace=0.0, hspace=0.14,
                                rect=(0.04, 0.045, 0.945, 0.935))
    gs = fig.add_gridspec(2, npanel, height_ratios=[1.1, 1])

    # top panel: full spectrum, camera-colored where pixels remain available
    # for continuum fitting, light gray where masked by a kept emission line
    # (out['coadd_linepix'], the final S/N-thresholded mask).
    specax = fig.add_subplot(gs[0, :])
    specax.plot(coadd_wave, coadd_flux, color='0.82', lw=0.6, zorder=1)

    full_linemask = np.zeros(len(coadd_wave), bool)
    for linepix in out['coadd_linepix'].values():
        full_linemask[linepix] = True
    plot_segments(specax, coadd_wave, coadd_flux, camids, ~full_linemask,
                 CAMERA_COLORS, lw=0.8, zorder=3)
    specax.margins(x=0)
    # lock in the range set by the spectrum itself, before overlaying the
    # (possibly much taller) per-patch line-fit models below. The lower
    # bound is sigma-clipped so a handful of noisy negative pixels don't
    # drag the whole panel down.
    from astropy.stats import sigma_clipped_stats
    _, flux_median, flux_std = sigma_clipped_stats(coadd_flux, sigma=3.)
    specax_ylim = (flux_median - 5. * flux_std, specax.get_ylim()[1])
    specax.set_ylim(specax_ylim)
    specax_ybottom = specax_ylim[0]

    # overlay every fitted patch's best-fit model (Gaussians + linear
    # continuum pedestal) directly on top of its masked (gray) stretch, so
    # masked regions show what was actually fit there instead of
    # disappearing into gray -- not just the patches featured as zoom panels
    # below. Strong lines can exceed the spectrum's own range and simply
    # clip against it, rather than rescaling the whole panel.
    for endpts, slope, intercept, pivotwave in contfit.iterrows(
            'endpts', 'slope', 'intercept', 'pivotwave'):
        s0, e0 = endpts
        modelkeep = np.ones(e0 - s0, bool)
        plot_segments(specax, coadd_wave[s0:e0], pf['bestfit'][s0:e0], camids[s0:e0],
                     modelkeep, CAMERA_COLORS_DARK, lw=1., zorder=4)
        cmodel = slope * (coadd_wave[s0:e0] - pivotwave) + intercept
        specax.plot(coadd_wave[s0:e0], cmodel, color='k', ls='--', lw=0.6, zorder=4)
    specax.set_ylim(specax_ylim)

    target = _target_label(objmeta, coadd_type)
    specax.text(0.02, 0.95, '\n'.join(target), transform=specax.transAxes,
               ha='left', va='top', fontsize=12, linespacing=1.5, bbox=bbox)

    # tight callout box around each zoomed patch's actual spectral feature
    # (sized to the local data/model extent, not the full panel height)
    callouts = []
    for patchid, endpts in rows.iterrows('patchid', 'endpts'):
        s, e = patch_windows[patchid]
        s0, e0 = endpts
        x0, x1 = coadd_wave[s], coadd_wave[e - 1]
        ylo = min(np.min(coadd_flux[s:e]), np.min(pf['bestfit'][s0:e0]))
        yhi = max(np.max(coadd_flux[s:e]), np.max(pf['bestfit'][s0:e0]))
        pad = 0.15 * (yhi - ylo)
        y0 = max(specax_ybottom, ylo - pad)
        y1 = min(specax_ylim[1], yhi + pad)
        callouts.append((x0, x1, y0, y1))

    # bottom row: one zoom panel per patch
    zoomaxes = []
    for ipanel, (i, (patchid, endpts, slope, intercept, pivotwave)) in enumerate(
            zip(order, rows.iterrows('patchid', 'endpts', 'slope', 'intercept', 'pivotwave'))):
        s0, e0 = endpts  # original patch fit window -- bestfit is only valid here
        s, e = patch_windows[patchid]  # widened to show the full continuum-pixel extent
        noise = pf['noises'][i]

        ax = fig.add_subplot(gs[1, ipanel])
        ax.plot(coadd_wave[s:e], coadd_flux[s:e], color='0.8', lw=0.8, zorder=1)

        contpix = patch_contpix.get(patchid)
        contkeep = np.zeros(e - s, bool)
        if contpix is not None:
            local = contpix[(contpix >= s) & (contpix < e)] - s
            contkeep[local] = True
        plot_segments(ax, coadd_wave[s:e], coadd_flux[s:e], camids[s:e], contkeep,
                     CAMERA_COLORS, lw=1.2, alpha=0.9, zorder=3)

        # bestfit is only defined over the original fit window [s0:e0) --
        # outside it, it drops to zero, so restrict the model curve to it.
        modelkeep = np.ones(e0 - s0, bool)
        plot_segments(ax, coadd_wave[s0:e0], pf['bestfit'][s0:e0], camids[s0:e0], modelkeep,
                     CAMERA_COLORS_DARK, lw=1.8, zorder=4)

        cmodel = slope * (coadd_wave[s:e] - pivotwave) + intercept
        ax.plot(coadd_wave[s:e], cmodel, color='k', ls='--', lw=1)
        ax.plot(coadd_wave[s:e], cmodel + noise, color='0.6', lw=0.8)
        ax.plot(coadd_wave[s:e], cmodel - noise, color='0.6', lw=0.8)

        ax.set_xlim(coadd_wave[s], coadd_wave[e - 1])
        ymax = 1.25 * np.max((np.max(coadd_flux[s:e]), np.max(pf['bestfit'][s0:e0]),
                            np.max(cmodel + noise)))
        ax.set_ylim(0., ymax)

        ax.text(0.95, 0.94, f'Patch "{patchid}"', transform=ax.transAxes,
                ha='right', va='top', fontweight='bold', fontsize=9)

        linelabels = []
        for line, iline in zip(pf['patchMap'][patchid][0], pf['patchMap'][patchid][2]):
            (ls, le), _ = lines.getLine(iline)
            if ls != le:
                linelabels.append(f'S/N({format_niceline(line)})={pf["linesnrs"][iline]:.1f}')
        ax.text(0.05, 0.94, '\n'.join(linelabels), transform=ax.transAxes,
                ha='left', va='top', fontsize=7, linespacing=1.5)

        zoomaxes.append(ax)

    # callout box in the top panel around each zoomed patch's feature, with
    # connector lines tying its bottom corners to the zoom panel below
    for ax, (x0, x1, y0, y1) in zip(zoomaxes, callouts):
        specax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                   edgecolor='k', lw=1.2, zorder=6))
        ytop = ax.get_ylim()[1]
        for x in (x0, x1):
            con = ConnectionPatch(xyA=(x, y0), coordsA='data', axesA=specax,
                                  xyB=(x, ytop), coordsB='data', axesB=ax,
                                  color='k', lw=1.2, zorder=5)
            fig.add_artist(con)

    for ax in fig.axes:
        ax.grid(False)

    # plain fig.text() rather than supxlabel/supylabel -- gives direct
    # control over placement instead of constrained_layout's auto-reserved
    # (and here, overly generous) margin; nudge x/y to taste
    labelsize = plt.rcParams.get('figure.labelsize', plt.rcParams['font.size'])
    fig.text(0.5, 0.012, r'Observed-frame Wavelength ($\mathrm{\AA}$)',
             ha='center', va='bottom', fontsize=labelsize)
    fig.text(0.012, 0.5, r'$F_{\lambda}\ (10^{-17}~{\rm erg}~{\rm s}^{-1}~{\rm cm}^{-2}~\AA^{-1})$',
             ha='left', va='center', rotation='vertical', fontsize=labelsize)

    outdir = os.path.dirname(outfile)
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    print(f'Wrote {outfile}')


if __name__ == '__main__':
    main()
