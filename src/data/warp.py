"""Apply the authors' own T1w→MNI152 warps — CLAUDE.md R4.

R4 forbids **estimating** a coordinate-space transform by hand. It does not
forbid applying one somebody else already estimated and shipped, which is what
this module does: ds004873 includes fmriprep-derived ANTs composite transforms
(``sub-*_from-T1w_to-MNI152NLin6Asym_mode-image_xfm.h5``) for 40 subjects.

Why this is needed
------------------
The authors compute CMRO₂ for the ``calc`` condition from the **CBV-corrected**
variant (``desc-CBV``), because CBV itself changes during the task. Every other
condition uses ``desc-orig``. Their ``B_Fig1.ipynb`` selects on exactly that::

    if par == 'cmro2' and cond == 'calc': ... desc-CBV ...
    if par == 'cmro2' and cond != 'calc': ... desc-orig ...

But ``desc-CBV`` is published only in native T2 and T1w space — **not** in
MNI152. Using ``desc-orig`` for calc, as the first pass did, is therefore not
what their pipeline does, and it plausibly explains the implausibly low
coupling ratio it produced (median n ≈ 0.27 against a task-activation norm of
2-4).

Warping their T1w-space ``desc-CBV`` maps with their own transform recovers the
variant they used, for 40 subjects rather than 30.
"""

from __future__ import annotations

import logging
from pathlib import Path

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

__all__ = ["apply_t1w_to_mni152", "warp_subject_map"]


def apply_t1w_to_mni152(
    moving: str | Path | nib.Nifti1Image,
    transform: str | Path,
    reference: str | Path | nib.Nifti1Image,
    order: int = 1,
) -> nib.Nifti1Image:
    """Warp a T1w-space volume into MNI152 using an ANTs ``.h5`` transform.

    Parameters
    ----------
    moving : path or Nifti1Image
        Volume in the subject's T1w space.
    transform : path
        ANTs composite ``.h5`` transform, T1w → MNI152NLin6Asym.
    reference : path or Nifti1Image
        Target grid. Use the dataset's own ``MNI152_T1_2mm.nii.gz`` so the
        output lands on exactly the grid the other MNI152 maps use.
    order : int
        Spline interpolation order. 1 (linear) for continuous quantitative
        maps; 0 for label images.

    Returns
    -------
    nibabel.Nifti1Image
        Resampled into the reference grid.
    """
    from nitransforms.manip import TransformChain

    mov = nib.load(moving) if not isinstance(moving, nib.Nifti1Image) else moving
    ref = (
        nib.load(reference) if not isinstance(reference, nib.Nifti1Image) else reference
    )

    # The output inherits the REFERENCE's voxel ordering, and ds004873 ships two
    # MNI152 grids with opposite handedness:
    #
    #   MNI152_T1_2mm.nii.gz          LAS, det -8   (bundled FSL template)
    #   sub-*_space-MNI152_*.nii.gz   RAS, det +8   (every published data map)
    #
    # Warping into the LAS template and then comparing arrays element-wise
    # against the RAS maps is exactly a left-right flip. Measured against ground
    # truth it scored r = 0.52 with medians matching to 2 d.p. and mask Dice
    # 0.95 — i.e. it looked like a plausible map, not an error. A published RAS
    # map as the grid gives r = 1.000000.
    #
    # For a left-hemisphere-only project (R3) an undetected flip means silently
    # analysing the right hemisphere, so this is a hard failure, not a warning.
    if nib.aff2axcodes(ref.affine)[0] != "R":
        raise ValueError(
            f"reference grid is {nib.aff2axcodes(ref.affine)}, not RAS. Every "
            "published ds004873 MNI152 map is RAS; warping into an LAS grid "
            "(e.g. the bundled MNI152_T1_2mm.nii.gz) yields a left-right "
            "flipped result that looks physiologically plausible. Pass a "
            "published space-MNI152 map as the reference instead."
        )

    xfm = TransformChain.from_filename(str(transform), fmt="itk")
    out = xfm.apply(mov, reference=ref, order=order)

    if out.shape[:3] != ref.shape[:3]:
        raise ValueError(
            f"warped image is {out.shape[:3]}, reference grid is {ref.shape[:3]}"
        )
    return out


def validate_warp(
    t1w_map: str | Path,
    published_mni: str | Path,
    transform: str | Path,
    reference: str | Path,
    min_r: float = 0.99,
) -> float:
    """Check a subject's warp against a map published in both spaces.

    ``desc-orig`` CMRO₂ and OEF exist in *both* T1w and MNI152 for most
    subjects, which gives per-subject ground truth: warp the T1w version and
    correlate it against the published MNI152 version. Anything below ``min_r``
    means the transform, the reference grid, or the file pairing is wrong for
    that subject.

    Returns
    -------
    float
        Pearson r against the published map.

    Raises
    ------
    ValueError
        If the correlation falls below ``min_r``.
    """
    from scipy.stats import pearsonr

    warped = np.asarray(apply_t1w_to_mni152(t1w_map, transform, reference).dataobj)
    truth = np.asarray(nib.load(published_mni).dataobj)
    ok = np.isfinite(warped) & np.isfinite(truth) & (warped != 0) & (truth != 0)
    r = float(pearsonr(warped[ok], truth[ok]).statistic)
    if r < min_r:
        raise ValueError(
            f"warp validation failed: r={r:.4f} < {min_r} against "
            f"{Path(published_mni).name}. A left-right flip scores ~0.52 here."
        )
    return r


def warp_subject_map(
    subject: str,
    moving: Path,
    transform: Path,
    reference: Path,
    out_path: Path,
    overwrite: bool = False,
) -> Path:
    """Warp one subject's map and cache it to ``out_path``.

    Skips work when the output already exists, so a rerun is cheap.

    Returns
    -------
    Path
        Location of the warped image.
    """
    if out_path.exists() and not overwrite:
        logger.debug("%s already warped, skipping", subject)
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    warped = apply_t1w_to_mni152(moving, transform, reference)

    data = np.asarray(warped.dataobj)
    finite = np.isfinite(data) & (data != 0)
    if not finite.any():
        raise ValueError(f"{subject}: warped image is entirely zero/non-finite")

    nib.save(warped, out_path)
    logger.info(
        "%s warped -> %s (%d non-zero voxels, median %.1f)",
        subject,
        out_path.name,
        int(finite.sum()),
        float(np.median(data[finite])),
    )
    return out_path
