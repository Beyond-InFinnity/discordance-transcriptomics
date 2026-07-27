"""Volumetric → surface → parcel projection (CLAUDE.md §7.2).

The Epp maps are volumetric MNI152; spin tests need spherical coordinates.
The sanctioned route is::

    MNI152 volume
      → neuromaps.transforms.mni152_to_fsaverage(density='10k')
      → parcellate with the Schaefer fsaverage5 annot
      → parcel vector (100 values, LH)

**R4: never hand-roll a coordinate-space transform.** Every space conversion
here goes through ``neuromaps.transforms``. Successive interpolation is a
silent, unrecoverable error source, so a map is projected exactly once.

Known limitation to state in the paper (§7.2): this discards subcortex and
cerebellum. Acceptable — the hypothesis concerns cortical DMN nodes.

Masking
-------
The per-subject mqBOLD maps are **unmasked** and contain physiologically
impossible values (OEF > 1) wherever the R2'/CBV denominator is small. Trilinear
interpolation then smears those values, and zeros from outside the brain, across
neighbouring vertices. :func:`project_to_parcels` therefore takes a ``mask``
and applies it *on the surface, after projection*, so masked-out vertices never
contribute to a parcel mean.
"""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import nibabel as nib
import numpy as np

logger = logging.getLogger(__name__)

__all__ = [
    "ParcelResult",
    "annot_to_gifti_label",
    "get_schaefer_annot",
    "gifti_atlas_paths",
    "project_to_parcels",
    "schaefer_gifti_for_nulls",
    "surface_from_mni152",
]

# neuromaps fsaverage density -> the FreeSurfer subject whose annot matches it.
_DENSITY_TO_FSAVERAGE = {
    "3k": "fsaverage3",
    "10k": "fsaverage5",
    "41k": "fsaverage6",
    "164k": "fsaverage",
}

_DENSITY_VERTICES = {"3k": 2562, "10k": 10242, "41k": 40962, "164k": 163842}


@dataclass(frozen=True)
class ParcelResult:
    """A parcel vector plus the coverage information needed to report it."""

    values: np.ndarray
    n_parcels: int
    n_empty: int
    vertices_per_parcel: np.ndarray
    frac_vertices_kept: float
    hemi: str

    @property
    def empty_parcels(self) -> np.ndarray:
        """Indices of parcels with no surviving vertices."""
        return np.flatnonzero(~np.isfinite(self.values))


@lru_cache(maxsize=8)
def get_schaefer_annot(
    n_parcels: int = 200,
    networks: int = 7,
    density: str = "10k",
    hemi: str = "L",
) -> Path:
    """Path to the Schaefer 2018 ``.annot`` matching a neuromaps density.

    Parameters
    ----------
    n_parcels : int
        Total parcels across both hemispheres (200 -> 100 per hemisphere).
    networks : {7, 17}
        Network solution.
    density : {'3k', '10k', '41k', '164k'}
        neuromaps fsaverage density; mapped to the matching FreeSurfer subject.
    hemi : {'L', 'R'}
        Hemisphere. Primary analyses are left only (R3).

    Returns
    -------
    Path
        Location of the ``.annot`` file.
    """
    from netneurotools import datasets as nnd

    try:
        version = _DENSITY_TO_FSAVERAGE[density]
    except KeyError as exc:
        raise ValueError(
            f"density {density!r} not one of {sorted(_DENSITY_TO_FSAVERAGE)}"
        ) from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        atlas = nnd.fetch_schaefer2018(version=version, verbose=0)
    key = f"{n_parcels}Parcels{networks}Networks"
    if key not in atlas:
        raise KeyError(f"{key!r} not in Schaefer atlas; have {sorted(atlas)[:6]}...")
    return Path(getattr(atlas[key], hemi.upper()))


@lru_cache(maxsize=8)
def get_parcellation(
    name: str, density: str = "10k", hemi: str = "L"
) -> tuple[np.ndarray, nib.gifti.GiftiImage, int]:
    """Resolve a parcellation name to (labels, gifti-for-nulls, n_parcels).

    Supports the primary Schaefer solutions and the Desikan-Killiany
    sensitivity parcellation (§7.1). Background label is 0 in both cases and is
    named such that ``neuromaps.nulls`` drops it.

    Parameters
    ----------
    name : {'schaefer200x7', 'schaefer400x7', 'dk68'}
    density : str
        neuromaps fsaverage density.
    hemi : {'L', 'R'}

    Returns
    -------
    labels : ndarray
        Per-vertex integer labels; 0 is background.
    gifti : GiftiImage
        Label image with a populated label table, for null generation.
    n_parcels : int
        Number of non-background parcels in this hemisphere.
    """
    if name == "dk68":
        import abagen

        idx = 0 if hemi.upper() == "L" else 1
        gii = nib.load(abagen.fetch_desikan_killiany(surface=True)["image"][idx])
        labels = np.asarray(gii.agg_data()).astype(int)
        if labels.shape[0] != _DENSITY_VERTICES[density]:
            raise ValueError(
                f"DK atlas has {labels.shape[0]} vertices but density {density!r} "
                f"expects {_DENSITY_VERTICES[density]}"
            )
        return labels, gii, int(labels.max())

    m = re.fullmatch(r"schaefer(\d+)x(\d+)", name)
    if not m:
        raise ValueError(
            f"unknown parcellation {name!r}; "
            "expected 'schaefer<N>x<networks>' or 'dk68'"
        )
    n_total, networks = int(m.group(1)), int(m.group(2))
    annot = get_schaefer_annot(n_total, networks, density, hemi)
    labels, _, _ = nib.freesurfer.read_annot(str(annot))
    return labels.astype(int), annot_to_gifti_label(annot), int(labels.max())


def annot_to_gifti_label(annot: str | Path) -> nib.gifti.GiftiImage:
    """Convert a FreeSurfer ``.annot`` to a GIFTI label image.

    ``neuromaps.nulls`` needs GIFTI label images, and it decides which parcels
    are background by looking up **label names** against its ``PARCIGNORE``
    list. A GIFTI built without a label table therefore keeps the medial wall
    as a real parcel, and the returned spin indices run one past the end of the
    data array. Carrying the annot's names across is what makes the drop work.

    Parameters
    ----------
    annot : path
        FreeSurfer ``.annot`` file.

    Returns
    -------
    nibabel.gifti.GiftiImage
        Label image with its label table populated from the annot.
    """
    labels, _, names = nib.freesurfer.read_annot(str(annot))
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in names]

    table = nib.gifti.GiftiLabelTable()
    for key, name in enumerate(names):
        entry = nib.gifti.GiftiLabel(key=key)
        entry.label = name
        table.labels.append(entry)

    darray = nib.gifti.GiftiDataArray(
        labels.astype(np.int32),
        intent="NIFTI_INTENT_LABEL",
        datatype="NIFTI_TYPE_INT32",
    )
    return nib.gifti.GiftiImage(darrays=[darray], labeltable=table)


def gifti_atlas_paths(
    parcellation: str, density: str = "10k", cache_dir: Path | None = None
) -> tuple[str, str]:
    """Materialise a parcellation as (LH, RH) GIFTI label files on disk.

    ``abagen.get_expression_data`` accepts surface atlases only as GIFTI
    **file paths** — not FreeSurfer ``.annot``, and not in-memory
    ``GiftiImage`` objects. Schaefer ships as ``.annot``, so it is converted
    once and cached.

    Parameters
    ----------
    parcellation : str
        Parcellation name.
    density : str
        neuromaps fsaverage density.
    cache_dir : Path, optional
        Where to write. Defaults to ``data/.cache/parcellations``.

    Returns
    -------
    (lh_path, rh_path) : tuple of str
    """
    from ..utils.config import REPO_ROOT

    cache_dir = cache_dir or REPO_ROOT / "data" / ".cache" / "parcellations"
    cache_dir.mkdir(parents=True, exist_ok=True)

    out = []
    for hemi in ("L", "R"):
        dest = cache_dir / f"{parcellation}_{density}_hemi-{hemi}.label.gii"
        if not dest.exists():
            if parcellation == "dk68":
                import abagen

                idx = 0 if hemi == "L" else 1
                src = abagen.fetch_desikan_killiany(surface=True)["image"][idx]
                nib.save(nib.load(src), dest)
            else:
                m = re.fullmatch(r"schaefer(\d+)x(\d+)", parcellation)
                if not m:
                    raise ValueError(f"unknown parcellation {parcellation!r}")
                annot = get_schaefer_annot(
                    int(m.group(1)), int(m.group(2)), density, hemi
                )
                nib.save(annot_to_gifti_label(annot), dest)
            logger.info("wrote %s", dest)
        out.append(str(dest))
    return out[0], out[1]


def schaefer_gifti_for_nulls(
    n_parcels: int = 200,
    networks: int = 7,
    density: str = "10k",
    hemi: str = "L",
) -> tuple[nib.gifti.GiftiImage]:
    """Schaefer parcellation packaged for ``neuromaps.nulls``.

    Returned as a 1-tuple for left-hemisphere-only analyses (R3), which is the
    shape ``get_parcel_centroids`` expects for a single hemisphere.
    """
    return (annot_to_gifti_label(get_schaefer_annot(n_parcels, networks, density, hemi)),)


def surface_from_mni152(
    img: str | Path | nib.Nifti1Image,
    density: str = "10k",
    method: Literal["linear", "nearest"] = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """Project an MNI152 volume onto fsaverage surfaces via neuromaps (R4).

    Parameters
    ----------
    img : path or Nifti1Image
        Volumetric map in MNI152 space.
    density : str
        Target fsaverage density.
    method : {'linear', 'nearest'}
        Interpolation. Use ``nearest`` for masks and any integer-valued map;
        linear interpolation of a binary mask produces meaningless fractions.

    Returns
    -------
    lh, rh : ndarray
        Vertex-wise data per hemisphere.
    """
    from neuromaps import transforms

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gii = transforms.mni152_to_fsaverage(img, fsavg_density=density, method=method)
    return gii[0].agg_data(), gii[1].agg_data()


def project_to_parcels(
    img: str | Path | nib.Nifti1Image,
    density: str = "10k",
    n_parcels: int = 200,
    networks: int = 7,
    hemi: str = "L",
    mask: np.ndarray | None = None,
    drop_zero: bool = True,
    valid_range: tuple[float, float] | None = None,
    method: Literal["linear", "nearest"] = "linear",
    parcellation: str | None = None,
) -> ParcelResult:
    """Project an MNI152 volume to a parcel vector.

    Parameters
    ----------
    img : path or Nifti1Image
        Volumetric map in MNI152 space.
    parcellation : str, optional
        Parcellation name (e.g. ``'dk68'``). Takes precedence over the
        ``n_parcels``/``networks`` pair, which only addresses Schaefer.
    density, n_parcels, networks, hemi
        Parcellation specification; see :func:`get_parcellation`.
    mask : ndarray, optional
        Boolean vertex mask in the *same* surface space, True where data is
        usable. Build it with :func:`surface_from_mni152` using
        ``method='nearest'`` so it stays binary.
    drop_zero : bool
        Treat exact zeros as missing. Volumetric maps use 0 as "outside brain",
        and interpolation drags those zeros into edge vertices.
    valid_range : (lo, hi), optional
        Discard values outside this range before averaging. For OEF, ``(0, 1)``
        removes the physiologically impossible values present in the unmasked
        per-subject maps.
    method : {'linear', 'nearest'}
        Interpolation for the data volume.

    Returns
    -------
    ParcelResult
    """
    lh, rh = surface_from_mni152(img, density=density, method=method)
    data = lh if hemi.upper() == "L" else rh

    if parcellation is not None:
        labels, _, _n = get_parcellation(parcellation, density, hemi)
    else:
        labels, _, _n = get_parcellation(
            f"schaefer{n_parcels}x{networks}", density, hemi
        )
    if labels.shape != data.shape:
        raise ValueError(
            f"annot has {labels.shape[0]} vertices but surface has {data.shape[0]}; "
            f"density {density!r} and parcellation are mismatched"
        )

    usable = np.isfinite(data)
    if drop_zero:
        usable &= data != 0
    if valid_range is not None:
        lo, hi = valid_range
        usable &= (data >= lo) & (data <= hi)
    if mask is not None:
        mask = np.asarray(mask).astype(bool).ravel()
        if mask.shape != data.shape:
            raise ValueError(
                f"mask has {mask.shape[0]} vertices, surface has {data.shape[0]}"
            )
        usable &= mask

    # Label 0 is the medial wall; parcels are 1..n.
    n = int(labels.max())
    values = np.full(n, np.nan)
    counts = np.zeros(n, dtype=int)
    for i in range(1, n + 1):
        sel = (labels == i) & usable
        counts[i - 1] = int(sel.sum())
        if counts[i - 1]:
            values[i - 1] = float(data[sel].mean())

    in_cortex = labels > 0
    result = ParcelResult(
        values=values,
        n_parcels=n,
        n_empty=int(np.isnan(values).sum()),
        vertices_per_parcel=counts,
        frac_vertices_kept=float((usable & in_cortex).sum() / max(in_cortex.sum(), 1)),
        hemi=hemi.upper(),
    )
    if result.n_empty:
        logger.info(
            "%d/%d parcels have no usable vertices (%.1f%% of cortical vertices kept)",
            result.n_empty,
            n,
            100 * result.frac_vertices_kept,
        )
    return result
