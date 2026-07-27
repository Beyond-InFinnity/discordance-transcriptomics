"""Typed configuration loading.

All parameters live in ``config/*.yaml`` (CLAUDE.md §10 — no magic numbers in
code). This module parses them into pydantic models so a typo in a YAML key
fails loudly at load time rather than silently changing an analysis.

The config hash produced by :func:`config_hash` is what keys the joblib cache
and what gets recorded in every result manifest (R10).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "REPO_ROOT",
    "BaseConfig",
    "ParcellationSpec",
    "config_hash",
    "load_config",
]

REPO_ROOT = Path(__file__).resolve().parents[2]


class _Strict(BaseModel):
    """Reject unknown keys so YAML typos surface immediately."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ParcellationSpec(_Strict):
    """One parcellation, always left-hemisphere for primary use (R3)."""

    name: str
    n_parcels: int
    n_parcels_lh: int
    space: str
    density: str
    hemi: Literal["L", "R"]
    networks: int | None = None


class ParcellationConfig(_Strict):
    primary: ParcellationSpec
    sensitivity: list[ParcellationSpec]


class NullConfig(_Strict):
    """R1 — every spatial correlation goes through one of these."""

    surface_method: Literal["alexander_bloch", "vasa", "hungarian"]
    volume_method: Literal["burt2020", "burt2018", "moran"]
    n_perm: int = Field(ge=1000)
    cache: bool


class CompetitiveConfig(_Strict):
    """R2 — gene-set nulls matched on size and differential stability."""

    n_draws: int = Field(ge=1000)
    match_on: list[str]
    ds_n_bins: int


class ExpressionPrimary(_Strict):
    probe_selection: str
    lr_mirror: str | None
    missing: str | None
    tolerance: int
    norm_matched: bool
    sample_norm: str
    gene_norm: str
    donors: str


class ExpressionConfig(_Strict):
    primary: ExpressionPrimary
    stability_threshold: float


class TargetConfig(_Strict):
    maps: list[str]
    coupling_n_transform: Literal["angle", "signed_log"]
    coupling_n_transform_sensitivity: Literal["angle", "signed_log"]
    n_tasks: int


class StatsConfig(_Strict):
    correlation: Literal["spearman", "pearson"]
    fdr_method: str
    alpha: float


class ReliabilityGate(_Strict):
    n_splits: int
    metric: str
    pass_threshold: float
    caveat_threshold: float


class DropoutGate(_Strict):
    severe_threshold: float
    evaluate_on: Literal["masked", "unmasked"] = "masked"


class ReproductionGate(_Strict):
    """Phase 1 — quantitative, not 'figures look similar'."""

    metric: Literal["spearman", "pearson"]
    pass_threshold: float
    caveat_threshold: float
    reference_map: str


class GateConfig(_Strict):
    p0_reliability: ReliabilityGate
    p0_dropout: DropoutGate
    p1_reproduction: ReproductionGate


class CovariateConfig(_Strict):
    hierarchy: list[str]
    mandatory: list[str]


class LoggingConfig(_Strict):
    level: str
    format: str


class BaseConfig(_Strict):
    """Top-level ``config/base.yaml``."""

    seed: int
    paths: dict[str, str]
    parcellation: ParcellationConfig
    nulls: NullConfig
    competitive: CompetitiveConfig
    expression: ExpressionConfig
    targets: TargetConfig
    stats: StatsConfig
    gates: GateConfig
    covariates: CovariateConfig
    logging: LoggingConfig

    def path(self, key: str) -> Path:
        """Resolve a configured path relative to the repo root."""
        return REPO_ROOT / self.paths[key]


def load_config(path: str | Path = "config/base.yaml") -> BaseConfig:
    """Load and validate the base configuration.

    Parameters
    ----------
    path : str or Path
        Path to the YAML config, absolute or relative to the repo root.

    Returns
    -------
    BaseConfig
        Validated, immutable configuration object.
    """
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with p.open() as fh:
        raw = yaml.safe_load(fh)
    return BaseConfig(**raw)


def config_hash(obj: Any, length: int = 12) -> str:
    """Deterministic short hash of a config object or plain dict.

    Used to key the multiverse cache and to stamp manifests (R10). Sorting keys
    makes the hash invariant to dict ordering, so the same parameters always
    produce the same cache key across runs and machines.

    Parameters
    ----------
    obj : Any
        A pydantic model, dict, or any JSON-serialisable structure.
    length : int
        Number of hex characters to return.

    Returns
    -------
    str
        Truncated SHA256 hex digest.
    """
    if isinstance(obj, BaseModel):
        obj = obj.model_dump(mode="json")
    payload = json.dumps(obj, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()[:length]
