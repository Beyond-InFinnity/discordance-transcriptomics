"""Tests for config loading (§10) and result manifests (R10)."""

from __future__ import annotations

import json

import pytest
import yaml
from pydantic import ValidationError

from src.utils.config import REPO_ROOT, BaseConfig, config_hash, load_config
from src.utils.manifest import Manifest, manifest, sha256_file


@pytest.fixture
def cfg() -> BaseConfig:
    return load_config()


class TestConfigLoading:
    def test_loads_real_config(self, cfg):
        assert cfg.seed == 42  # R7
        assert cfg.parcellation.primary.n_parcels_lh == 100
        assert cfg.nulls.n_perm >= 10_000

    def test_left_hemisphere_only(self, cfg):
        """R3: primary analyses are left hemisphere only."""
        assert cfg.parcellation.primary.hemi == "L"
        assert all(s.hemi == "L" for s in cfg.parcellation.sensitivity)

    def test_spearman_is_default_metric(self, cfg):
        """§11: brain maps are rarely bivariate normal."""
        assert cfg.stats.correlation == "spearman"

    def test_gate_thresholds_match_protocol(self, cfg):
        """§9 thresholds must not drift silently."""
        assert cfg.gates.p0_reliability.pass_threshold == 0.5
        assert cfg.gates.p0_reliability.caveat_threshold == 0.3
        assert cfg.gates.p0_reliability.n_splits == 1000
        assert cfg.gates.p0_dropout.severe_threshold == 0.5

    def test_dropout_proxy_is_mandatory_covariate(self, cfg):
        """Phase 0b carries forward into every downstream model."""
        assert "dropout_snr_coverage" in cfg.covariates.mandatory

    def test_t2star_is_not_a_covariate(self, cfg):
        """t2star is circular with OEF (mqBOLD derives OEF from T2* via R2'),
        so it must never be used as the dropout control."""
        assert not any("t2star" in c for c in cfg.covariates.mandatory)

    def test_venous_partial_volume_is_modelled(self, cfg):
        """Brain-vs-vein is a confound the protocol must control, not just note."""
        assert "venous_partial_volume" in cfg.covariates.mandatory

    def test_is_immutable(self, cfg):
        with pytest.raises(ValidationError):
            cfg.seed = 1

    def test_unknown_key_rejected(self, tmp_path):
        raw = yaml.safe_load((REPO_ROOT / "config/base.yaml").read_text())
        raw["a_typo_nobody_noticed"] = 1
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.safe_dump(raw))
        with pytest.raises(ValidationError, match="a_typo_nobody_noticed"):
            load_config(p)

    def test_path_helper_resolves_from_repo_root(self, cfg):
        assert cfg.path("nulls").is_absolute()
        assert cfg.path("nulls").name == "nulls"


class TestConfigHash:
    def test_deterministic(self, cfg):
        assert config_hash(cfg) == config_hash(cfg)

    def test_insensitive_to_dict_order(self):
        assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})

    def test_sensitive_to_values(self):
        assert config_hash({"a": 1}) != config_hash({"a": 2})


class TestFrozenGenesets:
    """R5: the gene sets are frozen before Phase 4."""

    @pytest.fixture
    def gs(self):
        return yaml.safe_load((REPO_ROOT / "config/genesets.yaml").read_text())

    def test_marked_frozen(self, gs):
        assert gs["meta"]["frozen"] is True

    def test_exploratory_is_empty_at_freeze(self, gs):
        """Anything here is post-hoc and may not carry confirmatory claims."""
        assert gs["exploratory"] == []

    def test_all_hallmark_sets_present(self, gs):
        names = {s["name"] for s in gs["msigdb"]}
        assert names == {
            "HALLMARK_GLYCOLYSIS",
            "HALLMARK_OXIDATIVE_PHOSPHORYLATION",
            "HALLMARK_ANGIOGENESIS",
            "HALLMARK_HYPOXIA",
            "GOBP_BLOOD_VESSEL_MORPHOGENESIS",
        }

    def test_curated_sets_have_sources(self, gs):
        """Every curated set must cite its source paper (§8.1)."""
        for name, spec in gs["curated"].items():
            assert spec.get("source"), f"{name} has no source citation"

    def test_h1_directions_match_hypothesis(self, gs):
        """H1: glycolytic/vascular positive, oxphos negative."""
        d = {s["name"]: s["direction_h1"] for s in gs["msigdb"]}
        assert d["HALLMARK_GLYCOLYSIS"] == "positive"
        assert d["HALLMARK_OXIDATIVE_PHOSPHORYLATION"] == "negative"
        assert gs["curated"]["glycolytic_enzymes"]["direction_h1"] == "positive"
        assert gs["curated"]["mitochondrial_density_proxy"]["direction_h1"] == "negative"

    def test_control_sets_have_no_directional_prediction(self, gs):
        """Interneuron subclass is a control — predicting a direction would be
        post-hoc rationalisation."""
        assert gs["curated"]["interneuron_subclass"]["direction_h1"] is None


class TestManifest:
    def test_writes_json(self, cfg, tmp_path):
        with manifest("unit_test", cfg, results_dir=tmp_path) as man:
            man.record(rho=0.42, p_spin=0.013)
            man.note("a note")

        out = json.loads((tmp_path / "unit_test.manifest.json").read_text())
        assert out["seed"] == 42
        assert out["results"]["rho"] == 0.42
        assert out["notes"] == ["a note"]

    def test_records_required_r10_fields(self, cfg, tmp_path):
        with manifest("r10", cfg, results_dir=tmp_path) as man:
            man.record(x=1)
        out = json.loads((tmp_path / "r10.manifest.json").read_text())
        for field in (
            "git_sha",
            "config_hash",
            "seed",
            "wall_clock_sec",
            "packages",
            "inputs",
        ):
            assert field in out, f"R10 requires {field}"

    def test_records_package_versions(self, cfg, tmp_path):
        with manifest("pkgs", cfg, results_dir=tmp_path) as man:
            man.record(x=1)
        pkgs = json.loads((tmp_path / "pkgs.manifest.json").read_text())["packages"]
        assert pkgs["numpy"] and pkgs["abagen"] and pkgs["neuromaps"]

    def test_checksums_inputs(self, cfg, tmp_path):
        f = tmp_path / "input.txt"
        f.write_text("hello")
        with manifest("ck", cfg, inputs=[f], results_dir=tmp_path) as man:
            man.record(x=1)
        out = json.loads((tmp_path / "ck.manifest.json").read_text())
        assert out["inputs"][str(f)] == sha256_file(f)

    def test_writes_manifest_even_on_failure(self, cfg, tmp_path):
        """A failed run must leave evidence, not nothing."""
        with (
            pytest.raises(RuntimeError),
            manifest("boom", cfg, results_dir=tmp_path) as man,
        ):
            man.record(partial=True)
            raise RuntimeError("analysis exploded")

        out = json.loads((tmp_path / "boom.manifest.json").read_text())
        assert any("FAILED" in n for n in out["notes"])
        assert out["results"]["partial"] is True

    def test_missing_input_recorded_not_raised(self, cfg, tmp_path):
        m = Manifest(name="x", seed=42)
        m.add_input(tmp_path / "does_not_exist")
        assert "MISSING" in m.inputs.values()


class TestOutputDeclarationsAreConstructible:
    """A crash while *declaring* outputs discards the whole run.

    The manifest write is the last statement in every phase script, after all
    the compute. ``p5_hierarchy`` declared its outputs by globbing::

        outputs=[str(p) for p in sorted(out_dir.glob(f"p5_*{tag}*.csv"))]

    ``tag`` is ``""`` for the pre-registered specification, so the pattern
    became ``p5_**.csv``. pathlib rejects ``**`` unless it is an entire path
    component, so the run raised ValueError *after* computing and writing every
    result, and the chained sensitivity run never started. Four hours of spin
    nulls produced no manifest.

    Any glob pattern built by interpolation must therefore survive its
    interpolations being empty.
    """

    @staticmethod
    def _glob_patterns(path):
        """Every ``.glob(...)`` pattern in a file, interpolations blanked."""
        import ast

        out = []
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in ("glob", "rglob")
                and node.args
            ):
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                out.append((node.lineno, arg.value))
            elif isinstance(arg, ast.JoinedStr):
                # Blank every {..} to model the shortest possible expansion.
                lit = "".join(
                    v.value
                    for v in arg.values
                    if isinstance(v, ast.Constant) and isinstance(v.value, str)
                )
                out.append((node.lineno, lit))
        return out

    def test_every_glob_pattern_survives_empty_interpolation(self):
        import pathlib

        root = pathlib.Path(__file__).resolve().parents[1]
        offenders = []
        for py in sorted((root / "scripts").glob("*.py")) + sorted(
            (root / "src").rglob("*.py")
        ):
            for lineno, pattern in self._glob_patterns(py):
                try:
                    next(pathlib.Path(root).glob(pattern), None)
                except ValueError as exc:
                    offenders.append(f"{py.name}:{lineno}  {pattern!r}  {exc}")
        assert not offenders, "unconstructible glob patterns:\n  " + "\n  ".join(
            offenders
        )
