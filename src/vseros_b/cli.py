"""
Command line entry-point for orchestrating the Stage‑1 recall pipeline.

The CLI keeps the contract intentionally lightweight: every command loads
the train/validation context (deduped interactions, cached aggregates, the
sample submission) and then delegates the heavy lifting to existing
experiment classes under ``vseros_b.exps``.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from vseros_b import data
from vseros_b.config import PATHS, WANDB_GROUP, WANDB_PROJECT
from vseros_b.candidates.io import load_compact_map
from vseros_b.features.builders import build_matrix
from vseros_b.artifacts import ensure_dir, save_df, save_json, log_artifact, log_metrics, download_artifact, log_table_df

import shutil

try:
    from vseros_b.exps.exp201_ranker_lgbm import DEFAULT_FEATURES as LGBM_DEFAULT_FEATURES
except Exception:  # pragma: no cover
    LGBM_DEFAULT_FEATURES: Sequence[str] = ()

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None

try:
    import wandb  # type: ignore
except Exception:  # pragma: no cover
    wandb = None

LOGGER = logging.getLogger("vseros_b.cli")

# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------

ExperimentSpec = Tuple[str, str]

EXPERIMENT_REGISTRY: Dict[str, ExperimentSpec] = {
    # recall sources
    "exp101_pop_decay": ("vseros_b.exps.exp101_pop_decay", "Exp101PopDecay"),
    "exp102_daily_trending": ("vseros_b.exps.exp102_daily_trending", "Exp102DailyTrending"),
    "exp103_covis_v1": ("vseros_b.exps.exp103_covis_v1", "Exp103CoVisV1"),
    "exp104_covis_2hop": ("vseros_b.exps.exp104_covis_2hop", "Exp104CoVis2Hop"),
    "exp105_item2vec": ("vseros_b.exps.exp105_item2vec", "Exp105Item2Vec"),
    "exp106_lightgcn_import": ("vseros_b.exp106_lightgcn_import", "Exp106LightGCNImport"),
    "exp107_ppr": ("vseros_b.exps.exp107_ppr", "Exp107PPR"),
    "exp108_recall_fusion": ("vseros_b.exps.exp108_recall_fusion", "Exp108RecallFusion"),
    # rankers
    "exp201_ranker_lgbm": ("vseros_b.exps.exp201_ranker_lgbm", "Exp201RankerLGBM"),
    "exp202_ranker_catboost": ("vseros_b.exps.exp202_ranker_catboost", "Exp202RankerCatBoost"),
    "exp203_ranker_xgb": ("vseros_b.exps.exp203_ranker_xgb", "Exp203RankerXGB"),
    "exp204_ranker_mlp": ("vseros_b.exps.exp204_ranker_mlp", "Exp204RankerMLP"),
    "exp205_ranker_stacking": ("vseros_b.exps.exp205_ranker_stacking", "Exp205RankerStacking"),
    "exp206_calibration": ("vseros_b.exps.exp206_calibration", "Exp206Calibration"),
    # blends & rules
    "exp301_blend_linear": ("vseros_b.exps.exp301_blend_linear", "Exp301BlendLinear"),
    "exp302_mmr_rerank": ("vseros_b.exps.exp302_mmr_rerank", "Exp302MMRRerank"),
    "exp303_hard_rules": ("vseros_b.exps.exp303_hard_rules", "Exp303HardRules"),
    "exp304_fallbacks": ("vseros_b.exps.exp304_fallbacks", "Exp304FallbackPipeline"),
}


DEFAULT_RECALL_STEPS: Sequence[str] = (
    "exp101_pop_decay",
    "exp102_daily_trending",
    "exp103_covis_v1",
    "exp104_covis_2hop",
    "exp105_item2vec",
    "exp106_lightgcn_import",
    "exp107_ppr",
)

DEFAULT_RANKER_STEPS: Sequence[str] = (
    "exp201_ranker_lgbm",
    "exp202_ranker_catboost",
    "exp203_ranker_xgb",
    "exp204_ranker_mlp",
    "exp205_ranker_stacking",
    "exp206_calibration",
)

DEFAULT_BLEND_STEPS: Sequence[str] = ("exp301_blend_linear",)
DEFAULT_RERANK_STEPS: Sequence[str] = ("exp302_mmr_rerank", "exp303_hard_rules")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if yaml is None:
        raise RuntimeError("PyYAML is required to read the configuration file.")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_paths(cfg: Mapping[str, Any]) -> None:
    if not cfg:
        PATHS.ensure()
        return
    for key in ("train_path", "sample_path", "artifact_dir", "interm_dir", "cand_dir", "metrics_dir", "sub_dir"):
        if key in cfg and getattr(PATHS, key, None) != Path(cfg[key]):
            setattr(PATHS, key, Path(cfg[key]))
    PATHS.ensure()


def start_wandb(enabled: bool, job_type: str, run_id: Optional[str], tags: Optional[Sequence[str]], config_dict: Mapping[str, Any]) -> Optional["wandb.sdk.wandb_run.Run"]:
    if not enabled or wandb is None:
        return None
    kwargs: Dict[str, Any] = {
        "project": WANDB_PROJECT,
        "group": WANDB_GROUP,
        "job_type": job_type,
        "config": dict(config_dict),
        "reinit": True,
    }
    if run_id:
        kwargs["name"] = run_id
    if tags:
        kwargs["tags"] = list(tags)
    try:
        return wandb.init(**kwargs)
    except Exception as exc:  # pragma: no cover - external dependency
        LOGGER.warning("Failed to init W&B: %s", exc)
        return None


def load_context(train_format: Optional[str] = None) -> Dict[str, Any]:
    ctx = data.load_and_prepare(
        data_path=PATHS.train_path,
        file_format=train_format,
        ensure_dirs=True,
    )
    sample_path = PATHS.sample_path
    if not sample_path.exists():
        raise FileNotFoundError(f"Sample submission not found at {sample_path}")
    sample_df = pd.read_csv(sample_path)
    ctx["sample_df"] = sample_df
    return ctx


def update_manifest(step: str, status: str, payload: Mapping[str, Any]) -> None:
    manifest_path = PATHS.artifact_dir / "run_manifest.json"
    manifest: Dict[str, Any]
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {"steps": []}
    manifest.setdefault("steps", [])
    manifest["steps"].append({
        "step": step,
        "status": status,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **payload,
    })
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


def instantiate_experiment(name: str, cfg_overrides: Optional[Mapping[str, Any]], verbose: bool) -> Any:
    if name not in EXPERIMENT_REGISTRY:
        raise KeyError(f"Unknown experiment: {name}")
    module_name, class_name = EXPERIMENT_REGISTRY[name]
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    cfg_instance = None
    if cfg_overrides:
        # Heuristic: search for dataclass attribute named "<ClassName>Config"
        cfg_name = None
        for attr in dir(module):
            if attr.endswith("Config") and attr.startswith(class_name[:6]):
                cfg_name = attr
                break
        if cfg_name:
            cfg_cls = getattr(module, cfg_name)
            try:
                cfg_instance = cfg_cls(**cfg_overrides)
            except Exception as exc:
                LOGGER.warning("Failed to instantiate %s config with overrides %s: %s", cfg_name, cfg_overrides, exc)
                cfg_instance = None

    try:
        if cfg_instance is not None:
            instance = cls(cfg_instance)
        else:
            instance = cls()
    except TypeError:
        # Try passing verbose flag if constructor expects it
        instance = cls(cfg_instance, verbose=verbose) if cfg_instance is not None else cls(verbose=verbose)

    if hasattr(instance, "verbose"):
        try:
            setattr(instance, "verbose", verbose)
        except Exception:
            pass
    return instance


def run_experiment(name: str, base_context: Mapping[str, Any], cfg_overrides: Optional[Mapping[str, Any]], wandb_run: Optional["wandb.sdk.wandb_run.Run"], verbose: bool) -> Tuple[Dict[str, Any], Any]:
    LOGGER.info("▶ Running %s", name)
    exp = instantiate_experiment(name, cfg_overrides, verbose=verbose)
    ctx = dict(base_context)
    if wandb_run is not None:
        ctx["wandb_run"] = wandb_run
    started = time.time()
    result: Dict[str, Any] = {"experiment": name}
    try:
        fit_out = exp.fit(ctx)
        if fit_out is not None and not isinstance(fit_out, (list, tuple, dict)):
            result["fit"] = str(type(fit_out).__name__)
    except Exception as exc:
        LOGGER.exception("Experiment %s failed during fit(): %s", name, exc)
        raise

    # evaluate()
    try:
        eval_out = exp.evaluate(ctx)
        if eval_out is not None:
            if isinstance(eval_out, pd.DataFrame):
                result["metrics"] = eval_out.to_dict(orient="records")
            else:
                result["metrics"] = eval_out
    except NotImplementedError:
        LOGGER.debug("%s does not implement evaluate()", name)
    except Exception as exc:
        LOGGER.warning("Evaluation for %s failed: %s", name, exc)

    # save()
    try:
        save_out = exp.save(ctx)  # type: ignore[assignment]
        if save_out is not None:
            if isinstance(save_out, tuple):
                result["artifacts"] = [str(x) for x in save_out if x is not None]
            else:
                result["artifacts"] = save_out
    except NotImplementedError:
        LOGGER.debug("%s does not implement save()", name)
    except Exception as exc:
        LOGGER.warning("Saving artifacts for %s failed: %s", name, exc)

    result["elapsed_sec"] = round(time.time() - started, 2)
    LOGGER.info("✔ %s finished in %.2fs", name, result["elapsed_sec"])
    return result, exp


def gather_summary(ctx: Mapping[str, Any]) -> Dict[str, Any]:
    train_df = ctx.get("train_df")
    val_df = ctx.get("val_df")
    split = ctx.get("split")
    summary = {
        "rows_total": len(ctx.get("df", [])),
        "rows_train": len(train_df) if train_df is not None else 0,
        "rows_val": len(val_df) if val_df is not None else 0,
        "users_train": int(train_df["user_id"].nunique()) if train_df is not None else 0,
        "items_train": int(train_df["item_id"].nunique()) if train_df is not None else 0,
    }
    if split is not None:
        summary.update({
            "train_range": [getattr(split, "train_start", None), getattr(split, "train_end", None)],
            "val_range": [getattr(split, "val_start", None), getattr(split, "val_end", None)],
        })
    return summary


def _latest_candidate_file(directory: Path) -> Optional[Path]:
    if not directory.exists():
        return None
    candidates = sorted(
        directory.glob("val_candidates*.parquet"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    for path in candidates:
        if path.is_file():
            return path
    return None


def _load_latest_fused_map(directory: Path) -> Optional[Dict[int, List[int]]]:
    latest = _latest_candidate_file(directory)
    if latest is None:
        return None
    try:
        return load_compact_map(latest)
    except Exception:
        LOGGER.warning("Failed to load fused candidates from %s", latest, exc_info=True)
        return None


def _calculate_coverage(cand_map: Optional[Mapping[int, Sequence[int]]], val_item_cnt: Optional[pd.Series], ks: Sequence[int] = (200, 500, 1000)) -> Dict[str, float]:
    if cand_map is None or not cand_map:
        return {}
    if val_item_cnt is None or val_item_cnt.empty:
        return {}
    total = float(val_item_cnt.sum())
    if total <= 0:
        return {}
    stats: Dict[str, float] = {}
    for k in ks:
        items: set[int] = set()
        for lst in cand_map.values():
            items.update(int(x) for x in lst[:k])
        if not items:
            stats[f"coverage@{k}"] = 0.0
            continue
        covered = float(val_item_cnt[val_item_cnt.index.isin(items)].sum())
        stats[f"coverage@{k}"] = covered / total
    return stats


def _log_directory_artifact(dir_path: Path, artifact_name: str, type_: str, metadata: Optional[Mapping[str, Any]] = None, aliases: Optional[Sequence[str]] = None):
    if not dir_path.exists():
        return None
    files = [p for p in dir_path.rglob("*") if p.is_file()]
    if not files:
        return None
    return log_artifact(files, name=artifact_name, type_=type_, metadata=dict(metadata or {}), aliases=list(aliases or []))


def _generate_submission(exp: Any, ctx: Mapping[str, Any], name: str, run_tag: str) -> Optional[Path]:
    if not hasattr(exp, "predict_submission"):
        return None
    sample_df = ctx.get("sample_df")
    if sample_df is None:
        LOGGER.warning("Sample dataframe missing in context; cannot build submission for %s", name)
        return None
    try:
        sub_df = exp.predict_submission(ctx, sample_df.copy())  # type: ignore[attr-defined]
    except NotImplementedError:
        return None
    except Exception:
        LOGGER.warning("Failed to generate submission via %s.predict_submission()", name, exc_info=True)
        return None

    if sub_df is None:
        return None

    sub_dir = ensure_dir(PATHS.sub_dir)
    sub_path = sub_dir / f"{name}_{run_tag}.csv"
    try:
        if isinstance(sub_df, pd.DataFrame):
            sub_df.to_csv(sub_path, index=False)
        else:
            pd.DataFrame(sub_df).to_csv(sub_path, index=False)
    except Exception:
        LOGGER.warning("Failed to persist submission for %s", name, exc_info=True)
        return None

    log_artifact(
        sub_path,
        name=f"submission-{name}-{run_tag}",
        type_="submission",
        metadata={"experiment": name, "run_tag": run_tag},
        aliases=["latest", run_tag],
    )
    try:
        preview = pd.read_csv(sub_path).head(10)
        log_table_df(f"{name}/submission_head", preview)
    except Exception:
        LOGGER.warning("Failed to log submission preview for %s", name, exc_info=True)
    return sub_path


def _run_and_log_exp108(ctx: Mapping[str, Any], overrides: Optional[Mapping[str, Any]], wandb_run: Optional["wandb.sdk.wandb_run.Run"], verbose: bool, cand_dir: Path) -> Dict[str, Any]:
    result, _ = run_experiment("exp108_recall_fusion", ctx, overrides, wandb_run, verbose=verbose)
    metrics_payload = result.get("metrics")
    if metrics_payload:
        if isinstance(metrics_payload, dict):
            log_metrics(metrics_payload, prefix="exp108/")
        elif isinstance(metrics_payload, list):
            for idx, row in enumerate(metrics_payload):
                if isinstance(row, Mapping):
                    log_metrics(row, prefix=f"exp108/recall@{idx}/")
    cand_map = _load_latest_fused_map(cand_dir)
    coverage_stats = _calculate_coverage(cand_map, ctx.get("val_item_cnt"))
    if coverage_stats:
        log_metrics(coverage_stats, prefix="exp108/")
        result["coverage"] = coverage_stats
    return result


def command_prepare_data(args: argparse.Namespace, cfg: Mapping[str, Any]) -> None:
    apply_paths(cfg.get("paths", {}))
    ctx = load_context(train_format=args.format)
    summary = gather_summary(ctx)
    LOGGER.info("Dataset summary: %s", summary)
    split = ctx.get("split")
    if split is not None:
        summary["split"] = {k: getattr(split, k) for k in vars(split)}

    run_tag = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    context_dir = ensure_dir(PATHS.artifact_dir / "contexts" / run_tag)

    train_path = save_df(context_dir / "train.parquet", ctx["train_df"])
    val_path = save_df(context_dir / "val.parquet", ctx["val_df"])
    sample_path = save_df(context_dir / "sample_submission.csv", ctx["sample_df"], index=False)
    meta_path = save_json(context_dir / "summary.json", summary)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    wandb_run = start_wandb(not args.no_wandb, job_type="prepare", run_id=args.run_id, tags=["prepare"], config_dict=summary)
    try:
        log_artifact(
            [train_path, val_path, sample_path, meta_path],
            name=f"context-{run_tag}",
            type_="dataset",
            metadata=summary,
            aliases=["latest", run_tag],
        )
        log_metrics({"rows_train": summary["rows_train"], "rows_val": summary["rows_val"]}, prefix="prepare/")
        if wandb_run is not None:
            wandb_run.summary.update(summary)  # type: ignore[attr-defined]
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    update_manifest("prepare-data", "completed", summary)


def command_build_candidates(args: argparse.Namespace, cfg: Mapping[str, Any]) -> None:
    apply_paths(cfg.get("paths", {}))
    ctx = load_context()
    exp_cfgs = cfg.get("experiments", {})
    wandb_run = start_wandb(not args.no_wandb, job_type="candidates", run_id=args.run_id, tags=["candidates"], config_dict={})
    results = []
    artifact_map: Mapping[str, str] = cfg.get("artifacts", {}).get("candidates", {})
    try:
        names = args.experiments or DEFAULT_RECALL_STEPS
        for name in names:
            artifact_ref = artifact_map.get(name) if isinstance(artifact_map, Mapping) else None
            if artifact_ref:
                target_dir = ensure_dir(PATHS.cand_dir / name)
                pulled = download_artifact(artifact_ref, target_dir / "_wandb_cache")
                if pulled is not None:
                    for src_file in pulled.rglob("*"):
                        if src_file.is_file():
                            dest = target_dir / src_file.name
                            try:
                                shutil.copy2(src_file, dest)
                            except Exception:
                                LOGGER.warning("Failed to copy cached artifact file %s → %s", src_file, dest, exc_info=True)
                    LOGGER.info("Using cached candidates for %s from %s", name, artifact_ref)
                    results.append({
                        "experiment": name,
                        "status": "cached",
                        "artifact_ref": artifact_ref,
                        "local_dir": str(target_dir),
                    })
                    continue
                else:
                    LOGGER.info("Artifact %s not available, rebuilding %s", artifact_ref, name)

            overrides = exp_cfgs.get(name)
            res, _ = run_experiment(name, ctx, overrides, wandb_run, verbose=args.verbose)
            metrics_payload = res.get("metrics")
            if metrics_payload:
                if isinstance(metrics_payload, dict):
                    log_metrics(metrics_payload, prefix=f"{name}/")
                elif isinstance(metrics_payload, list):
                    for idx, row in enumerate(metrics_payload):
                        if isinstance(row, Mapping):
                            log_metrics(row, prefix=f"{name}/v{idx}/")
            results.append(res)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    update_manifest("build-candidates", "completed", {"experiments": results})


def command_fuse_candidates(args: argparse.Namespace, cfg: Mapping[str, Any]) -> None:
    apply_paths(cfg.get("paths", {}))
    ctx = load_context()
    exp_cfgs = cfg.get("experiments", {})
    wandb_run = start_wandb(not args.no_wandb, job_type="fusion", run_id=args.run_id, tags=["fusion"], config_dict={})
    artifact_map: Mapping[str, str] = cfg.get("artifacts", {}).get("fusion", {})
    try:
        overrides = exp_cfgs.get("exp108_recall_fusion")
        cand_dir = ensure_dir(PATHS.cand_dir / "exp108_recall_fusion")

        artifact_ref = artifact_map.get("exp108_recall_fusion") if isinstance(artifact_map, Mapping) else None
        if artifact_ref:
            cache_dir = cand_dir / "_wandb_cache"
            pulled = download_artifact(artifact_ref, cache_dir)
            if pulled is not None:
                for src_file in pulled.rglob("*"):
                    if src_file.is_file():
                        dest = cand_dir / src_file.name
                        try:
                            shutil.copy2(src_file, dest)
                        except Exception:
                            LOGGER.warning("Failed to copy cached fusion artifact %s → %s", src_file, dest, exc_info=True)
                LOGGER.info("Using cached fusion candidates from %s", artifact_ref)
                cand_map = _load_latest_fused_map(cand_dir)
                coverage_stats: Dict[str, float] = {}
                if cand_map:
                    coverage_stats = _calculate_coverage(cand_map, ctx.get("val_item_cnt"))
                    if coverage_stats:
                        log_metrics(coverage_stats, prefix="exp108/")
                result = {
                    "experiment": "exp108_recall_fusion",
                    "status": "cached",
                    "artifact_ref": artifact_ref,
                    "coverage": coverage_stats if cand_map else {},
                }
            else:
                LOGGER.info("Artifact %s not available, rebuilding exp108", artifact_ref)
                result = _run_and_log_exp108(ctx, overrides, wandb_run, args.verbose, cand_dir)
        else:
            result = _run_and_log_exp108(ctx, overrides, wandb_run, args.verbose, cand_dir)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    update_manifest("fuse-candidates", "completed", {"experiments": [result]})


def command_build_features(args: argparse.Namespace, cfg: Mapping[str, Any]) -> None:
    apply_paths(cfg.get("paths", {}))
    ctx = load_context()

    cand_source = args.cand_source or "exp108_recall_fusion"
    cand_dir = PATHS.cand_dir / cand_source
    if not cand_dir.exists():
        raise FileNotFoundError(f"Candidate directory not found: {cand_dir}")
    pattern = "val_candidates"
    candidates: Optional[Path] = None
    for p in sorted(cand_dir.glob("val_candidates*.parquet"), key=lambda x: x.stat().st_mtime):
        candidates = p
    if candidates is None:
        raise FileNotFoundError(f"No validation candidates found in {cand_dir}")
    cand_map = load_compact_map(candidates)

    feature_names: Sequence[str]
    if not args.features or args.features == ["all"]:
        feature_names = tuple(LGBM_DEFAULT_FEATURES) if LGBM_DEFAULT_FEATURES else ()
        if not feature_names:
            raise RuntimeError("Could not resolve default feature set, pass --features explicitly.")
    else:
        feature_names = tuple(args.features)

    wandb_run = start_wandb(
        not args.no_wandb,
        job_type="features",
        run_id=args.run_id,
        tags=["features"],
        config_dict={
            "cand_source": cand_source,
            "features": list(feature_names),
            "out_tag": args.out_tag,
            "max_neg_per_pos": args.max_neg_per_pos,
            "seed": args.seed,
        },
    )

    try:
        train_mat, val_mat, norm_stats = build_matrix(
            ctx,
            cand_map=cand_map,
            feature_names=feature_names,
            out_tag=args.out_tag,
            max_neg_per_pos=args.max_neg_per_pos,
            seed=args.seed,
            save_artifacts=True,
        )
    except Exception:
        if wandb_run is not None:
            wandb_run.finish()
        raise

    payload = {
        "cand_source": cand_source,
        "rows_train": len(train_mat),
        "rows_val": len(val_mat),
        "feature_count": len([c for c in train_mat.columns if c not in ("user_id", "item_id", "label", "group_id")]),
        "out_tag": args.out_tag,
    }

    log_metrics(
        {
            "rows_train": payload["rows_train"],
            "rows_val": payload["rows_val"],
            "feature_count": payload["feature_count"],
        },
        prefix="features/",
    )

    feature_dir = PATHS.artifact_dir / "features" / args.out_tag
    artifacts_to_log = []
    for name in ("train_matrix.parquet", "val_matrix.parquet", "feature_schema.json"):
        p = feature_dir / name
        if p.exists():
            artifacts_to_log.append(p)

    if artifacts_to_log:
        log_artifact(
            artifacts_to_log,
            name=f"features-{args.out_tag}",
            type_="dataset",
            metadata={
                "cand_source": cand_source,
                "features": list(feature_names),
                "max_neg_per_pos": args.max_neg_per_pos,
            },
            aliases=["latest", args.out_tag],
        )

    if wandb_run is not None:
        try:
            wandb_run.summary.update(payload)  # type: ignore[attr-defined]
        finally:
            wandb_run.finish()

    if args.dump_stats:
        out_path = Path(args.dump_stats)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    update_manifest("build-features", "completed", payload)


def command_train_ranker(args: argparse.Namespace, cfg: Mapping[str, Any]) -> None:
    apply_paths(cfg.get("paths", {}))
    ctx = load_context()
    exp_cfgs = cfg.get("experiments", {})
    names = args.experiments or DEFAULT_RANKER_STEPS
    wandb_run = start_wandb(not args.no_wandb, job_type="ranker", run_id=args.run_id, tags=["rankers"], config_dict={})
    results = []
    run_tag = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    try:
        for name in names:
            overrides = exp_cfgs.get(name)
            res, exp_instance = run_experiment(name, ctx, overrides, wandb_run, verbose=args.verbose)

            metrics_payload = res.get("metrics")
            if metrics_payload:
                if isinstance(metrics_payload, dict):
                    log_metrics(metrics_payload, prefix=f"{name}/")
                elif isinstance(metrics_payload, list):
                    for idx, row in enumerate(metrics_payload):
                        if isinstance(row, Mapping):
                            log_metrics(row, prefix=f"{name}/v{idx}/")

            model_dir = PATHS.artifact_dir / "models" / name
            model_art = _log_directory_artifact(
                model_dir,
                artifact_name=f"{name}-model-{run_tag}",
                type_="model",
                metadata={"experiment": name, "run_tag": run_tag},
                aliases=["latest", run_tag],
            )
            if model_art is not None:
                try:
                    res["model_artifact"] = getattr(model_art, "name", f"{name}-model-{run_tag}")
                except Exception:
                    res["model_artifact"] = f"{name}-model-{run_tag}"

            submission_path = _generate_submission(exp_instance, ctx, name, run_tag)
            if submission_path:
                res.setdefault("submissions", []).append(str(submission_path))

            results.append(res)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    update_manifest("train-ranker", "completed", {"experiments": results})


def command_blend(args: argparse.Namespace, cfg: Mapping[str, Any]) -> None:
    apply_paths(cfg.get("paths", {}))
    ctx = load_context()
    exp_cfgs = cfg.get("experiments", {})
    names = args.experiments or DEFAULT_BLEND_STEPS
    wandb_run = start_wandb(not args.no_wandb, job_type="blend", run_id=args.run_id, tags=["blend"], config_dict={})
    results = []
    run_tag = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    try:
        for name in names:
            overrides = exp_cfgs.get(name)
            res, exp_instance = run_experiment(name, ctx, overrides, wandb_run, verbose=args.verbose)

            metrics_payload = res.get("metrics")
            if metrics_payload:
                if isinstance(metrics_payload, dict):
                    log_metrics(metrics_payload, prefix=f"{name}/")
                elif isinstance(metrics_payload, list):
                    for idx, row in enumerate(metrics_payload):
                        if isinstance(row, Mapping):
                            log_metrics(row, prefix=f"{name}/v{idx}/")

            model_dir = PATHS.artifact_dir / "models" / name
            art = _log_directory_artifact(
                model_dir,
                artifact_name=f"{name}-model-{run_tag}",
                type_="model",
                metadata={"experiment": name, "run_tag": run_tag},
                aliases=["latest", run_tag],
            )
            if art is not None:
                try:
                    res["model_artifact"] = getattr(art, "name", f"{name}-model-{run_tag}")
                except Exception:
                    res["model_artifact"] = f"{name}-model-{run_tag}"

            cand_dir = PATHS.cand_dir / name
            cand_map = _load_latest_fused_map(cand_dir)
            coverage_stats = _calculate_coverage(cand_map, ctx.get("val_item_cnt"))
            if coverage_stats:
                log_metrics(coverage_stats, prefix=f"{name}/")
                res["coverage"] = coverage_stats

            sub_path = _generate_submission(exp_instance, ctx, name, run_tag)
            if sub_path:
                res.setdefault("submissions", []).append(str(sub_path))

            results.append(res)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    update_manifest("blend", "completed", {"experiments": results})


def command_rerank(args: argparse.Namespace, cfg: Mapping[str, Any]) -> None:
    apply_paths(cfg.get("paths", {}))
    ctx = load_context()
    exp_cfgs = cfg.get("experiments", {})
    names = args.experiments or DEFAULT_RERANK_STEPS
    wandb_run = start_wandb(not args.no_wandb, job_type="rerank", run_id=args.run_id, tags=["rerank"], config_dict={})
    results = []
    run_tag = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    try:
        for name in names:
            overrides = exp_cfgs.get(name)
            res, exp_instance = run_experiment(name, ctx, overrides, wandb_run, verbose=args.verbose)

            metrics_payload = res.get("metrics")
            if metrics_payload:
                if isinstance(metrics_payload, dict):
                    log_metrics(metrics_payload, prefix=f"{name}/")
                elif isinstance(metrics_payload, list):
                    for idx, row in enumerate(metrics_payload):
                        if isinstance(row, Mapping):
                            log_metrics(row, prefix=f"{name}/v{idx}/")

            model_dir = PATHS.artifact_dir / "models" / name
            art = _log_directory_artifact(
                model_dir,
                artifact_name=f"{name}-model-{run_tag}",
                type_="model",
                metadata={"experiment": name, "run_tag": run_tag},
                aliases=["latest", run_tag],
            )
            if art is not None:
                try:
                    res["model_artifact"] = getattr(art, "name", f"{name}-model-{run_tag}")
                except Exception:
                    res["model_artifact"] = f"{name}-model-{run_tag}"

            cand_dir = PATHS.cand_dir / name
            cand_map = _load_latest_fused_map(cand_dir)
            coverage_stats = _calculate_coverage(cand_map, ctx.get("val_item_cnt"))
            if coverage_stats:
                log_metrics(coverage_stats, prefix=f"{name}/")
                res["coverage"] = coverage_stats

            sub_path = _generate_submission(exp_instance, ctx, name, run_tag)
            if sub_path:
                res.setdefault("submissions", []).append(str(sub_path))

            results.append(res)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    update_manifest("rerank", "completed", {"experiments": results})


def command_make_submission(args: argparse.Namespace, cfg: Mapping[str, Any]) -> None:
    apply_paths(cfg.get("paths", {}))
    ctx = load_context()
    exp_cfgs = cfg.get("experiments", {})
    target_exp = args.experiment or "exp304_fallbacks"
    wandb_run = start_wandb(not args.no_wandb, job_type="submission", run_id=args.run_id, tags=["submission"], config_dict={})
    run_tag = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    result_payload: Dict[str, Any] = {"experiment": target_exp, "status": "failed"}
    try:
        overrides = exp_cfgs.get(target_exp)
        result_payload, exp_instance = run_experiment(target_exp, ctx, overrides, wandb_run, verbose=args.verbose)

        metrics_payload = result_payload.get("metrics")
        if metrics_payload:
            if isinstance(metrics_payload, dict):
                log_metrics(metrics_payload, prefix=f"{target_exp}/")
            elif isinstance(metrics_payload, list):
                for idx, row in enumerate(metrics_payload):
                    if isinstance(row, Mapping):
                        log_metrics(row, prefix=f"{target_exp}/v{idx}/")

        cand_dir = PATHS.cand_dir / target_exp
        cand_map = _load_latest_fused_map(cand_dir)
        coverage_stats = _calculate_coverage(cand_map, ctx.get("val_item_cnt"))
        if coverage_stats:
            log_metrics(coverage_stats, prefix=f"{target_exp}/")
            result_payload["coverage"] = coverage_stats

        sub_path = _generate_submission(exp_instance, ctx, target_exp, run_tag)
        if sub_path:
            result_payload.setdefault("submissions", []).append(str(sub_path))

        if sub_path and Path(sub_path).exists():
            try:
                preview = pd.read_csv(sub_path).head(10)
                LOGGER.info("Submission preview (%s):\n%s", sub_path, preview)
            except Exception:
                LOGGER.warning("Failed to read submission preview at %s", sub_path, exc_info=True)

    finally:
        if wandb_run is not None:
            wandb_run.finish()
    update_manifest("make-submission", "completed", {"experiments": [result_payload]})


def command_report(args: argparse.Namespace, cfg: Mapping[str, Any]) -> None:
    apply_paths(cfg.get("paths", {}))
    manifest_path = PATHS.artifact_dir / "run_manifest.json"
    if not manifest_path.exists():
        LOGGER.warning("Manifest not found at %s", manifest_path)
        print("Manifest not found. Run some commands first.")
        return

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    steps = manifest.get("steps", [])
    print("=== Pipeline Report ===")
    for step in steps:
        status = step.get("status", "unknown")
        ts = step.get("ts", "")
        name = step.get("step", "unknown")
        print(f"[{status:<9}] {name} @ {ts}")
        metrics = step.get("experiments") or []
        if isinstance(metrics, list):
            for exp in metrics:
                exp_name = exp.get("experiment", "unknown")
                summary = []
                if "model_artifact" in exp:
                    summary.append(f"model={exp['model_artifact']}")
                if exp.get("coverage"):
                    cov_items = ", ".join(f"{k}:{v:.3f}" for k, v in exp["coverage"].items())
                    summary.append(f"coverage[{cov_items}]")
                if exp.get("submissions"):
                    summary.append(f"submissions={len(exp['submissions'])}")
                if summary:
                    print(f"  • {exp_name}: " + "; ".join(summary))

    checks: Dict[str, bool] = {}
    checks["features_train"] = (PATHS.artifact_dir / "features").exists()
    checks["models_dir"] = (PATHS.artifact_dir / "models").exists()
    checks["candidates_dir"] = PATHS.cand_dir.exists()
    checks["submissions_dir"] = PATHS.sub_dir.exists()
    checks["latest_submission"] = False
    latest_sub = None
    if PATHS.sub_dir.exists():
        subs = sorted(PATHS.sub_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
        if subs:
            latest_sub = subs[-1]
            checks["latest_submission"] = latest_sub.exists()

    print("\n=== Filesystem Checks ===")
    for key, ok in checks.items():
        status = "OK" if ok else "MISS"
        print(f"{key:<20}: {status}")
    if latest_sub:
        print(f"Latest submission: {latest_sub} ({latest_sub.stat().st_size/1024:.1f} KB)")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", type=Path, default=argparse.SUPPRESS, help="YAML config with overrides.")
    common.add_argument("--verbose", action="store_true", default=argparse.SUPPRESS, help="Enable debug logging.")
    common.add_argument("--run-id", type=str, default=argparse.SUPPRESS, help="Optional identifier for the W&B run and manifest.")
    common.add_argument("--no-wandb", action="store_true", default=argparse.SUPPRESS, help="Disable Weights & Biases logging.")

    parser = argparse.ArgumentParser(description="Vseros_B pipeline CLI", parents=[common])
    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare-data", help="Load and summarise the dataset.", parents=[common], add_help=False)
    p_prepare.add_argument("-h", "--help", action="help")
    p_prepare.add_argument("--format", type=str, default=None, help="Explicit file format for the train data.")
    p_prepare.add_argument("--output", type=str, default=None, help="Where to store summary JSON.")
    p_prepare.set_defaults(func=command_prepare_data)

    p_build_cand = sub.add_parser("build-candidates", help="Run recall-source experiments.", parents=[common], add_help=False)
    p_build_cand.add_argument("-h", "--help", action="help")
    p_build_cand.add_argument("--experiments", nargs="+", help="Subset of experiments to run.")
    p_build_cand.set_defaults(func=command_build_candidates)

    p_fuse = sub.add_parser("fuse-candidates", help="Merge candidate pools (exp108).", parents=[common], add_help=False)
    p_fuse.add_argument("-h", "--help", action="help")
    p_fuse.set_defaults(func=command_fuse_candidates)

    p_features = sub.add_parser("build-features", help="Materialise feature matrices.", parents=[common], add_help=False)
    p_features.add_argument("-h", "--help", action="help")
    p_features.add_argument("--cand-source", type=str, default=None, help="Candidate experiment to use (default: exp108_recall_fusion).")
    p_features.add_argument("--features", nargs="+", default=["all"], help="Feature names to include or 'all' for defaults.")
    p_features.add_argument("--out-tag", type=str, default="cli_run", help="Tag for feature artifacts.")
    p_features.add_argument("--max-neg-per-pos", type=int, default=None, help="Negative sampling cap (per positive).")
    p_features.add_argument("--seed", type=int, default=42, help="Random seed for negative sampling.")
    p_features.add_argument("--dump-stats", type=str, default=None, help="Optional path to dump feature build stats JSON.")
    p_features.set_defaults(func=command_build_features)

    p_ranks = sub.add_parser("train-ranker", help="Train ranker experiments.", parents=[common], add_help=False)
    p_ranks.add_argument("-h", "--help", action="help")
    p_ranks.add_argument("--experiments", nargs="+", help="Subset of ranker experiments to run.")
    p_ranks.set_defaults(func=command_train_ranker)

    p_blend = sub.add_parser("blend", help="Run blend / rerank / fallback experiments.", parents=[common], add_help=False)
    p_blend.add_argument("-h", "--help", action="help")
    p_blend.add_argument("--experiments", nargs="+", help="Subset of blend-stage experiments to run.")
    p_blend.set_defaults(func=command_blend)

    p_rerank = sub.add_parser("rerank", help="Apply reranking / rule-based stages.", parents=[common], add_help=False)
    p_rerank.add_argument("-h", "--help", action="help")
    p_rerank.add_argument("--experiments", nargs="+", help="Subset of rerank experiments to run.")
    p_rerank.set_defaults(func=command_rerank)

    p_sub = sub.add_parser("make-submission", help="Build the final submission (defaults to exp304).", parents=[common], add_help=False)
    p_sub.add_argument("-h", "--help", action="help")
    p_sub.add_argument("--experiment", type=str, default=None, help="Specific experiment to run for submission.")
    p_sub.set_defaults(func=command_make_submission)

    p_report = sub.add_parser("report", help="Show manifest summary and filesystem checks.", parents=[common], add_help=False)
    p_report.add_argument("-h", "--help", action="help")
    p_report.set_defaults(func=command_report)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "config"):
        args.config = None
    if not hasattr(args, "verbose"):
        args.verbose = False
    if not hasattr(args, "run_id"):
        args.run_id = None
    if not hasattr(args, "no_wandb"):
        args.no_wandb = False

    configure_logging(verbose=args.verbose)
    cfg = load_yaml(args.config)

    LOGGER.info("Starting command %s", args.command)
    args.func(args, cfg)
    LOGGER.info("Command %s finished successfully", args.command)


if __name__ == "__main__":  # pragma: no cover
    main()
