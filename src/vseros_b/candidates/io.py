# src/vseros_b/candidates/io.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple
import json, time, datetime as dt
import pandas as pd
import numpy as np

from vseros_b.config import PATHS, COL_USER, COL_ITEM
from vseros_b.artifacts import ensure_dir, save_df

CAND_FMT = "candidates/v1"
CAND_LONG_FMT = "candidates-long/v1"

@dataclass(frozen=True)
class CandMeta:
    format_version: str
    exp_name: str
    mode: str                # "val" | "test" | "train" (редко)
    M: int
    params: dict
    created_at: str

def _now_iso() -> str:
    return dt.datetime.utcnow().isoformat(timespec="seconds") + "Z"

def _meta_path(out_dir: Path) -> Path:
    return out_dir / "meta.json"

def _write_meta(out_dir: Path, meta: CandMeta):
    _meta_path(out_dir).write_text(json.dumps(meta.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")

def _read_meta(in_dir: Path) -> CandMeta:
    meta = json.loads(_meta_path(in_dir).read_text(encoding="utf-8"))
    return CandMeta(**meta)

# ---------- Save: compact ----------
def save_compact_map(
    exp_name: str,
    user2items: Mapping[int, List[int]],
    mode: str,
    M: Optional[int] = None,
    params: Optional[dict] = None,
    out_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """
    Сохраняет compact parquet + meta.json. Возвращает (parquet_path, meta_path).
    """
    M = int(M or max((len(v) for v in user2items.values()), default=0))
    out_dir = ensure_dir(out_dir or (PATHS.cand_dir / exp_name))
    rows = []
    for u, lst in user2items.items():
        if isinstance(lst, (list, tuple, np.ndarray)):
            s = " ".join(str(int(x)) for x in lst)
        elif isinstance(lst, str):
            s = lst
        else:
            s = ""
        rows.append((int(u), s))
    df = pd.DataFrame(rows, columns=[COL_USER, "items"])
    pq = out_dir / f"{mode}_candidates_{exp_name}_M{M}.parquet"
    save_df(pq, df, index=False)

    meta = CandMeta(
        format_version=CAND_FMT,
        exp_name=exp_name,
        mode=mode,
        M=M,
        params=dict(params or {}),
        created_at=_now_iso(),
    )
    _write_meta(out_dir, meta)
    return pq, _meta_path(out_dir)

# ---------- Save: long ----------
def save_long_rows(
    exp_name: str,
    rows: Iterable[Tuple[int, int, int, Optional[float], Optional[str]]],
    mode: str,
    params: Optional[dict] = None,
    out_dir: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """
    rows: iterable of (user_id, item_id, rank, score, source)
    """
    out_dir = ensure_dir(out_dir or (PATHS.cand_dir / exp_name))
    df = pd.DataFrame(rows, columns=[COL_USER, COL_ITEM, "rank", "score", "source"])
    df["rank"] = df["rank"].astype("int32")
    if "score" in df: df["score"] = df["score"].astype("float32")
    if "source" in df: df["source"] = df["source"].astype("string")

    pq = out_dir / f"{mode}_candidates_long_{exp_name}.parquet"
    save_df(pq, df, index=False)

    meta = CandMeta(
        format_version=CAND_LONG_FMT,
        exp_name=exp_name,
        mode=mode,
        M=int(df.groupby(COL_USER)["rank"].max().median()) if len(df) else 0,
        params=dict(params or {}),
        created_at=_now_iso(),
    )
    _write_meta(out_dir, meta)
    return pq, _meta_path(out_dir)

# ---------- Loaders ----------
def load_compact_map(path: Path) -> Dict[int, List[int]]:
    """
    Читает compact parquet (или любой parquet с колонками [user_id, items]).
    """
    df = pd.read_parquet(path)
    if "items" not in df.columns or COL_USER not in df.columns:
        raise ValueError(f"Invalid compact candidates parquet: need [{COL_USER}, 'items']")
    result: Dict[int, List[int]] = {}
    for u, s in zip(df[COL_USER].values, df["items"].values):
        if isinstance(s, str):
            arr = [int(x) for x in s.split()] if s.strip() else []
        elif isinstance(s, (list, tuple, np.ndarray)):
            arr = list(map(int, s))
        else:
            arr = []
        result[int(u)] = arr
    return result

def load_long(path: Path) -> pd.DataFrame:
    """
    Читает long parquet ([user_id, item_id, rank, score, source]).
    """
    df = pd.read_parquet(path)
    need = {COL_USER, COL_ITEM, "rank"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} must contain {need}")
    return df

# ---------- Utils ----------
def to_long_from_map(user2items: Mapping[int, List[int]], source: str, has_score: bool=False) -> Iterable[Tuple[int,int,int,Optional[float],Optional[str]]]:
    for u, lst in user2items.items():
        for r, it in enumerate(lst, 1):
            yield (int(u), int(it), int(r), None, source)

def validate_meta(dir_path: Path, expect_mode: Optional[str]=None) -> CandMeta:
    meta = _read_meta(dir_path)
    if meta.format_version not in (CAND_FMT, CAND_LONG_FMT):
        raise ValueError(f"Unknown format_version: {meta.format_version}")
    if expect_mode and meta.mode != expect_mode:
        raise ValueError(f"Mode mismatch: {meta.mode} != {expect_mode}")
    return meta
