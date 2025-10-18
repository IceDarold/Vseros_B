# src/vseros_b/exp106_lightgcn_import.py

from __future__ import annotations
import os
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd

from vseros_b.base_exp import BaseExp, BaseConfig
from vseros_b.config import PATHS
from vseros_b.artifacts import ensure_project_dirs


@dataclass
class Exp106Config(BaseConfig):
    """
    Конфиг импорта LightGCN/эмбеддингов из локальной папки или W&B Artifact.

    Параметры:
      source:
        - "auto"   → сначала локально (features/lgcn), при отсутствии — W&B
        - "wandb"  → всегда тянуть из Weights & Biases (нужен интернет/ключ)
        - "local"  → только локальные файлы
      wandb_art_name: имя артефакта (одинаковое имя → версии артефакта)
      wandb_alias:    алиас/версия (обычно "latest" или фиксированный тег)
      topk_eval:      K для HR/NDCG на валидации
    """
    source: str = "auto"
    wandb_art_name: str = "lgcn_embeds"
    wandb_alias: str = "latest"
    topk_eval: int = 20


class Exp106LightGCNImport(BaseExp):
    """
    Импорт готовых эмбеддингов (user,item) и маппингов (uid2idx, iid2idx),
    оценка и формирование сабмита.

    Источники:
      - Локальные файлы: features/lgcn/{lgcn.npz|user_emb.npy,item_emb.npy,uid2idx.json,item_id2idx.json}
      - W&B Artifact: <entity>/<project>/<wandb_art_name>:<alias>
        (или явно через переменную окружения TSHOP_LGCN_ART_REF="ent/proj/name:alias")

    Контракт ctx:
      - ctx["val_truth"]: Dict[int, Set[int]] — для evaluate()
    """

    def __init__(self, cfg: Optional[Exp106Config] = None, verbose: bool = True):
        super().__init__(name="exp106_lightgcn_import", verbose=verbose)
        self.cfg: Exp106Config = cfg or Exp106Config()

        ensure_project_dirs()
        self.feat_dir = Path(PATHS.feature_dir) / "lgcn"
        self.model_dir = Path(PATHS.artifact_dir) / "models" / self.name
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # контейнеры
        self.user_emb: Optional[np.ndarray] = None
        self.item_emb: Optional[np.ndarray] = None
        self.uid2idx: Dict[int, int] = {}
        self.iid2idx: Dict[int, int] = {}
        self.idx2uid: Optional[np.ndarray] = None
        self.idx2iid: Optional[np.ndarray] = None

    # -------------------------------------------------------------------------
    # ВНУТРЕННЕЕ: загрузка из локальных файлов / W&B
    # -------------------------------------------------------------------------
    def _local_available(self) -> bool:
        """Проверяем наличие локальных ресурсов в features/lgcn."""
        return (self.feat_dir / "lgcn.npz").exists() or (
            (self.feat_dir / "user_emb.npy").exists()
            and (self.feat_dir / "item_emb.npy").exists()
            and (self.feat_dir / "uid2idx.json").exists()
            and (self.feat_dir / "item_id2idx.json").exists()
        )

    def _load_local(self) -> None:
        """Грузим эмбеддинги/маппинги из features/lgcn."""
        if (self.feat_dir / "lgcn.npz").exists():
            z = np.load(self.feat_dir / "lgcn.npz", allow_pickle=True)
            self.user_emb = z["user_emb"].astype(np.float32)
            self.item_emb = z["item_emb"].astype(np.float32)

            # маппинги (вариант: пары в массиве)
            if "uid2idx" in z.files and "item_id2idx" in z.files:
                uid_pairs = z["uid2idx"]
                iid_pairs = z["item_id2idx"]
                self.uid2idx = {int(k): int(v) for k, v in uid_pairs.tolist()}
                self.iid2idx = {int(k): int(v) for k, v in iid_pairs.tolist()}
            else:
                # или json-файлы рядом
                self.uid2idx = json.loads((self.feat_dir / "uid2idx.json").read_text(encoding="utf-8"))
                self.iid2idx = json.loads((self.feat_dir / "item_id2idx.json").read_text(encoding="utf-8"))
        else:
            # «плоские» файлы
            self.user_emb = np.load(self.feat_dir / "user_emb.npy").astype(np.float32)
            self.item_emb = np.load(self.feat_dir / "item_emb.npy").astype(np.float32)
            self.uid2idx = json.loads((self.feat_dir / "uid2idx.json").read_text(encoding="utf-8"))
            self.iid2idx = json.loads((self.feat_dir / "item_id2idx.json").read_text(encoding="utf-8"))

        # сделаем обратные словари (в векторной форме)
        self.idx2uid = self._invert_mapping(self.uid2idx)
        self.idx2iid = self._invert_mapping(self.iid2idx)

        # sanity
        assert self.user_emb is not None and self.item_emb is not None, "Embeddings not loaded."
        assert self.user_emb.shape[0] == len(self.idx2uid), "User count mismatch."
        assert self.item_emb.shape[0] == len(self.idx2iid), "Item count mismatch."

    @staticmethod
    def _invert_mapping(m: Dict[int, int]) -> np.ndarray:
        """Инвертируем {ext_id: idx} → массив idx→ext_id"""
        size = max(m.values()) + 1 if m else 0
        arr = np.empty(size, dtype=np.int64)
        for ext, idx in m.items():
            arr[int(idx)] = int(ext)
        return arr

    def _download_from_wandb(self) -> None:
        """
        Скачиваем артефакт в локальный каталог (features/lgcn).
        Источники:
          - TSHOP_LGCN_ART_REF='entity/project/name:alias'
          - или собираем из окружения (WANDB_ENTITY/PROJECT) + cfg.wandb_art_name/alias
        """
        import wandb

        self.feat_dir.mkdir(parents=True, exist_ok=True)

        art_ref = os.environ.get("TSHOP_LGCN_ART_REF")
        if not art_ref:
            proj = os.environ.get("WANDB_PROJECT", "tshopping")
            ent = os.environ.get("WANDB_ENTITY")
            base = f"{proj}/{self.cfg.wandb_art_name}"
            if ent:
                base = f"{ent}/{base}"
            art_ref = f"{base}:{self.cfg.wandb_alias}"

        # если у нас уже есть ран (из BaseExp), используем его; иначе — временный
        wb_run = self.wb
        close_after = False
        if wb_run is None:
            wb_run = wandb.init(
                project=os.environ.get("WANDB_PROJECT", "tshopping"),
                entity=os.environ.get("WANDB_ENTITY") or None,
                job_type="lgcn_import",
                reinit=True,
                config={"artifact_ref": art_ref},
            )
            close_after = True

        art = wandb.use_artifact(art_ref, type="embeddings")
        dl_path = art.download(root=str(self.feat_dir))
        print(f"[exp106] downloaded W&B artifact to: {dl_path}")

        if close_after:
            try:
                wb_run.finish()
            except Exception:
                pass

    def _ensure_loaded(self) -> None:
        """
        Основная логика выбора источника и загрузки эмбеддингов.
        """
        src = (self.cfg.source or "auto").lower()

        if src == "local":
            if not self._local_available():
                raise FileNotFoundError(f"Local LGCN files not found under {self.feat_dir}")
            self._load_local()
            return

        if src == "wandb":
            self._download_from_wandb()
            self._load_local()
            return

        # auto:
        if self._local_available():
            self._load_local()
            return

        # локально нет → пробуем W&B
        try:
            self._download_from_wandb()
            self._load_local()
            return
        except Exception as e:
            raise RuntimeError(
                f"[exp106] cannot load embeddings (both local and wandb failed): {e}"
            )

    # -------------------------------------------------------------------------
    # BaseExp API
    # -------------------------------------------------------------------------
    def fit(self, ctx: Dict[str, Any]) -> "Exp106LightGCNImport":
        """
        Ничего не обучаем — только загружаем эмбеддинги/маппинги как «модель»
        и сохраняем копию в artifacts/models/exp106_lightgcn_import/.
        """
        t0 = time.time()
        self._ensure_loaded()

        # сохраним копию «модели» в model_dir (для истории)
        np.save(self.model_dir / "user_emb.npy", self.user_emb)
        np.save(self.model_dir / "item_emb.npy", self.item_emb)
        (self.model_dir / "uid2idx.json").write_text(json.dumps(self.uid2idx), encoding="utf-8")
        (self.model_dir / "item_id2idx.json").write_text(json.dumps(self.iid2idx), encoding="utf-8")

        # mini-инфо
        info = {
            "n_users": int(self.user_emb.shape[0]),
            "n_items": int(self.item_emb.shape[0]),
            "dim": int(self.user_emb.shape[1]),
            "source": self.cfg.source,
            "alias": self.cfg.wandb_alias,
        }
        pd.DataFrame([info]).to_csv(self.model_dir / "val_metrics.csv", index=False)

        if self.verbose:
            print("[exp106] embeddings loaded:", info)
            print(f"[exp106] fit done in {time.time() - t0:.1f}s")

        # W&B лог (если ран открыт BaseExp'ом)
        if self.wb is not None:
            try:
                self.wb.log({"model/loaded_dim": info["dim"]})
                # прикрепить файлы к ран-логу (на случай локального дебага)
                self.wb.save(str(self.model_dir / "user_emb.npy"))
                self.wb.save(str(self.model_dir / "item_emb.npy"))
            except Exception:
                pass

        return self

    def evaluate(self, ctx: Dict[str, Any]) -> pd.DataFrame:
        """
        Оценим HR@K и NDCG@K по валидатору.
        ctx["val_truth"]: Dict[user_id, Set[item_id]]
        """
        val_truth: Dict[int, set] = ctx.get("val_truth") or {}
        if not val_truth:
            if self.verbose:
                print("[exp106] no val_truth in ctx → skipping eval")
            res = pd.DataFrame({"k": [self.cfg.topk_eval], "HR@k": [np.nan], "NDCG@k": [np.nan]})
            res.to_csv(self.model_dir / "val_metrics.csv", index=False)
            return res

        K = int(self.cfg.topk_eval)
        uid2idx = self.uid2idx
        iid2idx = self.iid2idx
        I = self.item_emb.T  # [d, n_items]

        hits = 0
        ndcg_sum = 0.0
        total = 0

        for u_ext, gset in val_truth.items():
            if u_ext not in uid2idx or not gset:
                continue
            uidx = uid2idx[u_ext]
            u = self.user_emb[uidx:uidx + 1]  # [1, d]
            scores = (u @ I).ravel()  # [n_items]

            # top-K (без маскировки train-позитивов — для повторных покупок это корректно)
            if K < len(scores):
                idx_top = np.argpartition(scores, -K)[-K:]
                idx_top = idx_top[np.argsort(scores[idx_top])[::-1]]
            else:
                idx_top = np.argsort(scores)[::-1][:K]

            # попадание?
            hit = False
            gain = 0.0
            for pos, j in enumerate(idx_top, start=1):
                # быстрый membership по индексам
                if any(iid2idx.get(it, -1) == int(j) for it in gset):
                    hit = True
                    gain = 1.0 / np.log2(pos + 1.0)
                    break

            if hit:
                hits += 1
                ndcg_sum += gain
            total += 1

        hr = hits / max(1, total)
        ndcg = ndcg_sum / max(1, total)
        res = pd.DataFrame({"k": [K], "HR@k": [hr], "NDCG@k": [ndcg]})
        res.to_csv(self.model_dir / "val_metrics.csv", index=False)

        if self.verbose:
            print(f"[exp106] eval HR@{K}={hr:.4f} NDCG@{K}={ndcg:.4f} (users={total})")

        if self.wb is not None:
            try:
                self.wb.log({f"val/HR@{K}": hr, f"val/NDCG@{K}": ndcg})
            except Exception:
                pass

        return res

    def predict_submission(
        self,
        ctx: Dict[str, Any],
        sample_df: pd.DataFrame,
        k_top: int = 20
    ) -> pd.DataFrame:
        """
        Формируем сабмит:
          sample_df: колонки ["user_id"], на выходе — ["user_id","item_id"] (строка с item_id через пробел).
        """
        assert "user_id" in sample_df.columns, "sample_df must contain 'user_id' column."

        uid2idx = self.uid2idx
        I_t = self.item_emb.T  # [d, n_items]

        # len(iid2idx) → массив idx→ext_id (готовим при первой необходимости)
        if self.idx2iid is None:
            self.idx2iid = self._invert_mapping(self.iid2idx)

        recs: List[List[int]] = []
        miss = 0

        # Перебор пользователей (по одному — экономия памяти; при желании можно батчить)
        user_ids = sample_df["user_id"].astype("int64").to_numpy()
        for uid in user_ids:
            if uid not in uid2idx:
                recs.append([])
                miss += 1
                continue
            u = self.user_emb[uid2idx[uid]:uid2idx[uid] + 1]  # [1, d]
            scores = (u @ I_t).ravel()

            k = int(k_top)
            if k < scores.shape[0]:
                idx_top = np.argpartition(scores, -k)[-k:]
                idx_top = idx_top[np.argsort(scores[idx_top])[::-1]]
            else:
                idx_top = np.argsort(scores)[::-1][:k]

            items_ext = self.idx2iid[idx_top].tolist()
            recs.append(items_ext)

        out = sample_df.copy()
        out["item_id"] = [" ".join(map(str, arr)) if arr else "" for arr in recs]

        sub_path = Path(PATHS.submission_dir) / f"sub_{self.name}.csv"
        sub_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(sub_path, index=False)

        if self.verbose:
            print(f"[exp106] submission saved to {sub_path} (cold_users={miss})")

        if self.wb is not None:
            try:
                self.wb.save(str(sub_path))
            except Exception:
                pass

        return out
