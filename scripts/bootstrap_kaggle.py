"""Utility script to bootstrap a Kaggle or Colab runtime for Vseros_B experiments."""

from __future__ import annotations

import argparse

from vseros_b.artifacts import ensure_project_dirs
from vseros_b.config import COL_ITEM, COL_USER, PATHS, QUICK_MODE
from vseros_b.data import load_and_prepare


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap Kaggle runtime for Vseros_B experiments")
    parser.add_argument(
        "--quick",
        action="store_true",
        default=QUICK_MODE,
        help="Enable quick-mode while preparing data (reduces user/sample counts).",
    )
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Only create directories without loading the dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick:
        print("[bootstrap] quick-mode flag supplied (override via VSEROS_B_QUICK_MODE if needed).")

    print("[bootstrap] ensuring artefact directories…")
    ensure_project_dirs()
    for attr in ("project_root", "data_dir", "artifact_dir", "cand_dir", "metrics_dir"):
        print(f"[bootstrap] {attr}: {getattr(PATHS, attr)}")

    if args.skip_prepare:
        print("[bootstrap] skip requested, exiting early.")
        return

    print("[bootstrap] loading dataset (this may take a minute)…")
    context = load_and_prepare(ensure_dirs=False)
    print("[bootstrap] dataset summary:")
    users = context["train_df"][COL_USER].nunique()
    items = context["train_df"][COL_ITEM].nunique()
    print(
        "  rows_train={rows_train} rows_val={rows_val} users={users} items={items}".format(
            rows_train=len(context["train_df"]),
            rows_val=len(context["val_df"]),
            users=users,
            items=items,
        )
    )


if __name__ == "__main__":
    main()
