"""Quick diagnostic script to verify Kaggle/Colab runtimes."""

from __future__ import annotations

import importlib
import platform

REQUIRED_MODULES = [
    "numpy",
    "pandas",
    "scipy",
    "pyarrow",
    "wandb",
    "faiss",
    "lightgbm",
    "catboost",
    "gensim",
]


def main() -> None:
    print(f"Python: {platform.python_version()} on {platform.platform()}")
    for module in REQUIRED_MODULES:
        try:
            importlib.import_module(module)
            print(f"[ ok ] {module}")
        except Exception as err:  # pragma: no cover - environment dependent
            print(f"[FAIL] {module}: {err}")


if __name__ == "__main__":
    main()
