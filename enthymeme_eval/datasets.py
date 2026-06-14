from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from .config import SHUFFLE_SEED, Setting


DATA_DIR = Path("data")
GENERATED_SUBDIR = "generated"
ANLI_ORIGINAL_LIMIT = 1000


def load_rows(root: Path, setting: Setting) -> List[list]:
    """Load and shuffle the fixed data pool for one parameter-sweep setting."""

    if setting.dataset == "anli" and setting.variant == "original":
        rows = _load_anli_original(root)[:ANLI_ORIGINAL_LIMIT]
    elif setting.dataset == "arct" and setting.variant == "original":
        rows = _dedup_premise_claim(_load_arct_original(root))
    else:
        rows = _load_generated_csv(root, setting)
    return _shuffle(rows)


def _shuffle(rows: List[list]) -> List[list]:
    rows = list(rows)
    np.random.seed(SHUFFLE_SEED)
    np.random.shuffle(rows)
    return rows


def _dedup_premise_claim(rows: List[list]) -> List[list]:
    """Keep the first row for each exact Premise/Claim pair."""

    deduped = []
    seen = set()
    for row in rows:
        key = (str(row[0]), str(row[1]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def _load_anli_original(root: Path) -> List[list]:
    anli_dir = root / DATA_DIR / "anli" / "original"
    df = pd.read_json(anli_dir / "test.jsonl", lines=True)
    labels = pd.read_csv(anli_dir / "test-labels.lst", header=None)
    return [
        [row["obs1"], row["obs2"], row["hyp1"], row["hyp2"], labels.loc[idx][0]]
        for idx, row in df.iterrows()
    ]


def _load_arct_original(root: Path) -> List[list]:
    df = pd.read_csv(root / DATA_DIR / "arct" / "original" / "test.tsv", sep="\t")
    return [
        [
            row["reason"],
            row["claim"],
            row["warrant0"],
            row["warrant1"],
            int(row["correctLabelW0orW1"]) + 1,
        ]
        for _, row in df.iterrows()
    ]


def _load_generated_csv(root: Path, setting: Setting) -> List[list]:
    if setting.source is None:
        raise ValueError(f"Generated setting {setting.key} has no source CSV")

    df = pd.read_csv(root / DATA_DIR / setting.dataset / GENERATED_SUBDIR / setting.source)
    rows = []
    for _, row in df.iterrows():
        helpful = _split_steps(row["Helpful"], setting.steps)
        non_helpful = _split_steps(row["Non-Helpful"], setting.steps)
        rows.append([row["Premise"], row["Claim"]] + helpful + non_helpful + [1])
    return rows


def _split_steps(text: str, steps: int) -> List[str]:
    return [part.strip() for part in str(text).split(". ") if part.strip()][:steps]
