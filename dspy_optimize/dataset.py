"""Load labeled audio examples from a CSV and split them.

CSV format (header required):

    audio_path,frustration
    /path/to/call_001.mp3,yes
    /path/to/call_002.mp3,no
    ...

The `frustration` column must be exactly "yes" or "no" (lowercase).
Stratified split keeps the yes/no ratio identical in train and val,
which matters a lot when the dataset is small (34 examples) and
balanced (17/17).
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import List, Tuple

import dspy


def load_examples(csv_path: str) -> List[dspy.Example]:
    """Read the labels CSV into a list of `dspy.Example`s.

    Each example has `audio_path` (input) and `frustration` (gold label).
    Missing files are skipped with a warning so optimization doesn't
    crash mid-run on a typo.
    """
    examples: list[dspy.Example] = []
    missing: list[str] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "audio_path" not in reader.fieldnames \
                or "frustration" not in reader.fieldnames:
            raise ValueError(
                f"{csv_path} must have header columns: audio_path,frustration"
            )
        for row in reader:
            path = (row.get("audio_path") or "").strip()
            label = (row.get("frustration") or "").strip().lower()
            if not path or label not in ("yes", "no"):
                continue
            if not Path(path).is_file():
                missing.append(path)
                continue
            # `with_inputs("audio_path")` marks which fields are inputs vs labels.
            example = dspy.Example(
                audio_path=path, frustration=label,
            ).with_inputs("audio_path")
            examples.append(example)
    if missing:
        print(f"[dataset] WARN: {len(missing)} audio file(s) listed but not found "
              f"on disk; skipped. First few: {missing[:3]}")
    return examples


def stratified_split(
    examples: List[dspy.Example],
    val_ratio: float = 0.3,
    seed: int = 42,
) -> Tuple[List[dspy.Example], List[dspy.Example]]:
    """Stratified train/val split that preserves the yes/no ratio.

    With 34 balanced examples and val_ratio=0.3, you get roughly
    24 train (12 yes / 12 no) + 10 val (5 yes / 5 no).
    """
    yes = [e for e in examples if e.frustration == "yes"]
    no = [e for e in examples if e.frustration == "no"]
    rng = random.Random(seed)
    rng.shuffle(yes)
    rng.shuffle(no)

    def _split(items: list, ratio: float) -> tuple[list, list]:
        n_val = max(1, int(round(len(items) * ratio)))
        return items[n_val:], items[:n_val]

    train_yes, val_yes = _split(yes, val_ratio)
    train_no, val_no = _split(no, val_ratio)
    train = train_yes + train_no
    val = val_yes + val_no
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val
