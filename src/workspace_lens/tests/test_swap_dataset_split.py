"""CPU tests for the swap dataset's held-out split (``experiments/intervention_lens``):
relations that are the same across sources share a canonical ID, and the split is drawn over
those IDs, so no relation is in training under one name and held out under another."""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "experiments" / "intervention_lens"))

from build_swap_dataset_v2 import split_by_relation_id  # noqa: E402
from relation_ids import canonical_relation_id  # noqa: E402


def test_duplicated_relations_share_one_id() -> None:
    groups = [
        ["lre:factual:city in country", "ravel:city:Country", "counterfact:P17", "pararel:P17", "pararel:P1376"],
        ["lre:linguistic:verb past tense", "ravel:verb:Past Tense", "handwritten:past-tense"],
        ["lre:linguistic:adjective antonym", "wordnet:antonym", "handwritten:antonym"],
        ["lre:factual:person native language", "counterfact:P103", "pararel:P103"],
        ["lre:factual:company hq", "counterfact:P159", "pararel:P159"],
        ["lre:factual:country capital city", "counterfact:P36", "pararel:P36", "handwritten:country-capital"],
        ["lre:commonsense:object superclass", "wordnet:hypernym", "pararel:P279"],
        ["muse:en-fr", "muse:en-de", "muse:en-es"],
    ]
    for group in groups:
        assert len({canonical_relation_id(name) for name in group}) == 1, group
    assert canonical_relation_id("counterfact:P17") != canonical_relation_id("counterfact:P36")
    assert canonical_relation_id("lre:linguistic:word first letter") == "lre:linguistic:word first letter"


def test_split_is_drawn_over_canonical_ids() -> None:
    names = [f"source{i % 3}:rel{i // 3}" for i in range(60)]  # 20 IDs, three sources each
    items = [
        {"name": f"{name}#{j}", "relation": name, "relation_id": name.split(":")[1]}
        for name in names
        for j in range(15)
    ]
    items += [{"name": f"tiny#{j}", "relation": "tiny", "relation_id": "tiny"} for j in range(5)]
    splits, split_of_id = split_by_relation_id(
        items, heldout_fraction=0.2, dev_fraction=0.1, min_eval_pairs=20, max_eval_items_per_relation=30,
        rng=random.Random(0),
    )
    for split, rows in splits.items():
        for row in rows:
            assert split_of_id[row["relation_id"]] == split
    ids_by_split = {split: {row["relation_id"] for row in rows} for split, rows in splits.items()}
    assert len(ids_by_split["test"]) == 4 and len(ids_by_split["dev"]) == 2
    assert not ids_by_split["train"] & (ids_by_split["dev"] | ids_by_split["test"])
    assert split_of_id["tiny"] == "train"  # too few items to be held out
    for split in ("dev", "test"):
        for relation_id in ids_by_split[split]:
            assert sum(row["relation_id"] == relation_id for row in splits[split]) == 30
