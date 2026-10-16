"""Build the large swap dataset from RAVEL, the LRE relations, the hand-written templates
of ``build_swap_dataset.py`` and the extra sources of ``convert_extra_sources.py``.

Every source becomes relations of the same shape: a list of prompt templates with a slot for
the subject, and ``{subject: [answers]}``. An item swaps the lens vectors of subject X for
subject X' of the same relation and template, and succeeds when the model then gives X''s
answer. Items are written in the probe-swap format (as in ``build_swap_dataset.py``).

- RAVEL (Huang et al. 2024; ``data.tgz`` of github.com/explanare/ravel):
  ``ravel_{type}_entity_attributes.json`` (entity -> attribute -> value) and
  ``ravel_{type}_attribute_to_prompts.json`` (attribute -> templates with ``%s``). Only the
  categorical attributes in ``RAVEL_ATTRIBUTES`` are used.
- LRE (Hernandez et al. 2024; ``data/`` of github.com/evandez/relations): one JSON per
  relation with ``prompt_templates`` (``{}`` slot) and ``samples`` of subject and object.
  The factual, commonsense and linguistic relations (not the bias ones).

Filters, per relation and template (counts are printed): the subject is not a probe-swap or
readout-eval word; it is one token where the template puts it (the swap needs one W_U row);
the answer has a single-token surface (its first word, if it has several); Gemma already
ranks the answer among its top ``--max-rank`` words on the clean prompt; a pair's answers
share no token.

Held-out relations. Every relation has a canonical ID (``relation_ids.py``: the Wikidata
property where there is one, so the same relation from two sources is one ID). The split is
drawn over canonical IDs, not per source: among the IDs with at least ``--min-eval-pairs``
items, a seeded ``--heldout-fraction`` goes to ``test.json`` and a further
``--dev-fraction`` to ``dev.json`` (for choosing anything, such as layer windows, without
looking at the test relations); every other relation is in ``train.json``. Dev and test keep
at most ``--max-eval-items-per-relation`` items per canonical ID.

Each item records ``relation`` (the source's name), ``relation_id`` (canonical), the clean
rank of its own answer (``clean_answer_rank``, 0 = Gemma's top word; the filter only asks
for the top ``--max-rank``) and the clean rank of the swapped-in answer
(``swap_answer_clean_rank``: 0 means Gemma already says it with no edit). Ranks are taken on
fp32 logits before the soft-cap with formatting tokens removed, and a tie counts against the
answer. Runs in the jpp_lens venv from the repo root, on a GPU.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch as t

sys.path.insert(0, "experiments/intervention_lens")
sys.path.insert(0, "scripts")

from build_swap_dataset import (  # noqa: E402
    DOMAINS,
    excluded_words,
)
from relation_ids import canonical_relation_id  # noqa: E402

from jlens.hooks import ActivationRecorder  # noqa: E402
from lens_evals.eval_utils import (  # noqa: E402
    formatting_token_ids_of,
    resolve_swap_token,
    single_token_ids,
    surface_forms,
)
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.utils import token_id_mask_Bool_V, vocab_size_of  # noqa: E402

RAVEL_ATTRIBUTES = {
    "city": ["Country", "Continent", "Language"],
    "nobel_prize_winner": ["Field", "Country of Birth"],
    "verb": ["Past Tense", "Singular"],
    "physical_object": ["Biological Category", "Color", "Size", "Texture"],
    "occupation": ["Industry", "Work Location"],
}
LRE_CATEGORIES = ("factual", "commonsense", "linguistic")
SLOT = "{x}"


@dataclass
class Relation:
    source: str
    name: str
    templates: list[str]  # with SLOT
    facts: dict[str, list[str]] = field(default_factory=dict)


def ravel_relations(ravel_dir: Path) -> list[Relation]:
    relations = []
    for entity_type, attributes in RAVEL_ATTRIBUTES.items():
        entities_path = next(ravel_dir.rglob(f"ravel_{entity_type}_entity_attributes.json"), None)
        prompts_path = next(ravel_dir.rglob(f"ravel_{entity_type}_attribute_to_prompts.json"), None)
        if entities_path is None or prompts_path is None:
            print(f"RAVEL: no files for {entity_type}", flush=True)
            continue
        entities = json.loads(entities_path.read_text())
        prompts = json.loads(prompts_path.read_text())
        for attribute in attributes:
            templates = [template.replace("%s", SLOT) for template in prompts.get(attribute, []) if template.count("%s") == 1]
            facts = {
                entity: [str(values[attribute])]
                for entity, values in entities.items()
                if attribute in values and values[attribute] not in (None, "")
            }
            relations.append(Relation("ravel", f"ravel:{entity_type}:{attribute}", templates, facts))
    return relations


def lre_relations(lre_dir: Path) -> list[Relation]:
    relations = []
    for category in LRE_CATEGORIES:
        for path in sorted((lre_dir / category).glob("*.json")):
            data = json.loads(path.read_text())
            templates = [template.replace("{}", SLOT) for template in data.get("prompt_templates", []) if template.count("{}") == 1]
            facts: dict[str, list[str]] = {}
            for sample in data.get("samples", []):
                facts.setdefault(str(sample["subject"]), []).append(str(sample["object"]))
            relations.append(Relation("lre", f"lre:{category}:{data.get('name', path.stem)}", templates, facts))
    return relations


def handwritten_relations() -> list[Relation]:
    relations = []
    for templates in DOMAINS.values():
        for name, template, facts in templates:
            relations.append(
                Relation(
                    "handwritten",
                    f"handwritten:{name}",
                    [template.replace("{x}", SLOT)],
                    {word: [answer] if isinstance(answer, str) else list(answer) for word, answer in facts.items()},
                )
            )
    return relations


def render(template: str, subject: str) -> str:
    """The prompt for one subject; a template opening on the slot gets a "Fact: " prefix so the
    subject sits after a space (its swap token is the leading-space form)."""
    if template.startswith(SLOT):
        template = "Fact: " + template
    return template.replace(SLOT, subject).rstrip()


def subject_is_one_token(tokenizer, template: str, subject: str, swap_id: int) -> bool:
    """Whether the subject is exactly one token, the swap token, where the template puts it."""
    if template.startswith(SLOT):
        template = "Fact: " + template
    prefix = template.split(SLOT)[0]
    prefix_ids = tokenizer(prefix.rstrip(" "), add_special_tokens=False).input_ids
    with_subject_ids = tokenizer(prefix + subject, add_special_tokens=False).input_ids
    return len(with_subject_ids) == len(prefix_ids) + 1 and with_subject_ids[-1] == swap_id


def usable_answers(tokenizer, answers: list[str]) -> list[str]:
    """The answers as scoreable strings: each answer with a single-token surface form, or else
    its first word when that has one ("North America" -> "North"). Items store these, so the
    training loader and the probe-swap runner score exactly the ids checked here."""
    usable = []
    for answer in answers:
        if single_token_ids(tokenizer, surface_forms(answer)):
            usable.append(answer)
        elif " " in answer.strip() and single_token_ids(tokenizer, surface_forms(answer.strip().split()[0])):
            usable.append(answer.strip().split()[0])
    return usable


def answer_token_ids(tokenizer, answers: list[str]) -> list[int]:
    return sorted({i for answer in answers for i in single_token_ids(tokenizer, surface_forms(answer))})


@t.no_grad()
def answer_word_ranks(
    model, prompts: list[str], answer_id_sets: list[list[list[int]]], formatting_mask_Bool_V: t.Tensor,
    *, batch_size: int = 128,
) -> list[list[int]]:
    """Per prompt, for each of its answer-id sets, the rank of the set's best id among the
    model's next words: the number of other words whose logit is at least as high (0 = the
    top word; a tie counts against the answer), formatting tokens removed (probe-swap's word
    rank). Batched with right-padding (the model is causal, so a row's last real position
    never sees the padding), on fp32 logits before the soft-cap."""
    final_layer = model.n_layers - 1
    unembed_VF = model._lm_head.weight.float()
    mask_Bool_V = formatting_mask_Bool_V.to(unembed_VF.device)
    pad_id = getattr(model.tokenizer, "pad_token_id", None) or 0
    ranks: list[list[int]] = []
    for start in range(0, len(prompts), batch_size):
        rows = [model.encode(prompt, max_length=512)[0].cpu() for prompt in prompts[start : start + batch_size]]
        last_Int_B = t.tensor([len(row) - 1 for row in rows], device=unembed_VF.device)
        input_ids_Int_BS = t.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=pad_id)
        with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
            model.forward(input_ids_Int_BS.to(model.input_device))
            final_BF = recorder.activations[final_layer][t.arange(len(rows), device=last_Int_B.device), last_Int_B]
        logits_BV = (model._final_norm(final_BF.float()) @ unembed_VF.T).masked_fill(mask_Bool_V, float("-inf"))
        for row_idx, id_sets in enumerate(answer_id_sets[start : start + batch_size]):
            row_ranks = []
            for answer_ids in id_sets:
                at_or_above_Bool_V = logits_BV[row_idx] >= logits_BV[row_idx, answer_ids].max()
                at_or_above_Bool_V[answer_ids] = False
                row_ranks.append(int(at_or_above_Bool_V.sum()))
            ranks.append(row_ranks)
    return ranks


def load_extra_relations(paths: list[str]) -> list[Relation]:
    """Relations from JSON files written by the source converters: a list of
    ``{"source", "name", "templates" (with {x}), "facts": {subject: [answers]}}``."""
    relations = []
    for path in paths:
        for entry in json.loads(Path(path).read_text()):
            relations.append(Relation(entry["source"], entry["name"], list(entry["templates"]), dict(entry["facts"])))
    return relations


def split_by_relation_id(
    items: list[dict], *, heldout_fraction: float, dev_fraction: float, min_eval_pairs: int,
    max_eval_items_per_relation: int, rng: random.Random,
) -> tuple[dict[str, list[dict]], dict[str, str]]:
    """``({"train", "dev", "test"} -> items, relation_id -> split)``: the module docstring's
    draw over canonical IDs."""
    items_by_id: dict[str, list[dict]] = {}
    for item in items:
        items_by_id.setdefault(item["relation_id"], []).append(item)
    eligible = sorted(relation_id for relation_id, rows in items_by_id.items() if len(rows) >= min_eval_pairs)
    rng.shuffle(eligible)
    num_test, num_dev = round(heldout_fraction * len(eligible)), round(dev_fraction * len(eligible))
    split_of_id = {relation_id: "train" for relation_id in items_by_id}
    split_of_id.update({relation_id: "test" for relation_id in eligible[:num_test]})
    split_of_id.update({relation_id: "dev" for relation_id in eligible[num_test : num_test + num_dev]})
    splits: dict[str, list[dict]] = {"train": [], "dev": [], "test": []}
    for relation_id in sorted(items_by_id):
        rows = items_by_id[relation_id]
        split = split_of_id[relation_id]
        if split != "train" and len(rows) > max_eval_items_per_relation:
            rows = rng.sample(rows, max_eval_items_per_relation)
        splits[split].extend(rows)
    for rows in splits.values():
        rng.shuffle(rows)
    return splits, split_of_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model-name", default="google/gemma-4-E2B")
    parser.add_argument("--ravel-dir", required=True)
    parser.add_argument("--lre-dir", required=True, help="The relations repo's data/ folder.")
    parser.add_argument("--extra-relations", nargs="*", default=[], help="More relation JSON files.")
    parser.add_argument("--max-rank", type=int, default=10)
    parser.add_argument("--templates-per-relation", type=int, default=3)
    parser.add_argument("--subjects-per-relation", type=int, default=150)
    parser.add_argument("--pairs-per-template", type=int, default=150)
    parser.add_argument("--heldout-fraction", type=float, default=0.2, help="Share of canonical IDs in test.")
    parser.add_argument("--dev-fraction", type=float, default=0.1, help="Share of canonical IDs in dev.")
    parser.add_argument("--min-eval-pairs", type=int, default=20, help="Fewest items for a held-out ID.")
    parser.add_argument("--max-eval-items-per-relation", type=int, default=100)
    parser.add_argument(
        "--max-heldout-items", type=int, default=None, help="Optional cap on the items of dev and of test."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    model = get_hf_model(args.hf_model_name, attn_implementation="sdpa")
    tokenizer = model.tokenizer
    vocab_size = vocab_size_of(model)
    formatting_mask_Bool_V = token_id_mask_Bool_V(formatting_token_ids_of(tokenizer, vocab_size), vocab_size)
    excluded = excluded_words()

    relations = handwritten_relations() + ravel_relations(Path(args.ravel_dir)) + lre_relations(Path(args.lre_dir))
    relations += load_extra_relations(args.extra_relations)

    items: list[dict] = []
    stats = []
    for relation in relations:
        relation_id = canonical_relation_id(relation.name)
        templates = relation.templates[:]
        rng.shuffle(templates)
        templates = templates[: args.templates_per_relation]
        subjects = list(relation.facts)
        rng.shuffle(subjects)
        counts = {"eval_word": 0, "multi_token": 0, "no_answer_token": 0, "unknown": 0, "known": 0, "pairs": 0}
        for template_idx, template in enumerate(templates):
            candidates: list[tuple[str, list[str], list[int]]] = []
            for subject in subjects[: args.subjects_per_relation]:
                if subject.lower() in excluded:
                    counts["eval_word"] += 1
                    continue
                swap_id, _ = resolve_swap_token(tokenizer, subject)
                if swap_id is None or not subject_is_one_token(tokenizer, template, subject, swap_id):
                    counts["multi_token"] += 1
                    continue
                answers = usable_answers(tokenizer, relation.facts[subject])
                answer_ids = answer_token_ids(tokenizer, answers)
                if not answer_ids:
                    counts["no_answer_token"] += 1
                    continue
                candidates.append((subject, answers, answer_ids))
            ranks = answer_word_ranks(
                model,
                [render(template, subject) for subject, _, _ in candidates],
                [[answer_ids] for _, _, answer_ids in candidates],
                formatting_mask_Bool_V,
            )
            known: dict[str, tuple[list[str], list[int], int]] = {}
            for (subject, answers, answer_ids), (rank,) in zip(candidates, ranks, strict=True):
                if rank >= args.max_rank:
                    counts["unknown"] += 1
                    continue
                known[subject] = (answers, answer_ids, rank)
            counts["known"] += len(known)
            pairs = [
                (subject, other)
                for subject in known
                for other in known
                if subject != other and not set(known[subject][1]) & set(known[other][1])
            ]
            rng.shuffle(pairs)
            pairs = pairs[: args.pairs_per_template]
            counts["pairs"] += len(pairs)
            # The swapped-in answer's clean rank on each pair's prompt: one pass per subject.
            partners: dict[str, list[str]] = {}
            for subject, other in pairs:
                partners.setdefault(subject, []).append(other)
            swap_ranks = answer_word_ranks(
                model,
                [render(template, subject) for subject in partners],
                [[known[other][1] for other in others] for others in partners.values()],
                formatting_mask_Bool_V,
            )
            swap_rank_of = {
                (subject, other): rank
                for (subject, others), row_ranks in zip(partners.items(), swap_ranks, strict=True)
                for other, rank in zip(others, row_ranks, strict=True)
            }
            for subject, other in pairs:
                items.append(
                    {
                        "name": f"{relation.name}#{template_idx}:{subject}->{other}",
                        "category": relation.name,
                        "relation": relation.name,
                        "relation_id": relation_id,
                        "prompt": render(template, subject),
                        "intermediate": subject,
                        "swap_to": other,
                        "answer": known[subject][0][0],
                        "swap_answer": known[other][0][0],
                        "answer_surfaces": known[subject][0],
                        "swap_answer_surfaces": known[other][0],
                        "clean_answer_rank": known[subject][2],
                        "swap_answer_clean_rank": swap_rank_of[(subject, other)],
                    }
                )
        stats.append({"relation": relation.name, "relation_id": relation_id, "templates": len(templates), **counts})
        print(f"{relation.name:55s} id {relation_id:26s} templates {len(templates)}  known {counts['known']:4d}  "
              f"pairs {counts['pairs']:4d}  (eval {counts['eval_word']}, multi-token {counts['multi_token']}, "
              f"no answer token {counts['no_answer_token']}, unknown {counts['unknown']})", flush=True)

    splits, split_of_id = split_by_relation_id(
        items, heldout_fraction=args.heldout_fraction, dev_fraction=args.dev_fraction,
        min_eval_pairs=args.min_eval_pairs, max_eval_items_per_relation=args.max_eval_items_per_relation, rng=rng,
    )
    if args.max_heldout_items is not None:
        for split in ("dev", "test"):
            splits[split] = splits[split][: args.max_heldout_items]
    for row in stats:
        row["split"] = split_of_id.get(row["relation_id"], "train")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, split_items in splits.items():
        (out_dir / f"{split}.json").write_text(json.dumps({"items": split_items}, indent=1))
        words = {item["intermediate"] for item in split_items} | {item["swap_to"] for item in split_items}
        clean_top1 = sum(item["clean_answer_rank"] == 0 for item in split_items)
        no_edit = sum(item["swap_answer_clean_rank"] == 0 for item in split_items)
        relation_ids = sorted({item["relation_id"] for item in split_items})
        print(f"{split}: {len(split_items)} pairs over {len({item['relation'] for item in split_items})} relations "
              f"({len(relation_ids)} canonical IDs), {len(words)} distinct swap words; own answer top-1 on the clean "
              f"prompt {clean_top1}/{len(split_items)}, swapped-in answer already top-1 with no edit "
              f"{no_edit}/{len(split_items)}", flush=True)
        if split != "train":
            print(f"  {split} IDs: {relation_ids}", flush=True)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    with t.no_grad():
        main()
