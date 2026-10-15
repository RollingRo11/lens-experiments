"""Build the large swap dataset from RAVEL, the LRE relations and the hand-written templates
of ``build_swap_dataset.py``.

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
gives the answer (word rank below ``--max-rank`` on the clean prompt); a pair's answers share
no token. Whole relations are held out: a seeded ``--heldout-fraction`` of each source's
relations, plus the held-out domains of the hand-written set. Runs in the jpp_lens venv
from the repo root, on a GPU.
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
    HELDOUT_DOMAINS,
    excluded_words,
)

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


def handwritten_relations() -> list[tuple[Relation, bool]]:
    relations = []
    for domain, templates in DOMAINS.items():
        for name, template, facts in templates:
            relation = Relation(
                "handwritten",
                f"handwritten:{name}",
                [template.replace("{x}", SLOT)],
                {word: [answer] if isinstance(answer, str) else list(answer) for word, answer in facts.items()},
            )
            relations.append((relation, domain in HELDOUT_DOMAINS))
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
    model, prompts: list[str], answer_ids_list: list[list[int]], formatting_mask_Bool_V: t.Tensor,
    *, batch_size: int = 128,
) -> list[int]:
    """Per prompt, the best rank of its answer ids in the model's next-token distribution,
    formatting tokens removed (probe-swap's word rank). Batched with right-padding (the model is
    causal, so a row's last real position never sees the padding), on fp32 logits before the
    soft-cap so that no two tokens tie."""
    final_layer = model.n_layers - 1
    unembed_VF = model._lm_head.weight.float()
    mask_Bool_V = formatting_mask_Bool_V.to(unembed_VF.device)
    pad_id = getattr(model.tokenizer, "pad_token_id", None) or 0
    ranks: list[int] = []
    for start in range(0, len(prompts), batch_size):
        rows = [model.encode(prompt, max_length=512)[0].cpu() for prompt in prompts[start : start + batch_size]]
        last_Int_B = t.tensor([len(row) - 1 for row in rows], device=unembed_VF.device)
        input_ids_Int_BS = t.nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=pad_id)
        with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
            model.forward(input_ids_Int_BS.to(model.input_device))
            final_BF = recorder.activations[final_layer][t.arange(len(rows), device=last_Int_B.device), last_Int_B]
        logits_BV = (model._final_norm(final_BF.float()) @ unembed_VF.T).masked_fill(mask_Bool_V, float("-inf"))
        for row_idx, answer_ids in enumerate(answer_ids_list[start : start + batch_size]):
            best = logits_BV[row_idx, answer_ids].max()
            ranks.append(int((logits_BV[row_idx] > best).sum()))
    return ranks


def load_extra_relations(paths: list[str]) -> list[Relation]:
    """Relations from JSON files written by the source converters: a list of
    ``{"source", "name", "templates" (with {x}), "facts": {subject: [answers]}}``."""
    relations = []
    for path in paths:
        for entry in json.loads(Path(path).read_text()):
            relations.append(Relation(entry["source"], entry["name"], list(entry["templates"]), dict(entry["facts"])))
    return relations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model-name", default="google/gemma-4-E2B")
    parser.add_argument("--ravel-dir", required=True)
    parser.add_argument("--lre-dir", required=True, help="The relations repo's data/ folder.")
    parser.add_argument(
        "--extra-relations", nargs="*", default=[], help="More relation JSON files (one held-out draw per file)."
    )
    parser.add_argument("--max-rank", type=int, default=10)
    parser.add_argument("--templates-per-relation", type=int, default=3)
    parser.add_argument("--subjects-per-relation", type=int, default=150)
    parser.add_argument("--pairs-per-template", type=int, default=150)
    parser.add_argument("--heldout-fraction", type=float, default=0.25)
    parser.add_argument("--max-heldout-items", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    model = get_hf_model(args.hf_model_name, attn_implementation="sdpa")
    tokenizer = model.tokenizer
    vocab_size = vocab_size_of(model)
    formatting_mask_Bool_V = token_id_mask_Bool_V(formatting_token_ids_of(tokenizer, vocab_size), vocab_size)
    excluded = excluded_words()

    tagged: list[tuple[Relation, bool]] = handwritten_relations()
    sources = [ravel_relations(Path(args.ravel_dir)), lre_relations(Path(args.lre_dir))]
    sources += [load_extra_relations([path]) for path in args.extra_relations]
    for relations in sources:
        heldout_names = set(rng.sample([r.name for r in relations], round(args.heldout_fraction * len(relations))))
        tagged += [(relation, relation.name in heldout_names) for relation in relations]

    splits: dict[str, list[dict]] = {"train": [], "heldout": []}
    stats = []
    for relation, heldout in tagged:
        split = "heldout" if heldout else "train"
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
                [answer_ids for _, _, answer_ids in candidates],
                formatting_mask_Bool_V,
            )
            known: dict[str, tuple[list[str], list[int]]] = {}
            for (subject, answers, answer_ids), rank in zip(candidates, ranks, strict=True):
                if rank >= args.max_rank:
                    counts["unknown"] += 1
                    continue
                known[subject] = (answers, answer_ids)
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
            for subject, other in pairs:
                splits[split].append(
                    {
                        "name": f"{relation.name}#{template_idx}:{subject}->{other}",
                        "category": relation.name,
                        "prompt": render(template, subject),
                        "intermediate": subject,
                        "swap_to": other,
                        "answer": known[subject][0][0],
                        "swap_answer": known[other][0][0],
                        "answer_surfaces": known[subject][0],
                        "swap_answer_surfaces": known[other][0],
                    }
                )
        stats.append({"relation": relation.name, "split": split, "templates": len(templates), **counts})
        print(f"{split:7s} {relation.name:55s} templates {len(templates)}  known {counts['known']:4d}  "
              f"pairs {counts['pairs']:4d}  (eval {counts['eval_word']}, multi-token {counts['multi_token']}, "
              f"no answer token {counts['no_answer_token']}, unknown {counts['unknown']})", flush=True)

    rng.shuffle(splits["heldout"])
    splits["heldout"] = splits["heldout"][: args.max_heldout_items]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, items in splits.items():
        (out_dir / f"{split}.json").write_text(json.dumps({"items": items}, indent=1))
        words = {item["intermediate"] for item in items} | {item["swap_to"] for item in items}
        print(f"{split}: {len(items)} pairs over {len({item['category'] for item in items})} relations, "
              f"{len(words)} distinct swap words", flush=True)
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    with t.no_grad():
        main()
