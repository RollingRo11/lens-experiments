"""Turn the extra swap-data sources into relation files for ``build_swap_dataset_v2.py
--extra-relations``: a JSON list of ``{"source", "name", "templates" (with {x}), "facts":
{subject: [answers]}}``. Every template must end where the answer goes (the swap is scored on
the next token), and the subject slot is ``{x}``. Each relation also records its canonical
``relation_id`` (``relation_ids.py``: the Wikidata property, or the ID it shares with another
source's copy of the relation), which ``build_swap_dataset_v2.py`` holds out as one.

- CounterFact (ROME, ``counterfact.json``): per Wikidata relation, its rewrite prompts (``{}`` ->
  ``{x}``) and each record's subject -> true target.
- PARAREL (``data/pattern_data/graphs_json/P*.jsonl`` + ``data/trex_lms_vocab/P*.jsonl``): per
  relation, every paraphrase pattern that ends in ``[Y]`` (``[X]`` -> ``{x}``) and the T-REx
  subject -> object pairs.
- MUSE bilingual dictionaries (``en-<lang>.txt``): English word -> its translations, with a few
  "the <Language> word for {x} is" templates.
- WordNet (``nltk``): single-word lemmas only; antonyms, hypernyms, part holonyms and synonyms.

Run with ``uv run --no-project --with nltk python`` (WordNet), from the repo root.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from relation_ids import canonical_relation_id  # noqa: E402

LANGUAGES = {"fr": "French", "de": "German", "es": "Spanish"}
MUSE_TEMPLATES = ["The {lang} word for {x} is", "In {lang}, the word {x} is", "Translated into {lang}, {x} is"]
WORDNET_TEMPLATES = {
    "antonym": ["The opposite of {x} is", "The antonym of {x} is"],
    "hypernym": ["A {x} is a kind of", "Every {x} is a type of"],
    "part_holonym": ["A {x} is part of a", "The {x} is one part of a"],
    "synonym": ["Another word for {x} is", "A synonym of {x} is"],
}


def counterfact(path: Path) -> list[dict]:
    templates: dict[str, set[str]] = defaultdict(set)
    facts: dict[str, dict[str, list[str]]] = defaultdict(dict)
    for record in json.loads(path.read_text()):
        rewrite = record["requested_rewrite"]
        relation = rewrite["relation_id"]
        if rewrite["prompt"].count("{}") == 1:
            templates[relation].add(rewrite["prompt"].replace("{}", "{x}"))
        facts[relation].setdefault(rewrite["subject"], [rewrite["target_true"]["str"]])
    return [
        {"source": "counterfact", "name": f"counterfact:{relation}", "templates": sorted(templates[relation]), "facts": facts[relation]}
        for relation in sorted(facts)
        if templates[relation]
    ]


def pararel(root: Path) -> list[dict]:
    relations = []
    for pattern_path in sorted((root / "data" / "pattern_data" / "graphs_json").glob("P*.jsonl")):
        relation = pattern_path.stem
        templates = []
        for line in pattern_path.read_text().splitlines():
            if not line.strip():
                continue
            pattern = json.loads(line)["pattern"].strip().rstrip(".").strip()
            if pattern.endswith("[Y]") and pattern.count("[X]") == 1 and pattern.count("[Y]") == 1:
                templates.append(pattern[: -len("[Y]")].rstrip().replace("[X]", "{x}"))
        facts_path = root / "data" / "trex_lms_vocab" / f"{relation}.jsonl"
        if not templates or not facts_path.exists():
            continue
        facts: dict[str, list[str]] = {}
        for line in facts_path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                facts.setdefault(row["sub_label"], []).append(row["obj_label"])
        relations.append({"source": "pararel", "name": f"pararel:{relation}", "templates": sorted(set(templates)), "facts": facts})
    return relations


def muse(directory: Path) -> list[dict]:
    relations = []
    for code, language in LANGUAGES.items():
        path = directory / f"en-{code}.txt"
        if not path.exists():
            continue
        facts: dict[str, list[str]] = defaultdict(list)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0].isalpha() and parts[0] != parts[1]:
                facts[parts[0]].append(parts[1])
        templates = [template.replace("{lang}", language) for template in MUSE_TEMPLATES]
        relations.append({"source": "muse", "name": f"muse:en-{code}", "templates": templates, "facts": dict(facts)})
    return relations


def wordnet() -> list[dict]:
    from nltk.corpus import wordnet as wn

    def single(name: str) -> str | None:
        return name if name.isalpha() else None

    facts: dict[str, dict[str, list[str]]] = {kind: defaultdict(list) for kind in WORDNET_TEMPLATES}
    for synset in wn.all_synsets():
        lemmas = [lemma for lemma in synset.lemmas() if single(lemma.name())]
        for lemma in lemmas:
            word = lemma.name()
            facts["antonym"][word] += [a.name() for a in lemma.antonyms() if single(a.name())]
            facts["synonym"][word] += [other.name() for other in lemmas if other.name() != word]
            if synset.pos() == "n":
                facts["hypernym"][word] += [
                    h.lemmas()[0].name() for h in synset.hypernyms() if single(h.lemmas()[0].name())
                ]
                facts["part_holonym"][word] += [
                    h.lemmas()[0].name() for h in synset.part_holonyms() if single(h.lemmas()[0].name())
                ]
    return [
        {
            "source": "wordnet",
            "name": f"wordnet:{kind}",
            "templates": WORDNET_TEMPLATES[kind],
            "facts": {word: sorted(set(answers)) for word, answers in kind_facts.items() if answers},
        }
        for kind, kind_facts in facts.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--counterfact", default=None)
    parser.add_argument("--pararel", default=None)
    parser.add_argument("--muse-dir", default=None)
    parser.add_argument("--wordnet", action="store_true")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    built = {
        "counterfact": counterfact(Path(args.counterfact)) if args.counterfact else None,
        "pararel": pararel(Path(args.pararel)) if args.pararel else None,
        "muse": muse(Path(args.muse_dir)) if args.muse_dir else None,
        "wordnet": wordnet() if args.wordnet else None,
    }
    for source, relations in built.items():
        if relations is None:
            continue
        for relation in relations:
            relation["relation_id"] = canonical_relation_id(relation["name"])
        (out_dir / f"{source}.json").write_text(json.dumps(relations, ensure_ascii=False))
        n_facts = sum(len(relation["facts"]) for relation in relations)
        n_templates = sum(len(relation["templates"]) for relation in relations)
        print(f"{source}: {len(relations)} relations, {n_facts} subjects, {n_templates} templates", flush=True)
        for relation in relations[:3]:
            subject = next(iter(relation["facts"]))
            print(f"   {relation['name']}: {relation['templates'][0]!r}  {subject} -> {relation['facts'][subject][:3]}")


if __name__ == "__main__":
    main()
