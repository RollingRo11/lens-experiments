"""Build the swap dataset: templated prompts with one swappable word and a known answer,
"Fact: The capital of the country that contains the city of {x} is" -> Berlin for Munich.

Each template lists (word, answer) facts. An example swaps the lens vectors of a word X for
another word X' of the same template, and succeeds when the model then gives X''s answer.
Rows are written in the probe-swap item format (``prompt`` with X, ``intermediate`` X,
``swap_to`` X', ``answer``, ``swap_answer``, ``category`` = template), so the training stage
and the probe-swap runner read them unchanged.

Filters, in order (counts are printed):
1. Words that are probe-swap concepts or readout-eval intermediates are dropped, so neither
   eval scores a word whose lens vectors were trained on.
2. The word must be a single token (the swap needs one W_U row, as in probe-swap), and the
   answer must have a single-token surface form.
3. Gemma must already know the fact: the answer's word rank on the clean prompt is at most
   ``--max-rank``.
4. Pairs whose answers share a token are dropped (the swap must change the answer).

Whole domains (``HELDOUT_DOMAINS``) go to ``heldout.json``, never trained on. At most
``--pairs-per-template`` pairs per template are kept. Runs in the jpp_lens venv from the
repo root, on a GPU.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch as t

sys.path.insert(0, "scripts")

from lens_evals.causal_evals.probe_swap import PROBE_SWAP_PATH  # noqa: E402
from lens_evals.eval_utils import (  # noqa: E402
    formatting_token_ids_of,
    resolve_swap_token,
    single_token_ids,
    surface_forms,
)
from lens_evals.readout_evals.readout_eval_items import load_readout_eval_items  # noqa: E402
from workspace_lens import get_hf_model  # noqa: E402
from workspace_lens.utils import token_id_mask_Bool_V, vocab_size_of  # noqa: E402

# domain -> [(template name, template with {x}, {word: answer or [answers]})]
DOMAINS: dict[str, list[tuple[str, str, dict[str, str | list[str]]]]] = {
    "geography": [
        ("country-capital", "Fact: The capital city of {x} is", {
            "Norway": "Oslo", "Finland": "Helsinki", "Denmark": "Copenhagen", "Austria": "Vienna",
            "Portugal": "Lisbon", "Ireland": "Dublin", "Kenya": "Nairobi", "Cuba": "Havana",
            "Chile": "Santiago", "Turkey": "Ankara", "Iran": "Tehran", "Thailand": "Bangkok",
            "Vietnam": "Hanoi", "Indonesia": "Jakarta", "Netherlands": "Amsterdam", "Romania": "Bucharest",
            "Bulgaria": "Sofia", "Serbia": "Belgrade", "Nigeria": "Abuja", "Morocco": "Rabat",
            "Venezuela": "Caracas", "Iraq": "Baghdad", "Syria": "Damascus", "Lebanon": "Beirut",
            "Afghanistan": "Kabul", "Pakistan": "Islamabad", "Bangladesh": "Dhaka", "Nepal": "Kathmandu",
            "Philippines": "Manila", "Iceland": "Reykjavik", "Estonia": "Tallinn", "Latvia": "Riga",
            "Lithuania": "Vilnius", "Croatia": "Zagreb", "Ghana": "Accra", "Senegal": "Dakar",
            "Uganda": "Kampala", "Jamaica": "Kingston", "Qatar": "Doha", "Jordan": "Amman",
        }),
        ("city-country", "Fact: The city of {x} is located in the country of", {
            "Oslo": "Norway", "Lisbon": "Portugal", "Dublin": "Ireland", "Vienna": "Austria",
            "Nairobi": "Kenya", "Havana": "Cuba", "Bangkok": "Thailand", "Hanoi": "Vietnam",
            "Jakarta": "Indonesia", "Amsterdam": "Netherlands", "Helsinki": "Finland",
            "Copenhagen": "Denmark", "Lima": "Peru", "Cairo": "Egypt", "Toronto": "Canada",
            "Sydney": "Australia", "Munich": "Germany", "Milan": "Italy", "Barcelona": "Spain",
            "Osaka": "Japan", "Mumbai": "India", "Shanghai": "China", "Krakow": "Poland",
            "Istanbul": "Turkey", "Manila": "Philippines", "Zurich": "Switzerland",
            "Marseille": "France", "Hamburg": "Germany", "Naples": "Italy", "Seville": "Spain",
            "Porto": "Portugal", "Melbourne": "Australia", "Vancouver": "Canada", "Kyoto": "Japan",
            "Karachi": "Pakistan", "Lagos": "Nigeria", "Casablanca": "Morocco", "Prague": "Czech",
        }),
        ("city-capital-2hop", "Fact: The capital of the country that contains the city of {x} is", {
            "Munich": "Berlin", "Milan": "Rome", "Barcelona": "Madrid", "Osaka": "Tokyo",
            "Shanghai": "Beijing", "Krakow": "Warsaw", "Istanbul": "Ankara", "Toronto": "Ottawa",
            "Sydney": "Canberra", "Marseille": "Paris", "Lyon": "Paris", "Hamburg": "Berlin",
            "Naples": "Rome", "Seville": "Madrid", "Porto": "Lisbon", "Melbourne": "Canberra",
            "Vancouver": "Ottawa", "Kyoto": "Tokyo", "Karachi": "Islamabad", "Lagos": "Abuja",
            "Casablanca": "Rabat", "Zurich": "Bern", "Geneva": "Bern", "Bergen": "Oslo",
            "Gothenburg": "Stockholm", "Salzburg": "Vienna", "Cork": "Dublin", "Antwerp": "Brussels",
            "Rotterdam": "Amsterdam", "Alexandria": "Cairo",
        }),
        ("country-continent", "Fact: The country of {x} is located on the continent of", {
            "Kenya": "Africa", "Nigeria": "Africa", "Ghana": "Africa", "Ethiopia": "Africa",
            "Morocco": "Africa", "Senegal": "Africa", "Argentina": ["South", "America"],
            "Chile": ["South", "America"], "Venezuela": ["South", "America"], "Thailand": "Asia",
            "Vietnam": "Asia", "Nepal": "Asia", "Pakistan": "Asia", "Mongolia": "Asia",
            "Norway": "Europe", "Portugal": "Europe", "Austria": "Europe", "Ireland": "Europe",
            "Finland": "Europe", "Romania": "Europe",
        }),
    ],
    "language": [
        ("country-language", "Fact: The main language spoken in {x} is", {
            "Norway": "Norwegian", "Finland": "Finnish", "Denmark": "Danish", "Austria": "German",
            "Portugal": "Portuguese", "Netherlands": "Dutch", "Turkey": "Turkish", "Iran": "Persian",
            "Thailand": "Thai", "Vietnam": "Vietnamese", "Indonesia": "Indonesian",
            "Argentina": "Spanish", "Chile": "Spanish", "Cuba": "Spanish", "Israel": "Hebrew",
            "Romania": "Romanian", "Bulgaria": "Bulgarian", "Serbia": "Serbian", "Croatia": "Croatian",
            "Iceland": "Icelandic", "Estonia": "Estonian", "Latvia": "Latvian",
            "Lithuania": "Lithuanian", "Ukraine": "Ukrainian", "Korea": "Korean", "Pakistan": "Urdu",
            "Bangladesh": "Bengali", "Kenya": "Swahili",
        }),
        ("city-language-2hop", "Fact: The main language spoken in the country that contains the city of {x} is", {
            "Munich": "German", "Milan": "Italian", "Osaka": "Japanese", "Krakow": "Polish",
            "Istanbul": "Turkish", "Marseille": "French", "Hamburg": "German", "Naples": "Italian",
            "Seville": "Spanish", "Porto": "Portuguese", "Bergen": "Norwegian", "Gothenburg": "Swedish",
            "Salzburg": "German", "Rotterdam": "Dutch", "Kyoto": "Japanese", "Hanoi": "Vietnamese",
            "Bangkok": "Thai", "Lima": "Spanish", "Havana": "Spanish", "Tehran": "Persian",
            "Lyon": "French", "Lisbon": "Portuguese",
        }),
    ],
    "people": [
        ("person-country", "Fact: {x} was born in the country of", {
            "Beethoven": "Germany", "Picasso": "Spain", "Dante": "Italy", "Galileo": "Italy",
            "Tolstoy": "Russia", "Chopin": "Poland", "Gandhi": "India", "Confucius": "China",
            "Cervantes": "Spain", "Goethe": "Germany", "Voltaire": "France", "Copernicus": "Poland",
            "Plato": "Greece", "Aristotle": "Greece", "Socrates": "Greece", "Rembrandt": "Netherlands",
            "Nietzsche": "Germany", "Descartes": "France", "Ibsen": "Norway", "Sibelius": "Finland",
            "Dostoevsky": "Russia", "Vivaldi": "Italy", "Bach": "Germany", "Tchaikovsky": "Russia",
        }),
        ("person-capital-2hop", "Fact: The capital of the country where {x} was born is", {
            "Beethoven": "Berlin", "Picasso": "Madrid", "Dante": "Rome", "Tolstoy": "Moscow",
            "Chopin": "Warsaw", "Confucius": "Beijing", "Cervantes": "Madrid", "Goethe": "Berlin",
            "Voltaire": "Paris", "Plato": "Athens", "Ibsen": "Oslo", "Sibelius": "Helsinki",
            "Rembrandt": "Amsterdam", "Bach": "Berlin", "Vivaldi": "Rome", "Galileo": "Rome",
            "Aristotle": "Athens", "Copernicus": "Warsaw", "Dostoevsky": "Moscow", "Descartes": "Paris",
        }),
        ("person-language-2hop", "Fact: The language spoken in the country where {x} was born is", {
            "Beethoven": "German", "Picasso": "Spanish", "Dante": "Italian", "Tolstoy": "Russian",
            "Chopin": "Polish", "Confucius": "Chinese", "Cervantes": "Spanish", "Goethe": "German",
            "Voltaire": "French", "Plato": "Greek", "Ibsen": "Norwegian", "Sibelius": "Finnish",
            "Rembrandt": "Dutch", "Bach": "German", "Vivaldi": "Italian", "Galileo": "Italian",
            "Copernicus": "Polish", "Dostoevsky": "Russian", "Descartes": "French",
        }),
    ],
    "companies": [
        ("company-country", "Fact: The company {x} was founded in the country of", {
            "Toyota": "Japan", "Sony": "Japan", "Honda": "Japan", "Nintendo": "Japan",
            "Volkswagen": "Germany", "BMW": "Germany", "Siemens": "Germany", "Adidas": "Germany",
            "Nokia": "Finland", "IKEA": "Sweden", "Volvo": "Sweden", "Ferrari": "Italy", "Fiat": "Italy",
            "Nestle": "Switzerland", "Rolex": "Switzerland", "Lego": "Denmark", "Heineken": "Netherlands",
            "Philips": "Netherlands", "Renault": "France", "Peugeot": "France", "Michelin": "France",
            "Huawei": "China", "Alibaba": "China",
        }),
        ("company-capital-2hop", "Fact: The capital of the country where the company {x} was founded is", {
            "Toyota": "Tokyo", "Sony": "Tokyo", "Honda": "Tokyo", "Nintendo": "Tokyo",
            "Volkswagen": "Berlin", "Siemens": "Berlin", "Adidas": "Berlin", "Nokia": "Helsinki",
            "IKEA": "Stockholm", "Volvo": "Stockholm", "Ferrari": "Rome", "Fiat": "Rome", "Nestle": "Bern",
            "Lego": "Copenhagen", "Heineken": "Amsterdam", "Philips": "Amsterdam", "Renault": "Paris",
            "Peugeot": "Paris", "Huawei": "Beijing", "Alibaba": "Beijing",
        }),
    ],
    "food": [
        ("food-country", "Fact: The dish {x} originally comes from the country of", {
            "sushi": "Japan", "pizza": "Italy", "paella": "Spain", "taco": "Mexico", "croissant": "France",
            "kimchi": "Korea", "curry": "India", "pasta": "Italy", "burrito": "Mexico", "ramen": "Japan",
            "baguette": "France", "goulash": "Hungary", "pierogi": "Poland", "tempura": "Japan",
            "risotto": "Italy", "tapas": "Spain", "fondue": "Switzerland", "schnitzel": "Austria",
            "haggis": "Scotland", "moussaka": "Greece", "kebab": "Turkey",
        }),
        ("food-language-2hop", "Fact: The language spoken in the country where the dish {x} comes from is", {
            "sushi": "Japanese", "pizza": "Italian", "paella": "Spanish", "croissant": "French",
            "kimchi": "Korean", "ramen": "Japanese", "baguette": "French", "goulash": "Hungarian",
            "pierogi": "Polish", "tempura": "Japanese", "risotto": "Italian", "tapas": "Spanish",
            "schnitzel": "German", "moussaka": "Greek", "kebab": "Turkish", "taco": "Spanish",
            "burrito": "Spanish",
        }),
    ],
    "animals": [
        ("animal-young", "Fact: A baby {x} is called a", {
            "cat": "kitten", "dog": "puppy", "cow": "calf", "horse": "foal", "sheep": "lamb",
            "goat": "kid", "pig": "piglet", "duck": "duckling", "goose": "gosling", "frog": "tadpole",
            "kangaroo": "joey", "deer": "fawn", "lion": "cub", "hen": "chick", "seal": "pup",
        }),
        ("animal-sound", "Fact: The sound a {x} makes is called a", {
            "dog": "bark", "cat": "meow", "cow": "moo", "lion": "roar", "horse": "neigh",
            "sheep": "bleat", "pig": "oink", "duck": "quack", "bee": "buzz", "frog": "croak",
            "wolf": "howl", "crow": "caw", "donkey": "bray", "mouse": "squeak",
        }),
        ("animal-class", "Fact: In biology, the {x} belongs to the class of animals called", {
            "shark": "fish", "salmon": "fish", "tuna": "fish", "lizard": "reptiles", "crocodile": "reptiles",
            "frog": "amphibians", "toad": "amphibians", "beetle": "insects", "mosquito": "insects",
            "sparrow": "birds", "parrot": "birds", "penguin": "birds", "whale": "mammals",
            "dolphin": "mammals", "bat": "mammals", "tiger": "mammals", "salamander": "amphibians",
            "butterfly": "insects", "iguana": "reptiles",
        }),
    ],
    "words": [
        ("antonym", "Fact: The opposite of the word {x} is", {
            "hot": "cold", "tall": "short", "fast": "slow", "happy": "sad", "open": "closed",
            "early": "late", "rich": "poor", "strong": "weak", "loud": "quiet", "wet": "dry",
            "full": "empty", "hard": "soft", "young": "old", "thick": "thin", "heavy": "light",
            "deep": "shallow", "cheap": "expensive", "clean": "dirty", "push": "pull", "win": "lose",
            "buy": "sell", "north": "south", "east": "west", "love": "hate", "war": "peace",
            "first": "last", "inside": "outside", "begin": "end",
        }),
        ("plural", "Fact: The plural of the word {x} is", {
            "mouse": "mice", "child": "children", "man": "men", "woman": "women", "foot": "feet",
            "tooth": "teeth", "goose": "geese", "person": "people", "ox": "oxen", "cactus": "cacti",
            "leaf": "leaves", "knife": "knives", "wolf": "wolves", "louse": "lice", "fungus": "fungi",
        }),
        ("past-tense", "Fact: The past tense of the verb {x} is", {
            "go": "went", "eat": "ate", "run": "ran", "see": "saw", "take": "took", "give": "gave",
            "write": "wrote", "sing": "sang", "swim": "swam", "drink": "drank", "fly": "flew",
            "buy": "bought", "think": "thought", "teach": "taught", "catch": "caught", "speak": "spoke",
            "break": "broke", "choose": "chose", "drive": "drove", "ride": "rode", "begin": "began",
            "sit": "sat", "stand": "stood", "sleep": "slept", "feel": "felt", "keep": "kept",
        }),
    ],
    "science": [
        ("element-symbol", "Fact: The chemical symbol for the element {x} is", {
            "copper": "Cu", "zinc": "Zn", "lead": "Pb", "tin": "Sn", "helium": "He", "neon": "Ne",
            "hydrogen": "H", "potassium": "K", "chlorine": "Cl", "sulfur": "S", "uranium": "U",
            "platinum": "Pt", "nickel": "Ni", "magnesium": "Mg", "aluminum": "Al", "lithium": "Li",
            "argon": "Ar", "fluorine": "F", "phosphorus": "P", "titanium": "Ti", "cobalt": "Co",
        }),
        ("shape-sides", "Fact: The number of sides of a {x} is", {
            "triangle": ["three", "3"], "square": ["four", "4"], "pentagon": ["five", "5"],
            "hexagon": ["six", "6"], "heptagon": ["seven", "7"], "octagon": ["eight", "8"],
            "nonagon": ["nine", "9"], "decagon": ["ten", "10"],
        }),
    ],
    "sequences": [
        ("next-day", "Fact: The day of the week that comes right after {x} is", {
            "Monday": "Tuesday", "Tuesday": "Wednesday", "Wednesday": "Thursday",
            "Saturday": "Sunday", "Sunday": "Monday",
        }),
        ("next-month", "Fact: The month that comes right after {x} is", {
            "January": "February", "February": "March", "April": "May", "August": "September",
            "September": "October", "October": "November", "November": "December",
        }),
        ("next-number", "Fact: The number that comes right after {x} is", {
            "two": ["three", "3"], "three": ["four", "4"], "four": ["five", "5"], "five": ["six", "6"],
            "six": ["seven", "7"], "seven": ["eight", "8"], "eight": ["nine", "9"], "nine": ["ten", "10"],
        }),
    ],
    "culture": [
        ("instrument-family", "Fact: The {x} is a musical instrument in the family of", {
            "flute": "woodwind", "clarinet": "woodwind", "oboe": "woodwind", "saxophone": "woodwind",
            "cello": "string", "harp": "string", "viola": "string", "tuba": "brass", "trombone": "brass",
            "xylophone": "percussion", "cymbal": "percussion", "tambourine": "percussion",
        }),
        ("currency", "Fact: The currency used in {x} is the", {
            "Norway": "krone", "Denmark": "krone", "Turkey": "lira", "Thailand": "baht",
            "Vietnam": "dong", "Indonesia": "rupiah", "Pakistan": "rupee", "Israel": "shekel",
            "Romania": "leu", "Switzerland": "franc", "Korea": "won", "Argentina": "peso",
            "Chile": "peso", "Cuba": "peso", "Philippines": "peso", "Nigeria": "naira", "Kenya": "shilling",
        }),
        ("profession-workplace", "Fact: A {x} usually works in a", {
            "teacher": "school", "nurse": "hospital", "chef": "kitchen", "librarian": "library",
            "banker": "bank", "judge": "court", "waiter": "restaurant", "pharmacist": "pharmacy",
            "priest": "church", "mechanic": "garage", "baker": "bakery", "scientist": "laboratory",
            "farmer": "field", "miner": "mine",
        }),
    ],
}
# Never trained on: generalisation is measured on these.
HELDOUT_DOMAINS = ("food", "sequences", "culture")


def excluded_words() -> set[str]:
    """Lower-cased probe-swap concepts and readout-eval intermediates."""
    words = set()
    for item in json.loads(Path(PROBE_SWAP_PATH).read_text())["items"]:
        words |= {item["intermediate"].lower(), item["swap_to"].lower()}
    for item in load_readout_eval_items():
        words |= {intermediate.lower() for intermediate in item.intermediates}
    return words


@t.no_grad()
def answer_word_rank(model, prompt: str, answer_ids: list[int], formatting_mask_Bool_V: t.Tensor) -> int:
    """The best rank of ``answer_ids`` in the model's next-token distribution after
    ``prompt``, formatting tokens removed (probe-swap's word rank)."""
    input_ids_Int_1S = model.encode(prompt, max_length=512)
    final_layer = model.n_layers - 1
    from jlens.hooks import ActivationRecorder

    with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
        model.forward(input_ids_Int_1S)
        final_F = recorder.activations[final_layer][0, -1]
    logits_V = model.unembed(final_F.float()[None])[0].float()
    logits_V = logits_V.masked_fill(formatting_mask_Bool_V.to(logits_V.device), float("-inf"))
    best = logits_V[answer_ids].max()
    return int((logits_V > best).sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model-name", default="google/gemma-4-E2B")
    parser.add_argument("--max-rank", type=int, default=10)
    parser.add_argument("--pairs-per-template", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    model = get_hf_model(args.hf_model_name, attn_implementation="sdpa")
    tokenizer = model.tokenizer
    vocab_size = vocab_size_of(model)
    formatting_mask_Bool_V = token_id_mask_Bool_V(formatting_token_ids_of(tokenizer, vocab_size), vocab_size)
    excluded = excluded_words()
    rng = random.Random(args.seed)
    splits: dict[str, list[dict]] = {"train": [], "heldout": []}
    stats = []
    for domain, templates in DOMAINS.items():
        split = "heldout" if domain in HELDOUT_DOMAINS else "train"
        for name, template, facts in templates:
            counts = {"facts": len(facts), "eval_word": 0, "multi_token": 0, "unknown": 0}
            kept: dict[str, tuple[list[str], list[int]]] = {}
            for word, answer in facts.items():
                answers = [answer] if isinstance(answer, str) else list(answer)
                if word.lower() in excluded:
                    counts["eval_word"] += 1
                    continue
                token_id, _ = resolve_swap_token(tokenizer, word)
                answer_ids = sorted({i for a in answers for i in single_token_ids(tokenizer, surface_forms(a))})
                if token_id is None or len(tokenizer(" " + word, add_special_tokens=False).input_ids) != 1 or not answer_ids:
                    counts["multi_token"] += 1
                    continue
                if answer_word_rank(model, template.format(x=word), answer_ids, formatting_mask_Bool_V) >= args.max_rank:
                    counts["unknown"] += 1
                    continue
                kept[word] = (answers, answer_ids)
            pairs = [
                (word, other)
                for word in kept
                for other in kept
                if word != other and not set(kept[word][1]) & set(kept[other][1])
            ]
            rng.shuffle(pairs)
            pairs = pairs[: args.pairs_per_template]
            for word, other in pairs:
                splits[split].append(
                    {
                        "name": f"{name}:{word}->{other}",
                        "category": name,
                        "prompt": template.format(x=word),
                        "intermediate": word,
                        "swap_to": other,
                        "answer": kept[word][0][0],
                        "swap_answer": kept[other][0][0],
                        "answer_surfaces": kept[word][0],
                        "swap_answer_surfaces": kept[other][0],
                    }
                )
            stats.append({"domain": domain, "template": name, "split": split, "kept_words": len(kept), "pairs": len(pairs), **counts})
            print(f"{split:7s} {domain:10s} {name:22s} kept {len(kept):2d}/{counts['facts']:2d} words "
                  f"(eval {counts['eval_word']}, multi-token {counts['multi_token']}, unknown {counts['unknown']}), "
                  f"{len(pairs)} pairs", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, items in splits.items():
        (out_dir / f"{split}.json").write_text(json.dumps({"items": items}, indent=1))
        print(f"{split}: {len(items)} pairs over {len({item['category'] for item in items})} templates")
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
