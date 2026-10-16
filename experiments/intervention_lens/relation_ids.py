"""One canonical ID per relation of the swap datasets, so that the same relation from two
sources (or under two names) is held out as one.

The ID is the Wikidata property where the relation has one (CounterFact and PARAREL name
theirs; the others are mapped by hand below), else a short name shared by every source that
has the relation, else the relation's own name. Two relations get the same ID when one could
teach the other: the same subject type and the same kind of answer (for example LRE "city in
country", RAVEL city:Country and Wikidata P17 all map a place to its country). The mapping
errs towards merging: a merged pair can only make the held-out set stricter.

Relation names are those of ``build_swap_dataset_v2.py``: ``handwritten:<name>``,
``ravel:<entity type>:<attribute>``, ``lre:<category>:<name>``, ``counterfact:P<id>``,
``pararel:P<id>``, ``muse:en-<lang>`` and ``wordnet:<kind>``.
"""

from __future__ import annotations

# Wikidata properties that are the same relation as another one here, written as that one.
WIKIDATA_ALIASES = {
    "P1376": "P17",  # "capital of": a (capital) city -> its country, as P17 for a city
    "P279": "hypernym",  # "subclass of" = LRE object superclass = WordNet hypernym
}

CANONICAL_RELATION_IDS = {
    # Hand-written templates (build_swap_dataset.py)
    "handwritten:country-capital": "P36",
    "handwritten:city-country": "P17",
    "handwritten:country-continent": "P30",
    "handwritten:country-language": "P37",
    "handwritten:city-language-2hop": "city_language",
    "handwritten:person-country": "person_birth_country",  # "was born in the country of"
    "handwritten:company-country": "P740",  # "was founded in the country of"
    "handwritten:food-country": "P495",  # "originally comes from the country of"
    "handwritten:animal-class": "hypernym",  # "belongs to the class of animals called"
    "handwritten:antonym": "antonym",
    "handwritten:past-tense": "verb_past_tense",
    "handwritten:currency": "P38",
    "handwritten:profession-workplace": "occupation_work_location",
    # RAVEL
    "ravel:city:Country": "P17",
    "ravel:city:Continent": "P30",
    "ravel:city:Language": "city_language",
    "ravel:nobel_prize_winner:Field": "P101",
    "ravel:nobel_prize_winner:Country of Birth": "person_birth_country",
    "ravel:verb:Past Tense": "verb_past_tense",
    "ravel:physical_object:Color": "object_color",
    "ravel:occupation:Work Location": "occupation_work_location",
    # LRE
    "lre:factual:city in country": "P17",
    "lre:factual:company CEO": "P169",
    "lre:factual:company hq": "P159",
    "lre:factual:country capital city": "P36",
    "lre:factual:country currency": "P38",
    "lre:factual:country language": "P37",
    "lre:factual:food from country": "P495",
    "lre:factual:landmark in country": "P17",
    "lre:factual:landmark on continent": "P30",
    "lre:factual:person father": "P22",
    "lre:factual:person mother": "P25",
    "lre:factual:person native language": "P103",
    "lre:factual:person occupation": "P106",
    "lre:factual:person plays instrument": "P1303",
    "lre:factual:person sport position": "P413",
    "lre:factual:plays pro sport": "P641",
    "lre:factual:person university": "P69",
    "lre:factual:product by company": "P176",
    "lre:factual:star constellation name": "P59",
    "lre:commonsense:fruit inside color": "object_color",
    "lre:commonsense:fruit outside color": "object_color",
    "lre:commonsense:object superclass": "hypernym",
    "lre:commonsense:work location": "occupation_work_location",
    "lre:linguistic:adjective antonym": "antonym",
    "lre:linguistic:verb past tense": "verb_past_tense",
    # MUSE: one translation relation, whatever the language
    "muse:en-fr": "translation",
    "muse:en-de": "translation",
    "muse:en-es": "translation",
    # WordNet
    "wordnet:antonym": "antonym",
    "wordnet:hypernym": "hypernym",
    "wordnet:part_holonym": "P361",  # "is part of"
}


def canonical_relation_id(name: str) -> str:
    """The canonical ID of the relation called ``name`` (module docstring)."""
    if name in CANONICAL_RELATION_IDS:
        return CANONICAL_RELATION_IDS[name]
    source, _, rest = name.partition(":")
    if source in ("counterfact", "pararel") and rest.startswith("P"):
        return WIKIDATA_ALIASES.get(rest, rest)
    return name
