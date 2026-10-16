"""Command-line pipeline for fitting and scoring the J++ Lens.

The J++ Lens fits E expert Jacobians per layer through the LRP backward pass, each
averaged over the fit positions a per-layer k-means router assigns to it, and combines
them at inference into one linear map per layer, ``J_l = sum_e w_e J_{l,e}``, with
weights fitted on labelled readout items.

Stages, and what each writes (each also writes a JSON stamp with its arguments, the git
hash, a timestamp and the library version beside its output):

  fit-router     a per-layer router (.pt): unit norm, PCA and k-means++ over pooled
                 valid-position activations.
  fit-shard      a checkpoint directory with one contiguous prompt shard's per-expert
                 sufficient statistics; shards run on separate GPUs and merge exactly.
  merge-experts  the expert Jacobians (.pt) from the summed shard statistics; an expert
                 with fewer than --min-kept-positions positions falls back to the
                 layer's pooled Jacobian. Optionally also the pooled lens.
  pooled-lens    the experts' pooled Jacobian as a plain lens (.pt): the R-Lens of the
                 same fit.
  fit-weights    the J++ Lens (.pt, fp32): one expert-weight vector per layer, fitted
                 on the labelled items' readout residuals and saved in
                 <stem>_expert_weights.json.
  evaluate       readout ranks (.csv) and recall@k per eval with the macro row
                 (<stem>_pass_at_k.csv) for saved lenses.
  probe-swap     the probe-swap causal eval's trials (.parquet) and success tables
                 (<stem>_success.csv, <stem>_best_scale.csv).

The intervention-trained lens (one map per layer learned by Adam on edit pairs, see
:mod:`workspace_lens.fitting.intervention_fitting`) adds four stages:

  collect-pairs     a pairs directory: the clean pool (clean_pool.pt), the logit-space
                    metric (gram.pt) and one subdirectory of edit pairs per family.
  fit-intervention  a lens (.pt, fp32) fitted on one family's pairs; no model needed.
  fit-output-kl     the Tuned-Lens-style lens (.pt, fp32) fitted on the clean pool.
  build-experts     an ExpertJacobians file whose experts are whole lenses, for
                    fit-weights.

  fit-swap          a lens (.pt, fp32) whose band-layer maps are trained jointly by
                    descending on the swap loss (:mod:`workspace_lens.fitting.swap_fitting`):
                    the probe-swap edit on WikiText, rewarding the output moving from X to Y.

Each stage is a library function that takes a ``LensModel`` (so the tests run it on
tiny CPU models); :func:`main` is the only place a Hugging Face model is loaded.

The expert weights are fitted on a small share of the eval items (``--fit-fraction``,
default 20%) and every lens is scored on the rest: ``--items fit:<seed>`` and
``held-out:<seed>`` select the two splits (:func:`select_items`), taken over the scoreable
items (the ones the residual recorder keeps and the runner scores), since an item's split
depends on the whole list. fit-weights defaults to ``fit:0`` and evaluate to ``held-out:0``.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.metadata
import json
import logging
import os
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import torch as t

from jlens.protocol import LensModel
from lens_evals.causal_evals import (
    best_scale_rows,
    probe_swap_success_table,
    run_probe_swap,
)
from lens_evals.causal_evals.probe_swap import PROBE_SWAP_PATH, PROBE_SWAP_SCALES
from lens_evals.eval_utils import (
    formatting_token_ids_of,
    resolve_swap_token,
    single_token_ids,
    surface_forms,
)
from lens_evals.readout_evals.readout_eval_items import (
    DEFAULT_FIT_FRACTION,
    MODEL_CORRECTNESS_CSV,
    READOUT_EVALS_DATA_DIR,
    RECIPE_HF_MODEL_NAME,
    ReadoutEvalItem,
    load_readout_eval_items,
    load_recipe_items,
    non_semantic_token_ids,
    split_fitting_items,
)
from lens_evals.readout_evals.readout_eval_scoring import pass_at_k_by_weighting
from lens_evals.readout_evals.readout_evals import ReadoutEvalRunner
from workspace_lens import get_hf_model
from workspace_lens.config import (
    QWEN3_6_27B_RECIPE_READOUT_LAYERS,
    LensConfig,
)
from workspace_lens.fitting.condense_experts import (
    CondenseConfig,
    ExpertJacobians,
    ExpertWeighting,
    ExpertWeightingFitter,
)
from workspace_lens.fitting.expert_fitting import ExpertJacobianTrainer
from workspace_lens.fitting.intervention_fitting import (
    CleanPool,
    EditFamily,
    PairsMeta,
    TransportFit,
    collect_pairs,
    fit_intervention_lens,
    fit_output_kl,
    heldout_prompt_mask,
    intervention_lens_config,
    pairs_path,
    record_clean_pool,
    unembed_gram,
)
from workspace_lens.fitting.relp_fitting import ExpertJacobianRelPTrainer
from workspace_lens.fitting.swap_fitting import (
    SwapExample,
    SwapFit,
    draw_swap_examples,
    fit_swap_maps,
    full_length_prompts,
    swap_lens,
    swap_lens_config,
)
from workspace_lens.fitting.utils import (
    expert_checkpoint_filename,
    merge_expert_checkpoints,
    partition_name,
)
from workspace_lens.lenses.base_lens import BaseLens
from workspace_lens.lenses.jacobian_lens import JacobianLens
from workspace_lens.lenses.logit_lens import LogitLens
from workspace_lens.lrp import LRP_MODES
from workspace_lens.routing.cluster import collect_valid_position_activations
from workspace_lens.routing.router import (
    ActivationRouterCollection,
    PositionPartitionRouter,
    load_router,
)
from workspace_lens.utils import (
    check_model_matches_config,
    ensure_parent_dir,
    hf_model_name_of,
    load_lens_file,
    resolve_target_layer,
    token_id_mask_Bool_V,
    vocab_size_of,
)

logger = logging.getLogger(__name__)

### DEFAULTS

# The 7 readout layers, defined once in workspace_lens.config and shared with the
# eval runners.
RECIPE_LAYERS = list(QWEN3_6_27B_RECIPE_READOUT_LAYERS)
RECIPE_NUM_CLUSTERS = 8
RECIPE_SKIP_FIRST_N_POSITIONS = 16
RECIPE_MAX_SEQ_LEN = 128  # fit prompt truncation (router and shards)
RECIPE_MIN_KEPT_POSITIONS = 50
RECIPE_READOUT_MAX_SEQ_LEN = 512  # eval prompt truncation (runner + recorder)
DEFAULT_PASS_KS = [1, 10]
LOGIT_LENS_SPEC = "logit"  # a --lens value that builds the Logit Lens instead of reading a file
PROBE_SWAP_MAX_SEQ_LEN = 512
# merge-experts streams the shards to this device and sums there: the GPU when
# there is one.
DEFAULT_MERGE_DEVICE = "cuda" if t.cuda.is_available() else "cpu"
# The expert-weight search's defaults, for the fit-weights flags.
DEFAULT_CONDENSE_CONFIG = CondenseConfig()

# Every fit draws its prompts from the front of load_wikitext_prompts' deterministic
# stream, so a prefix of the pool is the same prompts by construction; 1000 is the
# default router pool and the smallest pool load_fit_prompts streams.
NUM_WIKITEXT_FIT_PROMPTS = 1000


### PROMPTS


def load_fit_prompts(num_prompts: int) -> list[str]:
    """The first ``num_prompts`` WikiText prompts. The pool streamed is
    ``load_wikitext_prompts(max(1000, num_prompts))``: the loader walks the
    dataset in a fixed order, so for any pool size the first 1000 prompts are
    the same, and a larger pool extends them without changing them. Imported
    lazily: the loader needs ``datasets``."""
    from jlens.examples import load_wikitext_prompts

    return load_wikitext_prompts(max(NUM_WIKITEXT_FIT_PROMPTS, num_prompts))[:num_prompts]


def load_fit_prompts_from_json(path: str, num_prompts: int) -> list[str]:
    """The first ``num_prompts`` of the JSON list of prompts at ``path`` (a saved
    prompt pool); an error when the file holds fewer."""
    prompts = json.loads(Path(path).read_text())
    if not isinstance(prompts, list) or len(prompts) < num_prompts:
        raise ValueError(
            f"{path} holds {len(prompts) if isinstance(prompts, list) else 'no'} prompts; "
            f"{num_prompts} requested"
        )
    return [str(prompt) for prompt in prompts[:num_prompts]]


def shard_prompt_slice(num_prompts: int, num_shards: int, shard_idx: int) -> slice:
    """Contiguous, near-equal slices with the remainder spread over the first
    shards."""
    base_size, remainder = divmod(num_prompts, num_shards)
    start = shard_idx * base_size + min(shard_idx, remainder)
    return slice(start, start + base_size + (1 if shard_idx < remainder else 0))


### ITEMS


def readout_runner(
    model: LensModel, *, layers: Sequence[int], max_seq_len: int
) -> ReadoutEvalRunner:
    """A lens-less runner, for the item-level methods that read out no lens
    (:meth:`~lens_evals.readout_evals.readout_evals.ReadoutEvalRunner.scoreable_items` and
    :meth:`~lens_evals.readout_evals.readout_evals.ReadoutEvalRunner.record_residuals`); with no
    lens to take them from, ``layers`` is required."""
    return ReadoutEvalRunner(model, {}, layers=layers, max_seq_len=max_seq_len)


def select_items(
    items: Sequence[ReadoutEvalItem], items_spec: str, *, fit_fraction: float
) -> list[ReadoutEvalItem]:
    """The ``--items`` selection: ``"all"`` -> ``items``; ``"fit:<seed>"`` /
    ``"held-out:<seed>"`` -> that split of
    :func:`~lens_evals.readout_evals.readout_eval_items.split_fitting_items` over ``items``
    with ``fit_fraction``. Pass the scoreable items
    (:meth:`~lens_evals.readout_evals.readout_evals.ReadoutEvalRunner.scoreable_items`): an
    item's split depends on the whole list."""
    if items_spec == "all":
        return list(items)
    split_name, _, seed = items_spec.partition(":")
    if split_name not in ("fit", "held-out") or not seed.isdigit():
        raise ValueError(
            f"items spec must be 'all', 'fit:<seed>' or 'held-out:<seed>', got {items_spec!r}"
        )
    fitting_items, held_out_items = split_fitting_items(
        items, seed=int(seed), fit_fraction=fit_fraction
    )
    return fitting_items if split_name == "fit" else held_out_items


### STAGES


def fit_router(
    model: LensModel,
    prompts: Sequence[str],
    *,
    layers: Sequence[int],
    num_clusters: int,
    projection_dim: int,
    skip_first_n_positions: int,
    max_seq_len: int,
    seed: int = 0,
) -> ActivationRouterCollection:
    """Stage 1: one PCA + k-means router per layer, fitted on the source
    positions of ``prompts`` (the positions the fit will route)."""
    activations_L_dict_PN = collect_valid_position_activations(
        model,
        prompts,
        list(layers),
        skip_first_n_positions=skip_first_n_positions,
        max_seq_len=max_seq_len,
    )
    router = ActivationRouterCollection(num_clusters, projection_dim, seed=seed)
    router.fit(activations_L_dict_PN)
    return router


def parse_router_spec(spec: str) -> tuple[str | None, str]:
    """``--router-path`` entry ``[name=]path``: the partition name (None: key the partition
    by its router's K) and the router file."""
    name, separator, path = spec.partition("=")
    return (name, path) if separator else (None, spec)


def load_partitions(specs: Sequence[str]) -> dict[int | str, ActivationRouterCollection | PositionPartitionRouter]:
    """The routers of ``--router-path`` keyed as the trainer keys partitions: by name where the
    entry names one, else by the router's K (:func:`parse_router_spec`, :func:`load_router`);
    duplicate names are an error (:func:`check_distinct_router_ks`)."""
    partitions: dict[int | str, ActivationRouterCollection | PositionPartitionRouter] = {}
    for spec in specs:
        name, path = parse_router_spec(spec)
        router = load_router(path)
        key: int | str = router.num_clusters if name is None else name
        if key in partitions:
            raise ValueError(f"two --router-path entries share the partition {key!r}")
        partitions[key] = router
    check_distinct_router_ks(list(partitions))
    return partitions


def check_distinct_router_ks(partitions: Sequence[int | str | ActivationRouterCollection]) -> None:
    """Two partitions with the same name (a router keyed by K is named ``K<K>``) would write the
    same checkpoint file; checked before the model is loaded so a bad command line fails in
    seconds. Takes partition keys, or routers (keyed by their K)."""
    names = [
        partition_name(item if isinstance(item, (int, str)) else item.num_clusters) for item in partitions
    ]
    if len(set(names)) != len(names):
        raise ValueError(
            f"partitions must have distinct names (one checkpoint each), got {names}; "
            "give routers with the same K a name with --router-path name=path"
        )


def fit_shard(
    model: LensModel,
    prompts: Sequence[str],
    *,
    routers: Sequence[ActivationRouterCollection] | dict[int | str, ActivationRouterCollection | PositionPartitionRouter],
    shard_idx: int,
    num_shards: int,
    config: LensConfig,
) -> str:
    """Stage 2: fit shard ``shard_idx`` of ``num_shards`` (a contiguous slice of
    ``prompts``, :func:`shard_prompt_slice`) and return its checkpoint
    directory, ``<artifacts>/<date>/<config.checkpoint_name>/shard<i>of<n>/``.

    ``routers`` are the frozen per-layer routers the shard buckets under: a list (each keyed
    by its K, so the Ks must be distinct) or a dict from partition key (K or name,
    :func:`load_partitions`) to router. Every router sees the same forward and
    backward passes (the trainer only changes the reduction over positions), so
    the passes are shared; only the per-partition reduction, the CPU accumulation and the
    accumulators' CPU RAM scale with the sum of the Ks. The shard directory then
    holds one ``experts_<name>_checkpoint.pt`` per partition (``experts_K<K>_checkpoint.pt``
    for one keyed by K). Two partitions with the same name would write the same file, so
    that is an error.

    The shard's config is ``config`` with the shard's ``checkpoint_name``; the
    trainer itself stamps the router geometry (K, projection dim) and the resolved
    LRP rules into the checkpoint
    (:meth:`~workspace_lens.fitting.expert_fitting.ExpertJacobianTrainer.write_checkpoint`),
    so shards fitted here merge with shards fitted through the trainers
    directly. The trainer is :class:`ExpertJacobianRelPTrainer` when
    ``config.lrp_mode`` is a RelP mode and the plain :class:`ExpertJacobianTrainer`
    otherwise. Every position enters with weight 1. The per-K checkpoint holds the
    shard's sufficient statistics.
    """
    if not 0 <= shard_idx < num_shards:
        raise ValueError(f"shard_idx must be in [0, {num_shards}), got {shard_idx}")
    if isinstance(routers, dict):
        router_collections_K_dict = dict(routers)
    else:
        check_distinct_router_ks(list(routers))
        router_collections_K_dict = {router.num_clusters: router for router in routers}
    check_distinct_router_ks(list(router_collections_K_dict))
    prompt_slice = shard_prompt_slice(len(prompts), num_shards, shard_idx)
    shard_prompts = list(prompts[prompt_slice])

    shard_config = dataclasses.replace(
        config, checkpoint_name=f"{config.checkpoint_name}/shard{shard_idx}of{num_shards}"
    )
    trainer_class = ExpertJacobianRelPTrainer if config.lrp_mode != "none" else ExpertJacobianTrainer
    trainer = trainer_class(
        shard_config,
        model,
        shard_prompts,
        router_collections_K_dict=router_collections_K_dict,
    )
    logger.info(
        "%s shard %d/%d: prompts [%d, %d) of %d, partitions %s, checkpoints in %s",
        trainer_class.__name__,
        shard_idx,
        num_shards,
        prompt_slice.start,
        prompt_slice.stop,
        len(prompts),
        {partition_name(key): router.num_clusters for key, router in router_collections_K_dict.items()},
        trainer.config.checkpoint_path,
    )
    trainer.fit()
    return trainer.config.checkpoint_path


def merge_experts(
    checkpoint_dirs: Sequence[str],
    *,
    num_clusters: int | str,
    min_kept_positions: int = RECIPE_MIN_KEPT_POSITIONS,
    checkpoint_name: str,
    device: str = DEFAULT_MERGE_DEVICE,
) -> ExpertJacobians:
    """Stage 3: sum the shards' sufficient statistics for one partition, ``num_clusters``
    (its K, or its name for a named partition; exact: every
    accumulator is a plain sum over disjoint prompt sets), then the per-expert
    means (:meth:`ExpertJacobians.from_sums`), with experts that kept fewer than
    ``min_kept_positions`` positions replaced by the layer's pooled Jacobian.
    The shards are streamed to ``device`` and summed there (the GPU when there
    is one; :meth:`ExpertJacobians.save` writes CPU tensors regardless). The
    merged config inherits shard 0's ``checkpoint_name`` (``<name>/shard0ofN``);
    ``checkpoint_name`` replaces it."""
    merged = merge_expert_checkpoints(
        [
            os.path.join(checkpoint_dir, expert_checkpoint_filename(num_clusters))
            for checkpoint_dir in checkpoint_dirs
        ],
        device=device,
    )
    experts_config = dataclasses.replace(merged.config, checkpoint_name=checkpoint_name)
    return ExpertJacobians.from_sums(
        merged.fit_sums, experts_config, min_kept_positions=min_kept_positions
    )


def fit_weights(
    model: LensModel,
    expert_jacobians: ExpertJacobians,
    items: Sequence[ReadoutEvalItem],
    *,
    layers: Sequence[int],
    excluded_token_ids: Sequence[int],
    cache_path: str | None,
    checkpoint_name: str,
    hf_model_name: str | None = None,
    max_seq_len: int = RECIPE_READOUT_MAX_SEQ_LEN,
    fitting_item_names: Sequence[str] | None = None,
    condense_config: CondenseConfig = DEFAULT_CONDENSE_CONFIG,
    collapse_fallbacks: bool = True,
) -> tuple[ExpertWeighting, JacobianLens]:
    """Stage 4: :meth:`ReadoutEvalRunner.record_residuals` for ``items`` (loaded from
    ``cache_path`` when it exists; items with no single-token candidate are
    skipped by the recorder, so the fit sees the recorded items only) ->
    :meth:`ExpertWeightingFitter.fit` on them, or on the ``fitting_item_names``
    subset (the fitting split; a name the recorder did not keep is an error) ->
    :meth:`ExpertJacobians.combine` into the inference lens. Save the lens with
    ``dtype=t.float32`` and the weighting with :meth:`ExpertWeighting.save`.
    ``condense_config`` holds the search's hyperparameters.
    ``hf_model_name`` keys the residuals cache; required for models without
    an HF name (test models). ``layers`` must be among the experts' layers,
    checked up front so a wrong layer set fails before the recording pass (the
    stage's expensive step) rather than after it. With ``collapse_fallbacks``
    (the default) every layer's copies of the pooled Jacobian are merged into one
    slot first (:meth:`ExpertJacobians.collapse_fallbacks`); the per-layer cluster
    counts and fallback share are logged either way."""
    unknown_layers = sorted(set(layers) - set(expert_jacobians.layers))
    if unknown_layers:
        raise ValueError(
            f"layers {unknown_layers} are not among the experts' layers "
            f"{expert_jacobians.layers}"
        )
    log_expert_summary(expert_jacobians, layers)
    if collapse_fallbacks:
        expert_jacobians = expert_jacobians.collapse_fallbacks()
        logger.info(
            "fallback copies collapsed: expert slots per layer %s",
            {layer: expert_jacobians.experts_L_dict_EFN[layer].shape[0] for layer in layers},
        )
    runner = readout_runner(model, layers=layers, max_seq_len=max_seq_len)
    residuals = runner.record_residuals(
        items, cache_path=cache_path, hf_model_name=hf_model_name
    )
    fitter = ExpertWeightingFitter(
        model, expert_jacobians, residuals, excluded_token_ids=excluded_token_ids
    )
    weighting = fitter.fit(
        layers, item_names=fitting_item_names, condense_config=condense_config
    )
    lens = expert_jacobians.combine(weighting.weights_L_dict_E, checkpoint_name=checkpoint_name)
    return weighting, lens


def log_expert_summary(expert_jacobians: ExpertJacobians, layers: Sequence[int]) -> None:
    """Log, per layer, the expert slots, the populated clusters, those with their own
    Jacobian, the fallback slots and the share of fit positions in fallback clusters
    (:meth:`ExpertJacobians.layer_summary`), then the slot counts."""
    summary = expert_jacobians.layer_summary()
    for layer in layers:
        row = summary[layer]
        logger.info(
            "layer %d: %d slots, %d populated clusters, %d with their own Jacobian, %d fallback "
            "slots holding %.1f%% of fit positions; counts %s",
            layer,
            row["slots"],
            row["populated"],
            row["own_jacobian"],
            row["fallback_slots"],
            100 * float(row["fallback_position_fraction"]),
            row["counts"],
        )


def evaluate(
    model: LensModel,
    lenses: dict[str, BaseLens],
    items: Sequence[ReadoutEvalItem],
    *,
    layers: Sequence[int],
    excluded_token_ids: Sequence[int],
    ks: Sequence[int] = tuple(DEFAULT_PASS_KS),
    max_seq_len: int = RECIPE_READOUT_MAX_SEQ_LEN,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 5: ``(ranks_df, pass_df)`` — :meth:`ReadoutEvalRunner.run`'s rank rows
    (``eval, lens, item, intermediate, layer, rank, n_candidates``) with
    ``excluded_token_ids`` forced out of every ranking (pass ``[]`` to rank the
    full vocabulary), and :func:`pass_at_k_by_weighting` over them: recall@k per
    eval and the ``macro`` row, per lens, k and weighting (``item`` and
    ``pair``). Scores only; no fitting. Every lens must have been fitted on
    ``model`` (:func:`~workspace_lens.utils.check_model_matches_config`)."""
    for lens in lenses.values():
        check_model_matches_config(model, lens.config)
    runner = ReadoutEvalRunner(
        model,
        lenses,
        layers=layers,
        max_seq_len=max_seq_len,
        excluded_token_ids=excluded_token_ids,
    )
    ranks_df = runner.run(items)
    return ranks_df, pass_at_k_by_weighting(ranks_df, ks)


def probe_swap(
    model: LensModel,
    lenses: dict[str, BaseLens],
    *,
    layers: Sequence[int],
    scales: Sequence[float] = PROBE_SWAP_SCALES,
    max_seq_len: int = PROBE_SWAP_MAX_SEQ_LEN,
    data_path: str = PROBE_SWAP_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stage 6: the probe-swap eval, ``(trials_df, success_table)``: the clamp-swap trials
    of every lens at the readout ``layers``
    (:func:`~lens_evals.causal_evals.run_probe_swap`), and the top-1 and top-5 counts per lens
    and scale (:func:`~lens_evals.causal_evals.probe_swap_success_table`). Every lens must have
    been fitted on ``model``."""
    for lens in lenses.values():
        check_model_matches_config(model, lens.config)
    trials_df, dropped_concepts = run_probe_swap(
        model, lenses, layers=layers, scales=scales, max_seq_len=max_seq_len, data_path=data_path
    )
    if dropped_concepts:
        logger.info("probe swap dropped concepts with no single-token form: %s", dropped_concepts)
    return trials_df, probe_swap_success_table(trials_df)


### THE INTERVENTION-TRAINED LENS

CLEAN_POOL_FILENAME = "clean_pool.pt"
GRAM_FILENAME = "gram.pt"
IDENTITY_INIT = "identity"  # an --init value: start every map at the identity (the Logit Lens)


def collect_intervention_pairs(
    model: LensModel,
    prompts: Sequence[str],
    *,
    families: Sequence[EditFamily],
    layers: Sequence[int],
    hf_model_name: str,
    out_dir: str,
    target_offset: int = -1,
    pairs_per_prompt: int = 32,
    rows_per_pass: int = 128,
    max_seq_len: int = RECIPE_MAX_SEQ_LEN,
    skip_first_n_positions: int = RECIPE_SKIP_FIRST_N_POSITIONS,
    seed: int = 0,
) -> dict[str, object]:
    """Stage: the clean pool (recorded once and reused by later calls on the same
    ``out_dir``, which must then have the same prompts and settings), the logit-space
    metric, and each family's pairs under ``<out_dir>/<family.label>/`` with its
    ``meta.json``. Returns the collection diagnostics."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    target_layer = resolve_target_layer(model.n_layers, target_offset)
    pool_meta = PairsMeta(
        hf_model_name=hf_model_name,
        d_model=model.d_model,
        layers=list(layers),
        relative_end_transport_layer=target_offset,
        max_seq_len=max_seq_len,
        skip_first_n_positions=skip_first_n_positions,
        num_prompts=len(prompts),
        family="clean",
        pairs_per_prompt=0,
        seed=seed,
    )
    pool_path = out_path / CLEAN_POOL_FILENAME
    if pool_path.exists():
        existing_meta = PairsMeta.load(out_path)
        if dataclasses.replace(existing_meta, seed=seed) != pool_meta:
            raise ValueError(f"{pool_path} was recorded with {existing_meta}, not {pool_meta}")
        pool = CleanPool.load(pool_path)
        logger.info("clean pool loaded from %s (%d rows)", pool_path, len(pool.prompt_idx_Int_R))
    else:
        pool_start_time = datetime.now(UTC)
        pool = record_clean_pool(
            model,
            prompts,
            layers=layers,
            target_layer=target_layer,
            max_seq_len=max_seq_len,
            skip_first_n_positions=skip_first_n_positions,
        )
        pool.save(pool_path)
        pool_meta.save(out_path)
        logger.info(
            "clean pool: %d rows in %.0fs, saved to %s",
            len(pool.prompt_idx_Int_R),
            (datetime.now(UTC) - pool_start_time).total_seconds(),
            pool_path,
        )
    gram_path = out_path / GRAM_FILENAME
    if not gram_path.exists() and hasattr(model, "_lm_head"):
        t.save(unembed_gram(model), gram_path)

    pairs_by_family, stats = collect_pairs(
        model,
        prompts,
        pool,
        families=families,
        layers=layers,
        target_layer=target_layer,
        pairs_per_prompt=pairs_per_prompt,
        rows_per_pass=rows_per_pass,
        max_seq_len=max_seq_len,
        skip_first_n_positions=skip_first_n_positions,
        seed=seed,
    )
    for family in families:
        family_dir = out_path / family.label
        family_dir.mkdir(exist_ok=True)
        for layer, layer_pairs in pairs_by_family[family.label].items():
            layer_pairs.save(pairs_path(family_dir, layer))
        dataclasses.replace(
            pool_meta, family=family.label, pairs_per_prompt=pairs_per_prompt
        ).save(family_dir)
    seconds = stats.seconds_per_prompt
    return {
        "families": [family.label for family in families],
        "num_prompts_with_pairs": len(seconds),
        "mean_seconds_per_prompt": sum(seconds) / len(seconds),
        "max_leak": stats.max_leak,
        "max_abs_y": stats.max_abs_y,
    }


def _init_lens(init: str, *, hf_model_name: str, d_model: int) -> JacobianLens | None:
    """``--init``: ``None`` for :data:`IDENTITY_INIT`, else the Jacobian-format lens
    file at that path."""
    if init == IDENTITY_INIT:
        return None
    lens = load_lens_file(init, hf_model_name=hf_model_name)
    if not isinstance(lens, JacobianLens) or lens.d_model != d_model:
        raise ValueError(f"--init {init} is not a d_model={d_model} Jacobian lens")
    return lens


def fit_intervention(
    pairs_dirs: str | Sequence[str],
    *,
    layers: Sequence[int],
    checkpoint_name: str,
    target: str = "sum_future",
    loss: str = "residual",
    init: str = IDENTITY_INIT,
    holdout_fraction: float = 0.1,
    holdout_seed: int = 0,
    device: str = DEFAULT_MERGE_DEVICE,
    **fit_kwargs: object,
) -> tuple[JacobianLens, dict[int, TransportFit]]:
    """Stage: fit a lens on one family's pairs, or on several families' pairs pooled
    (:func:`fit_intervention_lens`), from the identity or from the lens file ``init``.
    Needs no model: the pairs directories carry the settings, and the logit loss reads
    the first one's parent's ``gram.pt``."""
    if isinstance(pairs_dirs, str):
        pairs_dirs = [pairs_dirs]
    meta = PairsMeta.load(pairs_dirs[0])
    return fit_intervention_lens(
        pairs_dirs,
        layers=layers,
        target=target,  # type: ignore[arg-type]
        loss=loss,  # type: ignore[arg-type]
        init_lens=_init_lens(init, hf_model_name=meta.hf_model_name, d_model=meta.d_model),
        checkpoint_name=checkpoint_name,
        holdout_fraction=holdout_fraction,
        holdout_seed=holdout_seed,
        device=device,
        **fit_kwargs,
    )


def fit_output_kl_lens(
    model: LensModel,
    pool_dir: str,
    *,
    layers: Sequence[int],
    checkpoint_name: str,
    init: str = IDENTITY_INIT,
    holdout_fraction: float = 0.1,
    holdout_seed: int = 0,
    **fit_kwargs: object,
) -> tuple[JacobianLens, dict[int, TransportFit]]:
    """Stage: the Tuned-Lens-style lens on a pairs directory's clean pool
    (:func:`fit_output_kl` per layer), with the same held-out prompts as
    :func:`fit_intervention` for the same fraction and seed."""
    meta = PairsMeta.load(pool_dir)
    model_name = hf_model_name_of(model)
    if model.d_model != meta.d_model or (model_name is not None and model_name != meta.hf_model_name):
        raise ValueError(f"the pool was recorded on {meta.hf_model_name}, not this model")
    pool = CleanPool.load(Path(pool_dir) / CLEAN_POOL_FILENAME)
    heldout_prompts_Bool = heldout_prompt_mask(meta.num_prompts, holdout_fraction, seed=holdout_seed)
    init_lens = _init_lens(init, hf_model_name=meta.hf_model_name, d_model=meta.d_model)
    fits_L_dict: dict[int, TransportFit] = {}
    for layer in layers:
        init_FN = (
            init_lens.jacobians_L_dict_FN[layer]  # type: ignore[index]
            if init_lens is not None
            else t.eye(meta.d_model)
        )
        fits_L_dict[layer] = fit_output_kl(
            model,
            pool,
            layer=layer,
            heldout_prompts_Bool=heldout_prompts_Bool,
            init_FN=init_FN,
            **fit_kwargs,  # type: ignore[arg-type]
        )
        logger.info(
            "  layer %d: held-out KL %.4f (init) -> %.4f (step %d)",
            layer,
            fits_L_dict[layer].init_heldout_relative_error,
            fits_L_dict[layer].best_heldout_relative_error,
            fits_L_dict[layer].best_step,
        )
    lens = JacobianLens(
        jacobians={layer: fit.matrix_FN for layer, fit in fits_L_dict.items()},
        config=intervention_lens_config(
            meta,
            checkpoint_name=checkpoint_name,
            num_prompts_trained_on=int((~heldout_prompts_Bool).sum()),
        ),
    )
    return lens, fits_L_dict


def swap_examples_from_items(
    model: LensModel, items: Sequence[dict], *, max_seq_len: int
) -> tuple[list[str], list[SwapExample]]:
    """``(prompts, examples)`` for the supervised swap loss from items in the probe-swap
    format: X and Y resolved to their swap tokens as probe-swap does
    (:func:`resolve_swap_token`), the answer ids every single-token surface of Y's answer
    (``swap_answer_surfaces`` when given, else ``swap_answer``), read at the last position.
    Items with no swap token or answer token are dropped (logged;
    :func:`swap_examples_and_kept_items` also says which)."""
    prompts, examples, _ = swap_examples_and_kept_items(model, items, max_seq_len=max_seq_len)
    return prompts, examples


def swap_examples_and_kept_items(
    model: LensModel, items: Sequence[dict], *, max_seq_len: int
) -> tuple[list[str], list[SwapExample], list[int]]:
    """:func:`swap_examples_from_items` plus, per example, the index of its item."""
    prompts = sorted({item["prompt"] for item in items})
    prompt_index = {prompt: idx for idx, prompt in enumerate(prompts)}
    examples: list[SwapExample] = []
    kept_item_indices: list[int] = []
    for item_idx, item in enumerate(items):
        source_id, _ = resolve_swap_token(model.tokenizer, item["intermediate"])
        target_id, _ = resolve_swap_token(model.tokenizer, item["swap_to"])
        answers = item.get("swap_answer_surfaces") or [item["swap_answer"]]
        answer_ids = sorted(
            {token_id for answer in answers for token_id in single_token_ids(model.tokenizer, surface_forms(answer))}
        )
        if source_id is None or target_id is None or not answer_ids:
            logger.warning("dropping swap item %s: no single-token swap or answer form", item.get("name"))
            continue
        position = model.encode(item["prompt"], max_length=max_seq_len).shape[1] - 1
        examples.append(
            SwapExample(prompt_index[item["prompt"]], position, source_id, target_id, tuple(answer_ids))
        )
        kept_item_indices.append(item_idx)
    return prompts, examples, kept_item_indices


def fit_swap(
    model: LensModel,
    prompts: Sequence[str],
    *,
    layers: Sequence[int],
    hf_model_name: str,
    checkpoint_name: str,
    mode: str = "next_token",
    init: str = IDENTITY_INIT,
    examples_per_prompt: int = 8,
    top_k: int = 20,
    window: int = 8,
    max_seq_len: int = RECIPE_MAX_SEQ_LEN,
    skip_first_n_positions: int = RECIPE_SKIP_FIRST_N_POSITIONS,
    holdout_fraction: float = 0.1,
    holdout_seed: int = 0,
    num_heldout_examples: int = 256,
    seed: int = 0,
    swap_items: Sequence[dict] | None = None,
    monitor_items: Sequence[dict] | None = None,
    **fit_kwargs: object,
) -> tuple[JacobianLens, SwapFit]:
    """Stage: the swap-trained lens. Prompts that fill ``max_seq_len`` tokens are split
    into training and held-out prompts (:func:`heldout_prompt_mask`, the split the other
    intervention stages use); swap examples are drawn from each
    (:func:`draw_swap_examples`, X and Y never formatting tokens); the band layers' maps
    start from the identity or from the lens file ``init`` and are trained jointly
    (:func:`fit_swap_maps`). In supervised mode the rows are ``swap_items`` instead (a built
    swap dataset, :func:`swap_examples_from_items`), a seeded ``holdout_fraction`` of them
    held out for early stopping; ``prompts`` is then ignored. ``monitor_items`` (swap items
    from held-out relations) are scored along the way and logged, never trained on."""
    vocab_size = vocab_size_of(model)
    excluded_mask_Bool_V = token_id_mask_Bool_V(formatting_token_ids_of(model.tokenizer, vocab_size), vocab_size)
    if mode == "supervised":
        if swap_items is None:
            raise ValueError("supervised mode needs swap_items")
        prompts, examples = swap_examples_from_items(model, swap_items, max_seq_len=max_seq_len)
        order = t.randperm(len(examples), generator=t.Generator().manual_seed(holdout_seed)).tolist()
        num_heldout = max(1, round(holdout_fraction * len(examples)))
        heldout_examples = [examples[i] for i in order[:num_heldout]][:num_heldout_examples]
        train_examples = [examples[i] for i in order[num_heldout:]]
        num_training_prompts = len({example.prompt_idx for example in train_examples})
        logger.info("swap examples: %d training, %d held out", len(train_examples), len(heldout_examples))
    else:
        train_examples, heldout_examples, num_training_prompts = _draw_wikitext_swap_examples(
            model,
            prompts,
            mode=mode,
            examples_per_prompt=examples_per_prompt,
            top_k=top_k,
            excluded_mask_Bool_V=excluded_mask_Bool_V,
            window=window,
            max_seq_len=max_seq_len,
            skip_first_n_positions=skip_first_n_positions,
            holdout_fraction=holdout_fraction,
            holdout_seed=holdout_seed,
            num_heldout_examples=num_heldout_examples,
            seed=seed,
        )
    monitor_examples: list[SwapExample] = []
    if monitor_items:
        monitor_prompts, monitor_examples = swap_examples_from_items(model, monitor_items, max_seq_len=max_seq_len)
        offset = len(prompts)
        prompts = [*prompts, *monitor_prompts]
        monitor_examples = [
            dataclasses.replace(example, prompt_idx=example.prompt_idx + offset)
            for example in monitor_examples[:num_heldout_examples]
        ]
    init_lens = _init_lens(init, hf_model_name=hf_model_name, d_model=model.d_model)
    init_maps_L_dict_FN = {
        layer: (
            init_lens.jacobians_L_dict_FN[layer]  # type: ignore[index]
            if init_lens is not None
            else t.eye(model.d_model)
        )
        for layer in layers
    }
    fit = fit_swap_maps(
        model,
        prompts,
        train_examples,
        heldout_examples,
        init_maps_L_dict_FN=init_maps_L_dict_FN,
        mode=mode,  # type: ignore[arg-type]
        window=window,
        max_seq_len=max_seq_len,
        seed=seed,
        excluded_mask_Bool_V=excluded_mask_Bool_V,
        monitor_examples=monitor_examples,
        **fit_kwargs,  # type: ignore[arg-type]
    )
    config = swap_lens_config(
        hf_model_name=hf_model_name,
        checkpoint_name=checkpoint_name,
        layers=layers,
        d_model=model.d_model,
        num_prompts_trained_on=num_training_prompts,
        max_seq_len=max_seq_len,
        skip_first_n_positions=skip_first_n_positions,
    )
    return swap_lens(fit, config), fit


def _draw_wikitext_swap_examples(
    model: LensModel,
    prompts: Sequence[str],
    *,
    mode: str,
    examples_per_prompt: int,
    top_k: int,
    excluded_mask_Bool_V: t.Tensor,
    window: int,
    max_seq_len: int,
    skip_first_n_positions: int,
    holdout_fraction: float,
    holdout_seed: int,
    num_heldout_examples: int,
    seed: int,
) -> tuple[list[SwapExample], list[SwapExample], int]:
    """The next_token / counterfactual rows: training and held-out examples from the
    full-length training and held-out prompts, and the number of training prompts."""
    full_length = full_length_prompts(model, prompts, max_seq_len)
    heldout_prompts_Bool = heldout_prompt_mask(len(prompts), holdout_fraction, seed=holdout_seed)
    train_indices = [idx for idx in full_length if not heldout_prompts_Bool[idx]]
    heldout_indices = [idx for idx in full_length if heldout_prompts_Bool[idx]]
    draw_kwargs = dict(
        mode=mode,
        examples_per_prompt=examples_per_prompt,
        top_k=top_k,
        excluded_mask_Bool_V=excluded_mask_Bool_V.cpu(),
        max_seq_len=max_seq_len,
        skip_first_n_positions=skip_first_n_positions,
        window=window,
    )
    train_examples = draw_swap_examples(model, prompts, train_indices, seed=seed, **draw_kwargs)  # type: ignore[arg-type]
    heldout_examples = draw_swap_examples(model, prompts, heldout_indices, seed=seed + 1, **draw_kwargs)  # type: ignore[arg-type]
    heldout_examples = heldout_examples[:num_heldout_examples]
    logger.info(
        "swap examples: %d training (%d prompts), %d held out (%d of %d prompts full length)",
        len(train_examples),
        len(train_indices),
        len(heldout_examples),
        len(full_length),
        len(prompts),
    )
    return train_examples, heldout_examples, len(train_indices)


### STAMPS


def git_commit_hash() -> str | None:
    """``git rev-parse HEAD`` of the checkout this module lives in, or None
    when git or the repository is unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def library_version() -> str | None:
    """The installed ``jpp-lens`` package version, or None when the
    package is not installed (e.g. run from a bare checkout)."""
    try:
        return importlib.metadata.version("jpp-lens")
    except importlib.metadata.PackageNotFoundError:
        return None


def write_stamp(path: str, payload: dict[str, object]) -> None:
    """Write ``payload`` as JSON with provenance keys added: ``git_hash``,
    ``timestamp``, ``library_version``."""
    stamp = {
        "git_hash": git_commit_hash(),
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "library_version": library_version(),
        **payload,
    }
    Path(path).write_text(json.dumps(stamp, indent=2))


def sibling_path(output_path: str, suffix: str) -> str:
    """``foo/bar.pt`` with ``suffix`` in place of the extension: ``foo/bar<suffix>``."""
    return f"{os.path.splitext(output_path)[0]}{suffix}"


def stamp_path_beside(output_path: str) -> str:
    """``foo/bar.pt`` -> ``foo/bar_stamp.json``."""
    return sibling_path(output_path, "_stamp.json")


def write_stage_stamp(args: argparse.Namespace, output_path: str, **extra: object) -> str:
    """Called before a stage writes ``output_path``: creates its directory and
    writes the stage's stamp beside it (:func:`stamp_path_beside`) — the stage
    name, the parsed arguments, :func:`write_stamp`'s provenance keys and
    ``extra``. Returns the stamp path."""
    ensure_parent_dir(output_path)
    stamp_path = stamp_path_beside(output_path)
    write_stamp(stamp_path, {"stage": args.stage, "args": vars(args), **extra})
    return stamp_path


### COMMAND LINE


def parse_int_list(text: str) -> list[int]:
    """``"8,16,24"`` -> ``[8, 16, 24]`` (argparse ``type``)."""
    return [int(item) for item in text.split(",")]


def load_lenses(
    specs: Sequence[str], model: LensModel, *, hf_model_name: str, layers: Sequence[int]
) -> dict[str, BaseLens]:
    """``{name: lens}`` for each ``--lens`` value: a file is named by its stem
    (:func:`load_lens_file`), and :data:`LOGIT_LENS_SPEC` builds the Logit Lens at
    ``layers``. Two lenses with one name would silently shadow each other, so
    that is an error."""
    lenses: dict[str, BaseLens] = {}
    for spec in specs:
        lens_name = LOGIT_LENS_SPEC if spec == LOGIT_LENS_SPEC else Path(spec).stem
        if lens_name in lenses:
            raise ValueError(f"two lenses share the name {lens_name!r}: {specs}")
        if spec == LOGIT_LENS_SPEC:
            lenses[lens_name] = LogitLens(
                source_layers=list(layers),
                config=LensConfig(
                    hf_model_name=hf_model_name,
                    checkpoint_name=LOGIT_LENS_SPEC,
                    lens_type="logit",
                    d_model=model.d_model,
                ),
            )
        else:
            lenses[lens_name] = load_lens_file(spec, hf_model_name=hf_model_name)
    return lenses


def parse_float_list(text: str) -> list[float]:
    """``"0.5,1,2"`` -> ``[0.5, 1.0, 2.0]`` (argparse ``type``)."""
    return [float(item) for item in text.split(",")]


def _add_model_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--hf-model-name",
        default=RECIPE_HF_MODEL_NAME,
        help="HF model to load (bf16; the attention, experts and dequantization flags below). Default %(default)s.",
    )
    parser.add_argument(
        "--device-map",
        default=None,
        help="HF device_map for a model too large for one GPU ('auto' shards it over the "
        "visible GPUs). Default: the single-GPU .cuda() load.",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="HF attention path (DeepSeek-V4 supports only 'eager'). Default %(default)s.",
    )
    parser.add_argument(
        "--experts-implementation",
        default=None,
        help="HF MoE experts path ('eager' for DeepSeek-V4's per-expert loop). Default: the library's.",
    )
    parser.add_argument(
        "--dequantize-fp8",
        action="store_true",
        help="Unpack a native FP8/FP4 checkpoint to bf16 on load (DeepSeek-V4).",
    )


def _add_layers_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--layers",
        type=parse_int_list,
        default=RECIPE_LAYERS,
        help="Comma-separated source layers. Default %(default)s.",
    )


def _add_eval_items_arguments(parser: argparse.ArgumentParser, *, default_items: str) -> None:
    """The item-selection arguments shared by fit-weights and evaluate."""
    parser.add_argument(
        "--items",
        default=default_items,
        help="'all' (every scoreable eval item), 'fit:<seed>' (the fitting split, for the "
        "expert weights) or 'held-out:<seed>' (the rest, for scoring). Default %(default)s.",
    )
    parser.add_argument(
        "--fit-fraction",
        type=float,
        default=DEFAULT_FIT_FRACTION,
        help="Share of each eval's items in the fitting split; give evaluate the same value as "
        "fit-weights. Default %(default)s.",
    )
    _add_eval_data_arguments(parser)


def _add_eval_data_arguments(parser: argparse.ArgumentParser) -> None:
    """Where the eval items and the model-correctness filter are read from."""
    parser.add_argument(
        "--eval-data-dir",
        default=READOUT_EVALS_DATA_DIR,
        help="Directory of the lens-eval-<slug>.json files. Default %(default)s.",
    )
    parser.add_argument(
        "--correctness-csv",
        default=MODEL_CORRECTNESS_CSV,
        help="Model-correctness CSV (column = --hf-model-name). Default %(default)s.",
    )
    parser.add_argument(
        "--readout-max-seq-len",
        type=int,
        default=RECIPE_READOUT_MAX_SEQ_LEN,
        help="Eval prompt truncation. Default %(default)s.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python scripts/jpp_cli.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="stage", required=True)

    fit_router_parser = subparsers.add_parser(
        "fit-router", help="Fit the per-layer PCA + k-means router (stage 1)."
    )
    _add_model_argument(fit_router_parser)
    _add_layers_argument(fit_router_parser)
    fit_router_parser.add_argument(
        "--num-prompts",
        type=int,
        default=NUM_WIKITEXT_FIT_PROMPTS,
        help="First N of the WikiText pool. Default %(default)s.",
    )
    fit_router_parser.add_argument("--num-clusters", type=int, default=RECIPE_NUM_CLUSTERS)
    fit_router_parser.add_argument("--projection-dim", type=int, default=64)
    fit_router_parser.add_argument(
        "--skip-first-n-positions", type=int, default=RECIPE_SKIP_FIRST_N_POSITIONS
    )
    fit_router_parser.add_argument("--max-seq-len", type=int, default=RECIPE_MAX_SEQ_LEN)
    fit_router_parser.add_argument("--seed", type=int, default=0)
    fit_router_parser.add_argument(
        "--out", required=True, help="Router .pt path (stamp written beside it)."
    )

    fit_shard_parser = subparsers.add_parser(
        "fit-shard", help="Fit one prompt shard's expert sums (stage 2)."
    )
    _add_model_argument(fit_shard_parser)
    _add_layers_argument(fit_shard_parser)
    fit_shard_parser.add_argument(
        "--router-path",
        nargs="+",
        required=True,
        help="Router .pt file(s), space-separated, each as path or name=path. Several "
        "routers bucket the same backward passes into one checkpoint each: a router given "
        "without a name is keyed by its K (experts_K<K>_checkpoint.pt), a named one by its "
        "name (experts_<name>_checkpoint.pt), so routers with the same K need names.",
    )
    fit_shard_parser.add_argument("--shard-idx", type=int, default=0)
    fit_shard_parser.add_argument("--num-shards", type=int, default=1)
    fit_shard_parser.add_argument(
        "--num-prompts",
        type=int,
        default=64,
        help="First N of the WikiText pool, sharded contiguously. Default %(default)s.",
    )
    fit_shard_parser.add_argument(
        "--prompts-json",
        default=None,
        help="Read the prompt pool from this JSON list (the first --num-prompts are used) "
        "instead of streaming WikiText from the Hub, so many shard jobs need not stream it "
        "concurrently. Default: stream.",
    )
    fit_shard_parser.add_argument(
        "--target-offset",
        type=int,
        default=-1,
        help="relative_end_transport_layer. Default %(default)s (the final block).",
    )
    fit_shard_parser.add_argument(
        "--lrp-mode",
        choices=LRP_MODES,
        default="rlens",
        help="'none' fits standard gradients with ExpertJacobianTrainer; any other "
        "mode fits RelP rows with ExpertJacobianRelPTrainer. Default %(default)s.",
    )
    fit_shard_parser.add_argument("--jacobian-rows-per-pass", type=int, default=16)
    fit_shard_parser.add_argument("--max-seq-len", type=int, default=RECIPE_MAX_SEQ_LEN)
    fit_shard_parser.add_argument(
        "--skip-first-n-positions", type=int, default=RECIPE_SKIP_FIRST_N_POSITIONS
    )
    fit_shard_parser.add_argument("--checkpoint-every-n-prompts", type=int, default=4)
    fit_shard_parser.add_argument("--artifacts-base-dir", default="artifacts")
    fit_shard_parser.add_argument(
        "--checkpoint-name",
        default="recipe_experts",
        help="checkpoint_name of the shard configs; the shard writes to "
        "<artifacts-base-dir>/<date>/<name>/shard<i>of<n>/. Default %(default)s.",
    )

    merge_parser = subparsers.add_parser(
        "merge-experts",
        help="Merge shard checkpoints into expert Jacobians (stage 3).",
    )
    merge_parser.add_argument(
        "--checkpoint-dirs", nargs="+", required=True, help="Shard directories."
    )
    merge_parser.add_argument("--num-clusters", type=int, default=RECIPE_NUM_CLUSTERS)
    merge_parser.add_argument(
        "--partition",
        default=None,
        help="Merge the named partition (experts_<name>_checkpoint.pt) instead of the one "
        "keyed by --num-clusters.",
    )
    merge_parser.add_argument(
        "--min-kept-positions",
        type=int,
        default=RECIPE_MIN_KEPT_POSITIONS,
        help="Experts with fewer kept positions fall back to the pooled Jacobian. "
        "Default %(default)s.",
    )
    merge_parser.add_argument(
        "--checkpoint-name",
        default="recipe_experts",
        help="checkpoint_name stored in the experts' config (replaces the "
        "inherited shard-0 name). Default %(default)s.",
    )
    merge_parser.add_argument(
        "--device",
        default=DEFAULT_MERGE_DEVICE,
        help="Device the shards are streamed to and summed on. Default %(default)s "
        "(cuda when available).",
    )
    merge_parser.add_argument(
        "--out", required=True, help="ExpertJacobians .pt path (stamp beside it)."
    )
    merge_parser.add_argument(
        "--pooled-lens-out",
        default=None,
        help="Optional: also write the fp32 pooled (R-Lens) Jacobian lens of the merged "
        "sums to this path (see ExpertJacobians.pooled_lens: the experts file's pooled key "
        "is fp16).",
    )

    pooled_lens_parser = subparsers.add_parser(
        "pooled-lens",
        help="Optional: the pooled Jacobian lens of an experts file (the R-Lens of an LRP fit).",
    )
    pooled_lens_parser.add_argument(
        "--experts", required=True, help="ExpertJacobians file (the output of merge-experts)."
    )
    pooled_lens_parser.add_argument(
        "--checkpoint-name",
        default="recipe_pooled_lens",
        help="checkpoint_name of the pooled lens. Default %(default)s.",
    )
    pooled_lens_parser.add_argument(
        "--out", required=True, help="Lens .pt path (saved fp32); the stamp is written beside it."
    )

    fit_weights_parser = subparsers.add_parser(
        "fit-weights",
        help="Fit expert weights on labelled items and build the lens (stage 4).",
    )
    _add_model_argument(fit_weights_parser)
    _add_layers_argument(fit_weights_parser)
    fit_weights_parser.add_argument(
        "--experts",
        required=True,
        help="ExpertJacobians file (the output of merge-experts).",
    )
    fit_weights_parser.add_argument(
        "--checkpoint-name",
        default="recipe_lens",
        help="checkpoint_name of the combined lens. Default %(default)s.",
    )
    _add_eval_items_arguments(fit_weights_parser, default_items="fit:0")
    fit_weights_parser.add_argument(
        "--cache-path",
        default=None,
        help="Readout-residuals cache (recorded once for every eval item, so fits on "
        "different item splits share it). Optional.",
    )
    fit_weights_parser.add_argument("--k", type=int, default=DEFAULT_CONDENSE_CONFIG.pass_k)
    fit_weights_parser.add_argument("--top-n", type=int, default=DEFAULT_CONDENSE_CONFIG.top_n)
    fit_weights_parser.add_argument("--steps", type=int, default=DEFAULT_CONDENSE_CONFIG.steps)
    fit_weights_parser.add_argument(
        "--keep-fallback-copies",
        action="store_true",
        help="Search over every expert slot as stored. Default: each layer's copies of the "
        "pooled Jacobian (experts with too few positions) are merged into one slot first.",
    )
    fit_weights_parser.add_argument(
        "--out",
        required=True,
        help="Lens .pt path (saved fp32); <stem>_expert_weights.json and the "
        "stamp are written beside it.",
    )

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="Score lenses on the readout evals: recall@k, item- and pair-weighted (stage 5)."
    )
    _add_model_argument(evaluate_parser)
    _add_layers_argument(evaluate_parser)
    evaluate_parser.add_argument(
        "--lens",
        nargs="+",
        required=True,
        help="Lenses to score: .pt paths (each named by its file stem) or 'logit' for the "
        "Logit Lens.",
    )
    _add_eval_items_arguments(evaluate_parser, default_items="held-out:0")
    _add_scoring_arguments(evaluate_parser)

    probe_swap_parser = subparsers.add_parser(
        "probe-swap", help="Run the probe-swap causal eval (stage 6)."
    )
    _add_model_argument(probe_swap_parser)
    _add_layers_argument(probe_swap_parser)
    probe_swap_parser.add_argument(
        "--lens",
        nargs="+",
        required=True,
        help="Lenses: .pt paths (each named by its file stem) or 'logit' for the Logit Lens.",
    )
    probe_swap_parser.add_argument(
        "--scales",
        type=parse_float_list,
        default=list(PROBE_SWAP_SCALES),
        help="Comma-separated edit scales. Default %(default)s.",
    )
    probe_swap_parser.add_argument(
        "--max-seq-len", type=int, default=PROBE_SWAP_MAX_SEQ_LEN, help="Default %(default)s."
    )
    probe_swap_parser.add_argument(
        "--data",
        default=PROBE_SWAP_PATH,
        help="Swap items in the probe-swap JSON format (e.g. a held-out swap dataset). "
        "Default %(default)s.",
    )
    probe_swap_parser.add_argument(
        "--out",
        required=True,
        help="Trials parquet path; <stem>_success.csv, <stem>_best_scale.csv and the stamp are "
        "written beside it.",
    )

    _add_intervention_parsers(subparsers)
    return parser


def _add_fit_loop_arguments(
    parser: argparse.ArgumentParser, *, steps: int, batch_size: int, eval_every: int, patience: int
) -> None:
    """The Adam loop's flags, shared by fit-intervention and fit-output-kl."""
    parser.add_argument(
        "--init",
        default=IDENTITY_INIT,
        help=f"'{IDENTITY_INIT}' or a lens .pt to start every map from. Default %(default)s.",
    )
    parser.add_argument("--steps", type=int, default=steps)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--eval-every", type=int, default=eval_every)
    parser.add_argument(
        "--patience",
        type=int,
        default=patience,
        help="Stop after this many evaluations without a better held-out loss. Default %(default)s.",
    )
    parser.add_argument("--holdout-fraction", type=float, default=0.1)
    parser.add_argument("--holdout-seed", type=int, default=0)
    parser.add_argument("--checkpoint-name", required=True)
    parser.add_argument(
        "--out",
        required=True,
        help="Lens .pt path (saved fp32); <stem>_fit_history.json and the stamp are written "
        "beside it.",
    )


def _add_intervention_parsers(subparsers: argparse._SubParsersAction) -> None:
    collect_parser = subparsers.add_parser(
        "collect-pairs", help="Record the clean pool and the edit pairs of the intervention-trained lens."
    )
    _add_model_argument(collect_parser)
    _add_layers_argument(collect_parser)
    collect_parser.add_argument("--num-prompts", type=int, default=1024)
    collect_parser.add_argument(
        "--prompts-json", default=None, help="Read the prompt pool from this JSON list. Default: stream WikiText."
    )
    collect_parser.add_argument(
        "--families",
        default="gauss:0.05,swap:1",
        help="Comma-separated edit families: 'swap:<alpha>', 'gauss:<size>', 'zero', 'mean'. "
        "Default %(default)s.",
    )
    collect_parser.add_argument("--pairs-per-prompt", type=int, default=32)
    collect_parser.add_argument(
        "--rows-per-pass",
        type=int,
        default=128,
        help="Batch rows per forward pass (one clean row plus edits). Default %(default)s.",
    )
    collect_parser.add_argument("--target-offset", type=int, default=-1)
    collect_parser.add_argument("--max-seq-len", type=int, default=RECIPE_MAX_SEQ_LEN)
    collect_parser.add_argument(
        "--skip-first-n-positions", type=int, default=RECIPE_SKIP_FIRST_N_POSITIONS
    )
    collect_parser.add_argument("--seed", type=int, default=0)
    collect_parser.add_argument("--out-dir", required=True)

    fit_parser = subparsers.add_parser(
        "fit-intervention", help="Fit the intervention-trained lens on one family's pairs (no model)."
    )
    _add_layers_argument(fit_parser)
    fit_parser.add_argument(
        "--pairs-dir",
        nargs="+",
        required=True,
        help="One family's directory from collect-pairs, or several to fit on their pairs pooled.",
    )
    fit_parser.add_argument("--target", choices=["sum_future", "same_position"], default="sum_future")
    fit_parser.add_argument("--loss", choices=["residual", "logit"], default="residual")
    fit_parser.add_argument("--device", default=DEFAULT_MERGE_DEVICE)
    _add_fit_loop_arguments(fit_parser, steps=5000, batch_size=1024, eval_every=50, patience=20)

    kl_parser = subparsers.add_parser(
        "fit-output-kl", help="Fit the Tuned-Lens-style lens (KL to the model's own output) on the clean pool."
    )
    _add_model_argument(kl_parser)
    _add_layers_argument(kl_parser)
    kl_parser.add_argument("--pool-dir", required=True, help="The collect-pairs --out-dir.")
    _add_fit_loop_arguments(kl_parser, steps=2000, batch_size=256, eval_every=100, patience=10)

    experts_parser = subparsers.add_parser(
        "build-experts", help="An ExpertJacobians file whose experts are whole lenses (for fit-weights)."
    )
    _add_layers_argument(experts_parser)
    experts_parser.add_argument("--hf-model-name", default=RECIPE_HF_MODEL_NAME)
    experts_parser.add_argument("--lens", nargs="+", required=True, help="Lens .pt files, one per expert.")
    experts_parser.add_argument("--out", required=True)

    swap_parser = subparsers.add_parser(
        "fit-swap", help="Train the band layers' maps by descending on the swap loss (probe-swap's edit on WikiText)."
    )
    _add_model_argument(swap_parser)
    _add_layers_argument(swap_parser)
    swap_parser.add_argument("--num-prompts", type=int, default=1024)
    swap_parser.add_argument("--prompts-json", default=None)
    swap_parser.add_argument(
        "--mode",
        choices=["next_token", "counterfactual", "supervised"],
        default="next_token",
        help="next_token: X is the model's top prediction at s, the loss is Y's cross-entropy "
        "there. counterfactual: X is the token at s, the target is the run with Y written at s "
        "(KL over --window positions). supervised: rows from --swap-data, the loss is the "
        "cross-entropy of Y's answer at the last position. Default %(default)s.",
    )
    swap_parser.add_argument(
        "--swap-data", default=None, help="Supervised mode: swap items in the probe-swap JSON format."
    )
    swap_parser.add_argument(
        "--monitor-data",
        default=None,
        help="Swap items (e.g. held-out relations) scored at every evaluation and logged, never "
        "trained on or used for stopping.",
    )
    swap_parser.add_argument("--init", default=IDENTITY_INIT)
    swap_parser.add_argument("--examples-per-prompt", type=int, default=8)
    swap_parser.add_argument("--top-k", type=int, default=20, help="Y is drawn from these top candidates.")
    swap_parser.add_argument("--scale", type=float, default=1.0, help="The clamp swap's scale.")
    swap_parser.add_argument("--window", type=int, default=8)
    swap_parser.add_argument("--max-seq-len", type=int, default=RECIPE_MAX_SEQ_LEN)
    swap_parser.add_argument("--skip-first-n-positions", type=int, default=RECIPE_SKIP_FIRST_N_POSITIONS)
    swap_parser.add_argument("--steps", type=int, default=2000)
    swap_parser.add_argument("--batch-size", type=int, default=16)
    swap_parser.add_argument(
        "--length-pool",
        type=int,
        default=32,
        help="Draw this many batches' rows at once and group them by prompt length, so batches "
        "carry little padding (1: every batch drawn on its own). Default %(default)s.",
    )
    swap_parser.add_argument(
        "--clean-cache-gb",
        type=float,
        default=40.0,
        help="Run every prompt's clean pass once and keep it on the GPU when it takes at most this "
        "many GB and fits next to training (measured); 0: run it every step. Default %(default)s.",
    )
    swap_parser.add_argument(
        "--edit-precision",
        choices=["fp32", "model"],
        default="fp32",
        help="Dtype of the swap edit's products with the residuals: fp32, or the model's own "
        "(bf16, faster). Default %(default)s.",
    )
    swap_parser.add_argument(
        "--relative-lr", type=float, default=1e-2, help="Adam's step relative to each map's entry RMS."
    )
    swap_parser.add_argument(
        "--all-layers-probability",
        type=float,
        default=1.0,
        help="Share of steps that swap at every layer; the rest swap at one random window of "
        "--window-sizes consecutive layers. Default %(default)s (always every layer).",
    )
    swap_parser.add_argument(
        "--window-sizes", type=parse_int_list, default=[], help="e.g. 1,3,5 (see --all-layers-probability)."
    )
    swap_parser.add_argument("--eval-every", type=int, default=100)
    swap_parser.add_argument("--patience", type=int, default=5)
    swap_parser.add_argument("--holdout-fraction", type=float, default=0.1)
    swap_parser.add_argument("--holdout-seed", type=int, default=0)
    swap_parser.add_argument("--heldout-examples", type=int, default=256)
    swap_parser.add_argument("--seed", type=int, default=0)
    swap_parser.add_argument(
        "--wandb-project", default=None, help="Log the training curves to this W&B project (needs the tracking extra)."
    )
    swap_parser.add_argument("--wandb-run-name", default=None, help="Default: --checkpoint-name.")
    swap_parser.add_argument("--checkpoint-name", required=True)
    swap_parser.add_argument("--out", required=True)


def _add_scoring_arguments(parser: argparse.ArgumentParser) -> None:
    """The readout-scoring arguments of evaluate."""
    parser.add_argument(
        "--no-readout-filter",
        action="store_true",
        help="Rank the full vocabulary. Default: tokens with no letter or digit are left out "
        "of the ranking (Readout Filtering), the eval items' candidate tokens excepted.",
    )
    parser.add_argument(
        "--ks",
        type=parse_int_list,
        default=DEFAULT_PASS_KS,
        help="Comma-separated ks. Default %(default)s.",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Ranks CSV path; <stem>_pass_at_k.csv and the stamp are written beside it.",
    )


def _load_model(args: argparse.Namespace) -> LensModel:
    """The ``--hf-model-name`` model (bf16; ``--attn-implementation``, SDPA by
    default; sharded over the visible GPUs when ``--device-map`` is given;
    DeepSeek-V4's eager experts and dequantized load through their flags) — the
    one place a real model is loaded."""
    return get_hf_model(
        args.hf_model_name,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
        experts_implementation=args.experts_implementation,
        dequantize_fp8=args.dequantize_fp8,
    )


def _run_fit_router(args: argparse.Namespace) -> None:
    model = _load_model(args)
    router = fit_router(
        model,
        load_fit_prompts(args.num_prompts),
        layers=args.layers,
        num_clusters=args.num_clusters,
        projection_dim=args.projection_dim,
        skip_first_n_positions=args.skip_first_n_positions,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
    )
    stamp_path = write_stage_stamp(args, args.out)
    router.save(args.out)
    logger.info("router saved to %s (stamp %s)", args.out, stamp_path)


def _run_fit_shard(args: argparse.Namespace) -> None:
    routers = load_partitions(args.router_path)
    config = LensConfig(
        hf_model_name=args.hf_model_name,
        checkpoint_name=args.checkpoint_name,
        artifacts_base_dir=args.artifacts_base_dir,
        source_layers=args.layers,
        relative_end_transport_layer=args.target_offset,
        jacobian_rows_per_pass=args.jacobian_rows_per_pass,
        max_seq_len=args.max_seq_len,
        skip_first_n_positions=args.skip_first_n_positions,
        checkpoint_every_n_prompts=args.checkpoint_every_n_prompts,
        lrp_mode=args.lrp_mode,
    )
    prompts = (
        load_fit_prompts_from_json(args.prompts_json, args.num_prompts)
        if args.prompts_json is not None
        else load_fit_prompts(args.num_prompts)
    )
    checkpoint_dir = fit_shard(
        _load_model(args),
        prompts,
        routers=routers,
        shard_idx=args.shard_idx,
        num_shards=args.num_shards,
        config=config,
    )
    # The shard's output is a directory the trainer created; its stamp goes inside.
    write_stamp(
        os.path.join(checkpoint_dir, "fit_shard_stamp.json"),
        {
            "stage": args.stage,
            "args": vars(args),
            "checkpoint_dir": checkpoint_dir,
        },
    )
    logger.info("shard checkpoint in %s", checkpoint_dir)
    print(checkpoint_dir)


def _run_merge_experts(args: argparse.Namespace) -> None:
    expert_jacobians = merge_experts(
        args.checkpoint_dirs,
        num_clusters=args.partition if args.partition is not None else args.num_clusters,
        min_kept_positions=args.min_kept_positions,
        checkpoint_name=args.checkpoint_name,
        device=args.device,
    )
    stamp_path = write_stage_stamp(
        args,
        args.out,
        num_prompts_trained_on=expert_jacobians.config.num_prompts_trained_on,
        # Experts replaced by the pooled Jacobian, per layer.
        fallback_experts={
            str(layer): fallback_Bool_E.nonzero().flatten().tolist()
            for layer, fallback_Bool_E in expert_jacobians.fallback_L_dict_Bool_E.items()
        },
    )
    expert_jacobians.save(args.out)
    logger.info("experts saved to %s (stamp %s)", args.out, stamp_path)
    if args.pooled_lens_out is not None:
        _write_pooled_lens(
            args,
            expert_jacobians,
            args.pooled_lens_out,
            checkpoint_name=f"{args.checkpoint_name}_pooled",
            pooled_from="in-memory fp32 sums",
        )


def _write_pooled_lens(
    args: argparse.Namespace,
    expert_jacobians: ExpertJacobians,
    out_path: str,
    *,
    checkpoint_name: str,
    pooled_from: str,
) -> None:
    """Save the experts' pooled lens in fp32 with its stage stamp; ``pooled_from``
    records whether the map came from in-memory fp32 sums or a saved (fp16) file."""
    lens = expert_jacobians.pooled_lens(checkpoint_name=checkpoint_name)
    stamp_path = write_stage_stamp(
        args,
        out_path,
        num_prompts_trained_on=expert_jacobians.config.num_prompts_trained_on,
        layers=lens.source_layers,
        pooled_from=pooled_from,
    )
    lens.save(out_path, dtype=t.float32)
    logger.info("pooled lens (%s) saved to %s (stamp %s)", pooled_from, out_path, stamp_path)


def _run_pooled_lens(args: argparse.Namespace) -> None:
    _write_pooled_lens(
        args,
        ExpertJacobians.load(args.experts),
        args.out,
        checkpoint_name=args.checkpoint_name,
        pooled_from="saved experts file (fp16-rounded pooled key)",
    )


def load_items_and_exclusions(
    model: LensModel,
    *,
    eval_data_dir: str,
    correctness_csv: str,
    hf_model_name: str,
    layers: Sequence[int],
    readout_max_seq_len: int,
    items_spec: str = "all",
    fit_fraction: float = DEFAULT_FIT_FRACTION,
) -> tuple[list[ReadoutEvalItem], list[ReadoutEvalItem], list[int]]:
    """``(items, selected_items, excluded_token_ids)``: the eval items (five
    evals, correctness-filtered for ``hf_model_name``), the ``items_spec``
    selection over the scoreable ones (:func:`select_items` with ``fit_fraction``), and the
    Readout Filtering exclusion ids
    (:func:`~lens_evals.readout_evals.readout_eval_items.non_semantic_token_ids` over
    the LM-head width, sparing all six evals' candidates)."""
    items = load_recipe_items(
        data_dir=eval_data_dir, correctness_csv=correctness_csv, hf_model_name=hf_model_name
    )
    runner = readout_runner(model, layers=layers, max_seq_len=readout_max_seq_len)
    selected_items = select_items(
        runner.scoreable_items(items), items_spec, fit_fraction=fit_fraction
    )
    excluded_token_ids = non_semantic_token_ids(
        model.tokenizer,
        vocab_size=vocab_size_of(model),
        items=load_readout_eval_items(eval_data_dir),
    )
    return items, selected_items, excluded_token_ids


def _load_items_and_exclusions(
    args: argparse.Namespace, model: LensModel
) -> tuple[list[ReadoutEvalItem], list[ReadoutEvalItem], list[int]]:
    """:func:`load_items_and_exclusions` with a stage's parsed arguments; with
    ``--no-readout-filter`` the exclusion list is empty."""
    items, selected_items, excluded_token_ids = load_items_and_exclusions(
        model,
        eval_data_dir=args.eval_data_dir,
        correctness_csv=args.correctness_csv,
        hf_model_name=args.hf_model_name,
        layers=args.layers,
        readout_max_seq_len=args.readout_max_seq_len,
        items_spec=args.items,
        fit_fraction=args.fit_fraction,
    )
    if getattr(args, "no_readout_filter", False):
        excluded_token_ids = []
    return items, selected_items, excluded_token_ids


def warn_about_in_sample_items(lens_specs: Sequence[str], items: Sequence[ReadoutEvalItem]) -> None:
    """Warn when a lens's expert weights were fitted on some of the items it is about to be
    scored on, as recorded in the fit-weights stamp beside the lens file (for example after
    fit-weights and evaluate were given different ``--items`` seeds or ``--fit-fraction``)."""
    scored_names = {item.name for item in items}
    for spec in lens_specs:
        stamp_path = stamp_path_beside(spec)
        if spec == LOGIT_LENS_SPEC or not os.path.isfile(stamp_path):
            continue
        stamp = json.loads(Path(stamp_path).read_text())
        if stamp.get("stage") != "fit-weights":
            continue
        in_sample_names = scored_names & set(stamp["fitting_item_names"])
        if in_sample_names:
            logger.warning(
                "%s: %d of the %d scored items were in its expert-weight fitting split, so its "
                "scores on them are in-sample",
                spec,
                len(in_sample_names),
                len(scored_names),
            )


def _check_lens_files_exist(paths: Sequence[str]) -> None:
    """Fail before the model loads when a lens file is missing."""
    missing = [path for path in paths if path != LOGIT_LENS_SPEC and not os.path.isfile(path)]
    if missing:
        raise SystemExit(f"lens files not found: {missing}")


def _run_fit_weights(args: argparse.Namespace) -> None:
    expert_jacobians = ExpertJacobians.load(args.experts)
    model = _load_model(args)
    items, fitting_items, excluded_token_ids = _load_items_and_exclusions(args, model)
    weighting, lens = fit_weights(
        model,
        expert_jacobians,
        items,
        layers=args.layers,
        excluded_token_ids=excluded_token_ids,
        cache_path=args.cache_path,
        checkpoint_name=args.checkpoint_name,
        hf_model_name=args.hf_model_name,
        max_seq_len=args.readout_max_seq_len,
        fitting_item_names=[item.name for item in fitting_items],
        condense_config=CondenseConfig(pass_k=args.k, top_n=args.top_n, steps=args.steps),
        collapse_fallbacks=not args.keep_fallback_copies,
    )
    weights_path = sibling_path(args.out, "_expert_weights.json")
    write_stage_stamp(
        args,
        args.out,
        expert_weights_path=weights_path,
        experts_by_layer={
            str(layer): {key: value for key, value in row.items() if key != "counts"}
            for layer, row in expert_jacobians.layer_summary().items()
        },
        num_excluded_token_ids=len(excluded_token_ids),
        fitting_item_names=[item.name for item in fitting_items],
        train_macro_pass={
            str(layer): learned.train_macro_pass
            for layer, learned in weighting.learned_weights_L_dict.items()
        },
    )
    lens.save(args.out, dtype=t.float32)
    weighting.save(weights_path)
    logger.info("lens saved to %s (weights %s)", args.out, weights_path)


def _write_scores(
    args: argparse.Namespace,
    ranks_df: pd.DataFrame,
    pass_df: pd.DataFrame,
    *,
    num_excluded_token_ids: int,
    **stamp_extra: object,
) -> None:
    """Write the rank rows to ``--out``, the recall table beside them, and the
    stamp; print the macro rows."""
    pass_path = sibling_path(args.out, "_pass_at_k.csv")
    write_stage_stamp(
        args,
        args.out,
        pass_at_k_path=pass_path,
        num_excluded_token_ids=num_excluded_token_ids,
        **stamp_extra,
    )
    ranks_df.to_csv(args.out, index=False)
    pass_df.to_csv(pass_path, index=False)
    logger.info("ranks saved to %s (recall table %s)", args.out, pass_path)
    print(pass_df[pass_df["eval"] == "macro"].to_string(index=False))


def _run_evaluate(args: argparse.Namespace) -> None:
    _check_lens_files_exist(args.lens)
    model = _load_model(args)
    lenses = load_lenses(
        args.lens, model, hf_model_name=args.hf_model_name, layers=args.layers
    )
    _, items, excluded_token_ids = _load_items_and_exclusions(args, model)
    warn_about_in_sample_items(args.lens, items)
    ranks_df, pass_df = evaluate(
        model,
        lenses,
        items,
        layers=args.layers,
        excluded_token_ids=excluded_token_ids,
        ks=args.ks,
        max_seq_len=args.readout_max_seq_len,
    )
    _write_scores(
        args,
        ranks_df,
        pass_df,
        num_excluded_token_ids=len(excluded_token_ids),
        lenses=dict(zip(lenses, args.lens, strict=True)),
        scored_item_names=[item.name for item in items],
    )


def _run_probe_swap(args: argparse.Namespace) -> None:
    _check_lens_files_exist(args.lens)
    model = _load_model(args)
    lenses = load_lenses(args.lens, model, hf_model_name=args.hf_model_name, layers=args.layers)
    trials_df, success_df = probe_swap(
        model,
        lenses,
        layers=args.layers,
        scales=args.scales,
        max_seq_len=args.max_seq_len,
        data_path=args.data,
    )
    best_df = best_scale_rows(success_df)
    success_path = sibling_path(args.out, "_success.csv")
    best_path = sibling_path(args.out, "_best_scale.csv")
    write_stage_stamp(args, args.out, success_path=success_path, best_scale_path=best_path)
    trials_df.to_parquet(args.out, index=False)
    success_df.to_csv(success_path, index=False)
    best_df.to_csv(best_path, index=False)
    logger.info("probe-swap trials saved to %s (success table %s)", args.out, success_path)
    print(best_df.to_string(index=False))


def _run_collect_pairs(args: argparse.Namespace) -> None:
    prompts = (
        load_fit_prompts_from_json(args.prompts_json, args.num_prompts)
        if args.prompts_json is not None
        else load_fit_prompts(args.num_prompts)
    )
    families = [EditFamily.parse(text) for text in args.families.split(",")]
    stats = collect_intervention_pairs(
        _load_model(args),
        prompts,
        families=families,
        layers=args.layers,
        hf_model_name=args.hf_model_name,
        out_dir=args.out_dir,
        target_offset=args.target_offset,
        pairs_per_prompt=args.pairs_per_prompt,
        rows_per_pass=args.rows_per_pass,
        max_seq_len=args.max_seq_len,
        skip_first_n_positions=args.skip_first_n_positions,
        seed=args.seed,
    )
    stamp_path = os.path.join(args.out_dir, f"collect_pairs_{'_'.join(stats['families'])}_stamp.json")
    write_stamp(stamp_path, {"stage": args.stage, "args": vars(args), **stats})
    logger.info("pairs in %s: %s", args.out_dir, stats)


def _fit_loop_kwargs(args: argparse.Namespace) -> dict[str, object]:
    return {
        "steps": args.steps,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "eval_every": args.eval_every,
        "patience": args.patience,
    }


def _write_fitted_lens(
    args: argparse.Namespace, lens: JacobianLens, fits_L_dict: dict[int, TransportFit]
) -> None:
    """The lens in fp32, its per-layer fit history beside it, and the stamp (with the
    held-out errors at the init and at the kept step)."""
    history_path = sibling_path(args.out, "_fit_history.json")
    summary = {
        str(layer): {
            "best_step": fit.best_step,
            "init_heldout": fit.init_heldout_relative_error,
            "best_heldout": fit.best_heldout_relative_error,
            "lstsq_heldout": fit.lstsq_heldout_relative_error,
        }
        for layer, fit in fits_L_dict.items()
    }
    write_stage_stamp(args, args.out, fit_history_path=history_path, heldout=summary)
    lens.save(args.out, dtype=t.float32)
    Path(history_path).write_text(
        json.dumps({str(layer): fit.history for layer, fit in fits_L_dict.items()})
    )
    logger.info("lens saved to %s (history %s)", args.out, history_path)


def _run_fit_intervention(args: argparse.Namespace) -> None:
    lens, fits_L_dict = fit_intervention(
        args.pairs_dir,
        layers=args.layers,
        checkpoint_name=args.checkpoint_name,
        target=args.target,
        loss=args.loss,
        init=args.init,
        holdout_fraction=args.holdout_fraction,
        holdout_seed=args.holdout_seed,
        device=args.device,
        **_fit_loop_kwargs(args),
    )
    _write_fitted_lens(args, lens, fits_L_dict)


def _run_fit_output_kl(args: argparse.Namespace) -> None:
    lens, fits_L_dict = fit_output_kl_lens(
        _load_model(args),
        args.pool_dir,
        layers=args.layers,
        checkpoint_name=args.checkpoint_name,
        init=args.init,
        holdout_fraction=args.holdout_fraction,
        holdout_seed=args.holdout_seed,
        **_fit_loop_kwargs(args),
    )
    _write_fitted_lens(args, lens, fits_L_dict)


def _run_build_experts(args: argparse.Namespace) -> None:
    _check_lens_files_exist(args.lens)
    lenses = [load_lens_file(path, hf_model_name=args.hf_model_name) for path in args.lens]
    if not all(isinstance(lens, JacobianLens) for lens in lenses):
        raise SystemExit("build-experts takes Jacobian-format lenses only")
    expert_jacobians = ExpertJacobians.from_lenses(lenses, layers=args.layers)  # type: ignore[arg-type]
    write_stage_stamp(args, args.out, experts=list(args.lens))
    expert_jacobians.save(args.out)
    logger.info("experts (%d lenses) saved to %s", len(lenses), args.out)


def _wandb_run(args: argparse.Namespace) -> object | None:
    """A W&B run for ``--wandb-project`` (``None`` without it), configured with the stage's
    arguments. Imported lazily: wandb is the optional ``tracking`` extra."""
    if args.wandb_project is None:
        return None
    import wandb

    return wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or args.checkpoint_name,
        config={key: value for key, value in vars(args).items() if not key.startswith("wandb")},
    )


def _run_fit_swap(args: argparse.Namespace) -> None:
    run = _wandb_run(args)
    if args.mode == "supervised":
        prompts: list[str] = []  # the rows come from --swap-data
    elif args.prompts_json is not None:
        prompts = load_fit_prompts_from_json(args.prompts_json, args.num_prompts)
    else:
        prompts = load_fit_prompts(args.num_prompts)
    lens, fit = fit_swap(
        _load_model(args),
        prompts,
        layers=args.layers,
        hf_model_name=args.hf_model_name,
        checkpoint_name=args.checkpoint_name,
        mode=args.mode,
        init=args.init,
        examples_per_prompt=args.examples_per_prompt,
        top_k=args.top_k,
        window=args.window,
        max_seq_len=args.max_seq_len,
        skip_first_n_positions=args.skip_first_n_positions,
        holdout_fraction=args.holdout_fraction,
        holdout_seed=args.holdout_seed,
        num_heldout_examples=args.heldout_examples,
        seed=args.seed,
        log_fn=(lambda metrics: run.log(metrics, step=int(metrics["step"]))) if run is not None else None,
        swap_items=json.loads(Path(args.swap_data).read_text())["items"] if args.swap_data else None,
        monitor_items=json.loads(Path(args.monitor_data).read_text())["items"] if args.monitor_data else None,
        scale=args.scale,
        steps=args.steps,
        batch_size=args.batch_size,
        length_pool=args.length_pool,
        clean_cache_gb=args.clean_cache_gb,
        edit_dtype=t.float32 if args.edit_precision == "fp32" else None,
        relative_lr=args.relative_lr,
        eval_every=args.eval_every,
        patience=args.patience,
        window_sizes=args.window_sizes,
        all_layers_probability=args.all_layers_probability,
    )
    history_path = sibling_path(args.out, "_fit_history.json")
    write_stage_stamp(
        args,
        args.out,
        fit_history_path=history_path,
        best_step=fit.best_step,
        heldout_loss={"init": fit.init_heldout_loss, "best": fit.best_heldout_loss},
        heldout_hit_rate={"init": fit.init_heldout_hit_rate, "best": fit.best_heldout_hit_rate},
    )
    lens.save(args.out, dtype=t.float32)
    Path(history_path).write_text(json.dumps(fit.history))
    logger.info("lens saved to %s (history %s)", args.out, history_path)
    if run is not None:
        run.summary.update(
            {
                "best_step": fit.best_step,
                "best_heldout_loss": fit.best_heldout_loss,
                "best_heldout_hit_rate": fit.best_heldout_hit_rate,
                "lens_path": args.out,
            }
        )
        run.finish()


STAGE_HANDLERS = {
    "fit-router": _run_fit_router,
    "fit-shard": _run_fit_shard,
    "merge-experts": _run_merge_experts,
    "pooled-lens": _run_pooled_lens,
    "fit-weights": _run_fit_weights,
    "evaluate": _run_evaluate,
    "probe-swap": _run_probe_swap,
    "collect-pairs": _run_collect_pairs,
    "fit-intervention": _run_fit_intervention,
    "fit-output-kl": _run_fit_output_kl,
    "build-experts": _run_build_experts,
    "fit-swap": _run_fit_swap,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    STAGE_HANDLERS[args.stage](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
