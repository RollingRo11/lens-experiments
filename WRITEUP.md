# Experiments on Gemma 4 E2B: clustering choices and trained lens maps

This note describes what we tried on top of the J++ Lens code in this repository, and what came
out. Everything was run on one model, `google/gemma-4-E2B`, between 2026-10-10 and 2026-10-13.
The code for every experiment is in this repository (see [Where things are](#where-things-are)).
Results, lens files and plots are not in git; they live on the cluster.

## Contents

- [Short version](#short-version)
- [Background](#background): what a lens is, and the lenses we compare
- [Setup and how to read the numbers](#setup-and-how-to-read-the-numbers)
- [A scoring bug we found and fixed](#a-scoring-bug-we-found-and-fixed)
- [Part 1. What Gemma's activations look like](#part-1-what-gemmas-activations-look-like)
- [Part 2. Which clustering to use for J++](#part-2-which-clustering-to-use-for-j)
- [Part 3. Learning the lens map by gradient descent](#part-3-learning-the-lens-map-by-gradient-descent)
- [Part 4. Training the map so that swaps work](#part-4-training-the-map-so-that-swaps-work)
- [Part 5. A second set of questions (WorkspaceBench)](#part-5-a-second-set-of-questions-workspacebench)
- [Facts about Gemma 4 E2B that matter here](#facts-about-gemma-4-e2b-that-matter-here)
- [What is not done or not checked](#what-is-not-done-or-not-checked)
- [Where things are](#where-things-are)
- [Names used in this note and in the code](#names-used-in-this-note-and-in-the-code)
- [References](#references)

## Short version

1. **Which clustering J++ uses matters less than it first looked.** Swapping k-means (8 clusters)
   for a density-based clustering (ADP, 191 clusters) moved recall@10 from 32 to 38. But the
   control with the cluster labels shuffled at random also got 33, and a bootstrap over the
   questions puts every gap between clusterings inside the noise. What clearly helps is the J++
   recipe itself over a single averaged Jacobian (32 vs 28), and any Jacobian lens over the logit
   lens (28 vs 13).
2. **Choosing a cluster per activation at read time ("routing") is worse than J++'s one fixed
   mix**, and worse than a single averaged Jacobian. This matches what the J++ post reports.
3. **Fitting the lens map by regression on (edit, effect) pairs did not beat the Jacobian.** The
   version trained on swap edits read out almost nothing. We found why: those edits never contain
   the average residual vector, so the map never learns what to do with it.
4. **Training the map through the J-lens paper's own swap test worked for swapping, not for
   reading.** On 24 relations it never saw, swapping along the trained lens's directions changed
   Gemma's answer to the right new answer 42% of the time (all layers), against 3% for the J-lens
   and 6% for the logit lens. Its recall@10 is 13, the same as the logit lens. The editing power
   sits in Gemma's late layers (25 and up).
5. **We found and fixed a scoring bug** that gave too much credit to lenses with large outputs.
   Every recall@10 in this note uses the fixed scoring.

## Background

**Residual stream.** A transformer keeps one running vector per token position, the residual
stream (Elhage et al., 2021). Each layer reads it and adds its output back in. For Gemma 4 E2B
the vector has 1,536 numbers. After the last layer, a final normalisation and the unembedding
matrix turn it into one score (a logit) per vocabulary token, about 262,000 of them.

**Lens.** A lens reads an intermediate layer's residual vector as a ranked list of tokens. This
lets you see what the model is "thinking" mid-computation, for example the bridge entity "Brazil"
in "the language spoken in the country where the Amazon River ends is ___".

The lenses compared in this note:

| Lens | What it does | Source |
|---|---|---|
| Logit lens | Apply the final norm and unembedding directly to the layer's vector. | nostalgebraist (2020) |
| Tuned Lens | Learn one map per layer so the readout matches the model's own final output. | Belrose et al. (2023) |
| J-lens (Jacobian lens) | Multiply the vector by the layer's *Jacobian* first, then unembed. | Gurnee et al. (2026) |
| R-Lens | The J-lens, but the Jacobian is computed through a Layer-wise Relevance Propagation (LRP) backward pass. | Blank and Bhatia; LRP from Bach et al. (2015) |
| J++ Lens | Split the activations into clusters, compute one Jacobian per cluster, and mix them with learned weights. | Ayonrinde and Lindsey (2026) |

Short reminders on the terms in that table:

- **Jacobian.** For a function from vectors to vectors, the Jacobian is the matrix of slopes:
  entry (i, j) says how much output i moves when input j moves a little. It is the best straight-line
  stand-in for the function near one point. The J-lens uses the Jacobian of the final layer's
  vector with respect to the layer-l vector, averaged over many prompts and positions. Gurnee et
  al. call the reading a vector's "first-order causal effect on the model's outputs".
- **LRP backward pass.** A different rule for passing gradients back through norms and gated MLPs.
  The J++ post uses it because it stops noise building up through those parts.
- **Expert Jacobians and Jacobian Filtering (J++).** The J++ post clusters each layer's
  activations with k-means into E clusters, which it calls *experts per layer*. It averages one
  Jacobian per cluster (an *expert Jacobian*) and then learns one signed weight per expert, so
  experts that give poor readouts get low or negative weight. It calls this weighting *Jacobian
  Filtering*. The weights are learned on a small labelled set of questions. At read time the
  result is still one fixed matrix per layer. The single average over all activations is the
  post's *Pooled Jacobian*; computed with LRP, that is the R-Lens.
- **Readout Filtering (J++).** Tokens with no letter or digit are left out of the ranking.

**How a lens is scored.**

- **Recall@10.** The readout questions come from the J-lens paper's repository: association,
  multihop, multilingual, poetry and typo. Each question has one or more hidden intermediate words
  the model should be using but does not say (multilingual has several per question). A word
  counts as found if it is among the lens's 10 top tokens at any of the layers read. A question
  scores the share of its words found. Recall@10 is the average over the questions of a task,
  then over the 5 tasks. (The code calls it `pass_at_k`.)
- **Probe-swap.** A test from the J-lens paper. Take a two-hop prompt like "the language spoken in
  the country where the Amazon River ends is". At every layer of the *workspace band* (the J-lens
  paper's name for the middle layers) and every position, push the residual vector along the
  lens's direction for "Brazil" and for "Mexico" so that their two coordinates trade places. If
  the model now says "Spanish", the lens's directions are the ones the model really uses.
- **Swap success.** The same edit as probe-swap, scored the same way, on our own larger sets of
  templated prompts. It is the share of swaps where the model's top next word becomes the answer
  for the swapped-in word. This is close to what the causal abstraction literature calls
  interchange intervention accuracy (Geiger et al., 2021), and RAVEL (Huang et al., 2024) uses the
  same kind of test.

## Setup and how to read the numbers

- **Model:** `google/gemma-4-E2B` (a base model, not chat-tuned), 35 layers, residual width 1,536.
- **Layers read:** 4, 8, 12, 16, 20, 24, 28, unless a section says otherwise.
- **Fitting text:** WikiText-103 (Merity et al., 2016), cut to 128 tokens. 1,024 prompts for the
  Jacobians (R-Lens and J++). The clustering used 1,000 prompts, from position 16 to the
  second-last token, which gives 110,804 activation vectors per layer.
- **Plain J-lens:** fitted with ordinary gradients on 64 prompts, the size the J++ post uses in
  its Table 5. It is the reference for Parts 3 and 4.
- **Questions:** only those Gemma itself answers correctly (graded with the prompt's trailing space
  removed; Gemma reads a trailing space as its own token). Typo and association are always kept.
  20% of each task (about 69 questions in total) is used to fit J++'s weights. The other 80% is
  scored. Scored questions per task: association 82, multihop 12, multilingual 47, poetry 68,
  typo 77. Every lens is scored on the same split.
- **Hardware:** NVIDIA A100 and H200 GPUs on Northeastern's Explorer cluster.

**How big a difference is real.** The sample is small. One multihop question moves the average
by 1.7 points. The same lens scored on an H200 and an A100 differed by about 2 points before the
scoring fix. J++'s weights are fitted on about 69 questions and the best weights are picked on
those same questions. So treat gaps of 2 to 3 points as noise.

A bootstrap (resampling the scored questions within each task, 4,000 times) puts it more
precisely. The 95% interval on a single lens's recall@10 is about ±5 points. For the difference
between two lenses on the same questions:

| Difference | Gap | 95% interval |
|---|---|---|
| J++ with ADP minus J++ with k-means 8 | 5.7 | −1.9 to 13.2 |
| J++ with ADP minus J++ with ADP labels shuffled | 5.4 | −1.0 to 11.9 |
| J++ with k-means 8 minus R-Lens | 4.2 | 1.0 to 8.3 |
| R-Lens minus logit lens | 15.6 | 13.0 to 18.3 |

Only the last two gaps are clearly above zero. The bootstrap does not include the noise from
fitting J++'s weights, so the real intervals are wider.

## A scoring bug we found and fixed

Gemma squashes its output scores with a soft cap, 30 · tanh(score / 30) (Gemma Team, 2024). The
readout eval applied that cap in bf16, a 16-bit format that keeps about 3 significant digits. So
every score above roughly 100 came out as exactly 30.0. A lens with large outputs could have
hundreds or thousands of tokens tied at 30.0. The ranking code then gave every tie to the right
answer: "no token strictly higher" counted as rank 1. By that rule, a lens that gave every token
the top score would get 100%.

The fix (2026-10-13) ranks fp32 scores taken before the cap. The cap never changes which token is
higher, so this is the true ranking and nothing ties. The change is in `ranking_logits` and
`masked_lens_logits` in `src/workspace_lens/utils.py`, with a test.

What it changed (recall@10, old → fixed):

| Lens | Old | Fixed |
|---|---|---|
| J++ with ADP clusters | 38 | 38 |
| J++ with k-means 8 | 33 | 32 |
| R-Lens | 29 | 28 |
| Plain J-lens | 34 | 25 |
| Logit lens | 15 | 13 |
| Swap-trained lenses (Part 4), four versions | 24 to 69 | 14 to 19 |

The J++ and R-Lens maps barely tie, so their numbers hardly moved. The plain J-lens ties at layer
28 in about 80% of readouts, and the swap-trained lenses tied even more. Swap success and
probe-swap are measured on the model's own output after the edit, not on the lens's scores, so
the bug never touched them.

**Every recall@10 in this note is from the fixed scoring.** Where a lens was never rescored, the
note says so. In every lens we did rescore, the fixed score was equal to or lower than the old one.

One place still uses the old scoring: the step that fits J++'s expert weights (`fit-weights`).
J++'s maps barely tie, so we left it, but it is not fixed.

## Part 1. What Gemma's activations look like

Before changing the clustering, we looked at the activation vectors J++ clusters (110,804 per
layer; each scaled to length 1, as J++'s router does).

- **k-means loss falls smoothly as the number of clusters grows** (K = 1 to 64, k-means++ start,
  in the router's 64 PCA dimensions). There is no elbow at 8 or anywhere else, so the data do not
  point to a natural number of clusters.
- **PCA:** the router's 64 directions keep only about 40 to 55% of the variance, depending on the
  layer.
- **Intrinsic dimension** (TwoNN, Facco et al., 2017, which compares each point's distance to its
  nearest and second-nearest neighbour) is low, roughly 4 to 15 depending on the layer, with a peak
  at layer 12. The estimate rises when fewer points are used, so it depends on scale. Treat it as
  rough.
- **2D and 3D maps** (t-SNE, van der Maaten and Hinton, 2008; UMAP, McInnes et al., 2018, tuned per
  layer on trustworthiness, Venna and Kaski, 2001, and distance correlation) show a large central
  mass plus small islands. The islands are mostly WikiText formatting tokens (its escaped
  `@-@`, `@,@`, `@.@`, the split `'` + `s`, and ` .`). An interactive 3D page with k-means, ADP and
  HDBSCAN colourings was made but is not in git.

## Part 2. Which clustering to use for J++

The J++ post clusters with k-means into 8 clusters per layer. We tried other ways to make the
clusters, keeping everything else in the J++ recipe the same (LRP Jacobians on 1,024 prompts,
weights fitted on the same questions, Readout Filtering on).

The other clusterings ran on 20,000 random activations per layer, in the full 1,536 dimensions
(no PCA). A fitting token takes the cluster of its nearest point among those 20,000.

- **ADP** (Advanced Density Peaks; d'Errico et al., 2021, in the DADApy library, Glielmo et al.,
  2022). It finds clusters as peaks of the point density and picks their number itself. With its
  default setting (Z = 1.65) it found 191, 85, 134, 3, 14, 64 and 78 clusters at the 7 layers.
  Clusters with fewer than 50 fitting tokens fall back to the R-Lens Jacobian, as J++'s code does.
- **k-means with ADP's counts:** the same points, with K per layer set to ADP's count. This asks
  whether ADP helps because of where it draws the lines or just because it makes more clusters.
- **ADP top 7 + rest:** ADP's 7 largest clusters, everything else in an 8th. Same count as the
  original.
- **ADP, labels shuffled:** ADP's labels randomly permuted over the points. Same cluster sizes,
  random members. The weights are still fitted. This is the "random clusters" control.
- **HDBSCAN** (Campello et al., 2013; McInnes et al., 2017), with the setting closest to 8
  clusters (2 to 11 per layer). HDBSCAN marks some points as noise; tokens nearest a noise point
  were left out of every expert (37 to 85% of fitting tokens).

Recall@10 (fixed scoring):

| Lens | Average | Typo | Multilingual | Multihop | Association | Poetry |
|---|---|---|---|---|---|---|
| J++, ADP (191 and others) | **38** | 90 | 58 | 33 | 10 | 0 |
| J++, k-means with ADP's counts | 36 | 94 | 57 | 17 | 7 | 4 |
| J++, HDBSCAN | 34 | 78 | 44 | 42 | 7 | 0 |
| J++, ADP top 7 + rest | 34 | 91 | 46 | 25 | 6 | 1 |
| J++, ADP labels shuffled | 33 | 94 | 47 | 17 | 6 | 0 |
| J++, k-means 8 (the post's recipe) | 32 | 90 | 41 | 25 | 6 | 0 |
| R-Lens (one pooled Jacobian) | 28 | 75 | 42 | 17 | 7 | 0 |
| Logit lens | 13 | 21 | 24 | 17 | 1 | 0 |

Scored questions: typo 77, multilingual 47, multihop 12, association 82, poetry 68.

What this shows, with the noise from the setup section in mind:

- Every J++ variant beats the R-Lens, and the R-Lens beats the logit lens by a wide margin.
- ADP is on top, but the shuffled-label control (33) is as good as k-means 8 (32), and the
  bootstrap interval for ADP minus shuffled crosses zero. So we cannot say ADP's cluster
  boundaries help. More clusters plus the weight fitting may be what matters, and even that is not
  clearly above noise.
- Poetry is near 0 for every lens. Association is low for every lens.
- Cost: the 191-expert fit took 5 h 51 min on an A100, and its weight fitting ran out of memory on
  a 40 GB A100 (it needs an H200). The 8-cluster fit takes about 1 h on one H200.

### Routing each activation to its own cluster

J++ uses one fixed mix of experts for every activation. We also tried choosing experts per
activation at read time: each activation uses the experts of its k nearest of the 20,000 labelled
points, weighted by how many fall in each cluster (k = 1 or 20), with no learned weights. The J++
post calls this *k-means routing* and reports that it does worse.

On Gemma it also does worse: 21 to 28 average across the clusterings and k, against 28 for the
R-Lens and 32 to 38 for the fixed J++ mixes. Routing loses most on multilingual and association.

## Part 3. Learning the lens map by gradient descent

The J-lens averages a derivative. We asked whether we could instead *learn* the per-layer matrix
directly, by gradient descent. Both lens papers argue against the most literal version, which is
the Tuned Lens: matching the model's own final output teaches the map what the vector *predicts*,
not what it *does*, and the J-lens paper finds it "skips ahead" to the output.

So we trained on the effect of edits instead. For each training pair: change the layer-l vector at
one position by some edit δ, run the rest of the model, and record y, the summed change in the
final layer's vector at that position and all later ones (the same structure as the J-lens
estimator). Then fit one 1,536 × 1,536 matrix A per layer so that A δ ≈ y, with Adam on a squared
error. For tiny random edits, the best A is the Jacobian, so this is a check that the pipeline is
right. For bigger, realistic edits, A could capture effects the Jacobian misses (Nanda's notes on
attribution patching describe where the straight-line approximation breaks).

Edit types, each collected on the same 1,024 WikiText prompts (32 pairs per prompt and layer):

- **Gaussian:** small random noise, 5% of the vector's length.
- **Swap:** replace the vector with one from a random other prompt, δ = α (h′ − h), α = 1, 0.5 or
  0.25. This is the main idea.
- **Removal:** δ = −h (zero the vector out).
- **Tuned-Lens-style:** for comparison, a map fitted to match the model's own next-token
  distribution (KL divergence), with no edits.

We also tried the J++ weighting step with these maps as the "experts" (one weight per map per
layer, fitted on the same ~69 questions as J++).

Results, recall@10:

| Map | Fixed scoring | Old scoring |
|---|---|---|
| Mix of maps via J++ weighting (swap + removal, swap α = 0.5, Gaussian, R-Lens) | 34 | 40 |
| Plain J-lens (reference) | 25 | 34 |
| Tuned-Lens-style | 21 | 21 |
| Gaussian edits | 16 | 24 |
| Swap + removal edits | 13 | 13 |
| Logit lens (reference) | 13 | 15 |
| Swap edits only, α = 1 | not rescored | 2 |
| Swap, α = 0.5 and 0.25 | not rescored | 1 and 0 |
| Swap, loss measured in logit space | not rescored | 2 |
| Swap, started from the R-Lens | not rescored | 19 |

What we learned:

- **The Gaussian check passed where it could.** The Gaussian map matched the plain J-lens at layer
  28 (entry-wise correlation 0.96) and roughly at layers 20 to 24 (0.84 to 0.87), but not at
  layers 4 to 16 (0.06 to 0.57). That is a property of Gemma, not a bug: in fp32, tiny edits match
  the J-lens's prediction to about 1% at layers 16 to 28, but at layers 4 to 12 even a 0.2% edit
  is 35 to 110% off. Early layers are far from straight-line at any edit size we can use.
- **Swap-only maps failed, and we found why.** A swap edit is a difference of two vectors,
  h′ − h. The part every vector shares (the average residual vector) cancels out, so the map is
  never told what to do with it. The trained maps blew the average part up 3 to 10 times relative
  to the part that varies, so every position read out the same tokens. Subtracting the average
  before reading lifted swap α = 1 from 2 to 18 (old scoring), while barely moving the J-lenses.
- **Adding removal edits** (which do contain the average) fixed the collapse, but the result (13)
  only matches the logit lens.
- **The mix (34)** gets above J++ k-means 8 (32) on the fixed scoring, but its weights are fitted on
  the same questions as J++, it mostly picks the R-Lens at layers 4 to 12 and the Gaussian map at
  24 to 28, and the gap is inside the noise. It is J++'s weighting step with different experts,
  not a new loss.

## Part 4. Training the map so that swaps work

Since regression on edits did not help, we trained the per-layer matrices through the J-lens
paper's swap test itself.

**How it works.** The lens is one 1,536 × 1,536 matrix A per layer, with no bias and no
non-linearity, so it is used exactly like a J-lens. A token's lens direction at a layer is its
unembedding row times A, scaled to length 1. A training example is a templated prompt with a
known answer, for example "The capital of the country that contains {Texas} is", plus another
word of the same kind, for example "Bavaria". The training step:

1. Run Gemma on the prompt (the clean run).
2. Run it again, and at the chosen layers and every position but the first, swap the vector's
   coordinates along the "Texas" and "Bavaria" lens directions. This is probe-swap's edit.
3. The loss is the cross-entropy of the swapped-in word's answer ("Berlin") at the last position.
   Only word tokens count, as in probe-swap's scoring.
4. Gradients flow back through the frozen Gemma into every layer's matrix at once.

Gemma itself is never changed. Only the lens matrices are trained.

**Data.** We built three datasets, keeping only single-token swap words and facts Gemma already
gets right, and holding out whole relations (not just pairs) for testing. Words used by probe-swap
and the readout questions were excluded from training.

| Version | Training swaps | Relations | Distinct swap words | Held out | Sources |
|---|---|---|---|---|---|
| Small | 3,040 | 19 | 229 | 6 templates (612 swaps; 400 scored) | Hand-written templates (capitals, languages, animals, antonyms, plurals, element symbols, ...) |
| Medium | 6,630 | 46 | 855 | 14 relations (600 swaps; 423 scored) | Hand-written, RAVEL (Huang et al., 2024), LRE relations (Hernandez et al., 2024) |
| Large | 71,280 | 96 | 3,328 | 24 relations (1,500 swaps) | The above plus CounterFact (Meng et al., 2022), PARAREL (Elazar et al., 2021), MUSE bilingual dictionaries (Conneau et al., 2018), WordNet (Miller, 1995) |

**Training settings.** Adam, step size 1% of each matrix's typical entry size, cosine decay to
zero, no weight decay. Started from the identity matrix (the logit lens) or from the plain J-lens.

- Small and medium: matrices at the 7 usual layers, all swapped together. Batch 16. 2,000 steps
  (small, with early stopping) or 6,000 steps (medium).
- Large: one matrix for every layer, 0 to 33, started from the identity. Half the steps swap at
  all layers; the other half swap at one random window of 1, 3 or 5 neighbouring layers with the
  rest switched off, so each matrix also has to work on its own. Batch 256. Trained for 4,000
  steps, then again from scratch for 16,000 steps because the first run had not settled. Training
  curves: W&B runs [swapall_v3](https://wandb.ai/rohan-kathuria-neu/lens-training/runs/lv8vlk9h)
  and [swapall_v3_16000](https://wandb.ai/rohan-kathuria-neu/lens-training/runs/w8cqoeqo) (the
  W&B project may not be public).

### Results: swapping

Swap success (top-1) on relations never seen in training, large dataset, 1,500 swaps:

| Layers swapped | Large, 16k steps | Large, 4k steps | Plain J-lens | Logit lens |
|---|---|---|---|---|
| All layers (0 to 33) | 41.5% | 41.3% | 3.3% | 5.7% |
| Workspace band (8 to 26) | 38.3% | 37.9% | 1.8% | 1.1% |
| Best window of 5 | 46.1% (L26 to 30) | 45.2% (L25 to 29) | 4.5% | 13.5% |
| Best window of 3 | 43.1% (L28 to 30) | 38.7% (L28 to 30) | 5.1% | 11.7% |
| Best single layer | 32.9% (L31) | 25.8% (L29) | 5.1% | 11.3% |
| Best single layer below 23 | 4.9% | 3.9% | 1.1% | 1.3% |

All lenses here are swapped at scale 1. On the same test, an earlier all-layer lens trained the
same way on the medium dataset (6,000 steps) got 24.0% at all layers and 17.4% over the band.

The plain J-lens here was fitted at all 34 layers on 64 prompts, so it can be swapped at the same
layers.

- **The editing power is in the late layers.** Every lens, trained or not, does almost nothing
  when swapped at a single layer below 23.
- **The longer run mostly helped single layers and small windows.** All-layer swapping did not
  move. During training, success on the unseen relations peaked around step 2,000 (about 43%) and
  slid to about 40% by the end, while success on held-out pairs from the training relations stayed
  near 98%. Training loss was flat after about step 10,000.

Earlier, smaller datasets (7 layers, all swapped together):

| Lens | Small set, 400 swaps | Medium set, 423 swaps |
|---|---|---|
| Trained on small set, from identity | 16% | 13% |
| Trained on small set, from J-lens | 2% | 8% |
| Trained on medium set, from identity | | 18% |
| Trained on medium set, from J-lens | | 14% |
| Plain J-lens | 11% | 9% |
| R-Lens | 4% | 5% |
| J++ k-means 8 | 4% | 4% |
| Logit lens | 0% | 1% |

Each lens is scored at its best swap scale. More distinct swap words helped. On the small set, the
version started from the J-lens got worse on new relations than the J-lens it started from; with
the medium set it no longer did. Our reading
of why (from the maths of the setup, not measured): each swap only moves A along the two swapped
words' unembedding rows, so a matrix trained on about 200 words only changes along about 200 of its
1,536 input directions.

**Probe-swap** (the J-lens paper's 90 two-hop prompts; only 33 to 35 can be scored on Gemma, because
the eval drops items Gemma gets wrong to begin with and items whose words are not single tokens):

- At the 7 usual layers, every untrained lens gets at most 2 of 35 right (top-1). The two lenses
  trained on the medium set get 9 of 33.
- Over the workspace band (layers 8 to 26), the large-set lenses (4k and 16k steps) and the earlier
  all-layer lens trained on the medium set all get 7 of 33 (21%), against 1 of 33 for the all-layer
  plain J-lens and the logit lens.
- Probe-swap's domains (countries, languages, and so on) overlap our training data, though none of
  its words were trained on. Treat these gains as suggestive.

### Results: reading

Recall@10 (fixed scoring, 7 usual layers):

| Lens | Average | Multihop (12 questions) | Multilingual | Typo |
|---|---|---|---|---|
| J++ k-means 8 (reference) | 32 | 25 | 41 | 90 |
| Plain J-lens (reference) | 25 | 25 | 33 | 61 |
| Small set, from identity | 19 | 42 | 30 | 22 |
| Small set, from J-lens | 17 | 33 | 29 | 22 |
| Medium set, from identity | 16 | 33 | 28 | 18 |
| Medium set, from J-lens | 14 | 42 | 21 | 5 |
| Large set, 16k steps | 13 | 33 | 19 | 13 |
| Large set, 4k steps | 10 | 25 | 15 | 8 |
| Logit lens (reference) | 13 | 17 | 24 | 21 |

**Our reading:** training through the swap buys real power to *edit* the model, mostly in the
late layers, on relations it never saw. It does not buy better *readouts*. The trained lenses read
out about as well as the logit lens, well below the Jacobian lenses. The one task where they look
good, multihop, has only 12 questions.

### Training speed

- A Jacobian fit (R-Lens or J++, 1,024 prompts) takes about 1 h on an H200 or about 3 h on an A100,
  because each prompt needs about 192 backward passes. Training the small-set swap lens took about
  6 minutes.
- The first large-set run took 0.93 s per step (batch 256). Profiling showed 63% of every batch was
  padding (prompts are about 9 to 11 tokens, padded to the longest in the batch), and the clean
  run, which never changes, was redone every step (about 20% of a step).
- Fixes: batches grouped by prompt length, an optional stored clean run, and leaner edit maths.
  In profiling the step fell to 0.31 s. In the real 16k-step run it was 0.45 s per step (2 h in
  total), because the stored clean run (29 GB) does not fit next to training on an 80 GB A100.

## Part 5. A second set of questions (WorkspaceBench)

WorkspaceBench ([github.com/camilablank/workspace-bench](https://github.com/camilablank/workspace-bench))
scores lenses with an LLM judge. Its 5 single-token families have the same format as our readout
questions, so we scored those with our own recall@10, with no judge. We kept only questions Gemma
answers correctly and dropped any that already appear in our data, so J++'s weights never saw them.
Questions scored: association 84, multihop 31, multilingual 48, poetry 83, typo 82.

| Lens | Average | Association | Multihop | Multilingual | Typo |
|---|---|---|---|---|---|
| J++, ADP | 54 | 22 | 60 | 98 | 89 |
| J++, k-means 8 | 52 | 15 | 58 | 98 | 89 |
| R-Lens | 49 | 23 | 53 | 94 | 77 |
| Plain J-lens | 45 | 17 | 60 | 83 | 65 |
| Swap-trained, medium set, from identity | 41 | 11 | 77 | 94 | 21 |
| Swap-trained, small set, from J-lens / from identity | 37 / 36 | 2 / 2 | 71 / 71 | 85 / 85 | 24 / 23 |
| Swap-trained, medium set, from J-lens | 31 | 11 | 69 | 69 | 7 |
| Logit lens | 27 | 2 | 31 | 81 | 22 |

Poetry is 0 for every lens; the average still counts it, as in our own evals. Other lenses: the
other J++ clusterings scored 49 to 52, the Part 3 mix 50, the Tuned-Lens-style map 37 and the
Gaussian map 30. The large-set lenses were not scored here. The ordering matches our own
questions: J++ leads, and the swap-trained lenses are best on multihop bridges but weak on typo and
association.

We also set up the LLM-judged version with a local judge (Qwen3.6-27B served by vLLM) instead of the
paid Gemini judge. That job failed when vLLM would not start, and it has not been rerun, so there
are no judged results. The chat-format families could not run at all: Gemma 4 E2B is a base model
and its tokenizer has no chat template.

## Facts about Gemma 4 E2B that matter here

- Layers 15 to 34 reuse the attention keys and values of layers 13 and 14. So an edit at layer 15
  or later cannot reach other positions; it only affects its own position's later layers.
- Each block multiplies the whole residual vector by a per-layer scalar (`layer_scalar`). The final
  vector's change is only about 0.06 to 0.17 times an edit made at layers 16 to 28. So the identity
  matrix is a poor stand-in for how edits there reach the output.
- bf16 rounding swamps edits smaller than about 5% of the vector's length.
- Early layers (4 to 12) are far from straight-line, even for very small edits (see Part 3).
- Gemma gets none of the poetry questions right when the prompt ends in a space, because it reads
  the space as its own token and then predicts markup. Grading with the trailing space removed
  fixes this (`scripts/grade_correctness.py`).

## What is not done or not checked

- J++'s weight fitting (`fit-weights`) still ranks with the old capped bf16 scores.
- Several Part 3 maps were never rescored with the fixed scoring (marked in the table).
- The LLM-judged WorkspaceBench run failed and has not been rerun.
- The large-set swap lenses were not scored on WorkspaceBench.
- All results are on one model. Nothing here was checked on Qwen3.6-27B, the J++ post's model.
- Only one seed per fit. The bootstrap intervals above cover question sampling only.

## Where things are

Code in this repository:

| What | Where |
|---|---|
| Learning maps from (edit, effect) pairs (Part 3) | `src/workspace_lens/fitting/intervention_fitting.py`; CLI stages `collect-pairs`, `fit-intervention`, `fit-output-kl`, `build-experts` |
| Training maps through the swap edit (Part 4) | `src/workspace_lens/fitting/swap_fitting.py`; CLI stage `fit-swap` |
| The scoring fix | `ranking_logits` and `masked_lens_logits` in `src/workspace_lens/utils.py` |
| Clustering by nearest labelled point (ADP, HDBSCAN) | `centroid_labels_C` in `src/workspace_lens/routing/router.py` |
| Whole lenses as J++ experts (the Part 3 mix) | `ExpertJacobians.from_lenses` in `src/workspace_lens/fitting/condense_experts/expert_jacobians.py` |
| Grading which questions Gemma gets right | `scripts/grade_correctness.py` |
| Part 1 scripts and SLURM jobs | `experiments/kmeans_k_sweep/`, `experiments/dimensionality/` |
| Part 2 scripts and SLURM jobs | `experiments/adp_router/`, `experiments/routed_lens/` |
| Parts 3 and 4 scripts, dataset builders and SLURM jobs | `experiments/intervention_lens/` |
| Part 5 scripts and SLURM jobs | `experiments/workspace_bench/` |

Tests: `uv run pytest -q` (291 pass on CPU).

Results, lens files and saved activations are on the Explorer cluster under
`/projects/RohanKathuria/jpp_lens/artifacts/gemma4_e2b/`. The fixed-scoring tables are
`eval_held_out_all_exact_pass_at_k.csv` (Part 2), `intervention/eval_held_out_*exact*_pass_at_k.csv`
(Parts 3 and 4), and `workspace_bench/eval_all_lenses_exact_pass_at_k.csv` (Part 5). Training
curves are in the W&B project [lens-training](https://wandb.ai/rohan-kathuria-neu/lens-training).

## Names used in this note and in the code

| In this note | In the code or file names | Literature term |
|---|---|---|
| J++ (k-means 8) | `jpp_lens.pt`, `jpp_lens_v2` | J++ Lens (Ayonrinde and Lindsey) |
| Number of clusters | `num_clusters`, `K` | experts per layer, E (J++ post) |
| Weight fitting | `fit-weights`, `ExpertWeighting` | Jacobian Filtering (J++ post) |
| R-Lens | `pooled_lens` | Pooled Jacobian with LRP (J++ post); R-Lens (Blank and Bhatia) |
| Plain J-lens | `plain_jlens64` | J-lens (Gurnee et al.) |
| Routing per activation | `routed_*` | k-means routing (J++ post) |
| Recall@10 | `pass_at_k` | recall@k (J++ post); pass@k (J-lens repo) |
| Layers 8 to 26 | `band` | workspace band (J-lens paper) |
| Maps from (edit, effect) pairs | `ilens_*` | no standard name |
| Swap-trained lens | `swapdata_*` (small), `swapdata2_*` (medium), `swapall_identity` (medium, all layers), `swapall_v3*` (large) | no standard name |
| Swap success | `hit_rate`, `monitor_hit_rate` | close to interchange intervention accuracy (Geiger et al.) |

## References

- Arthur and Vassilvitskii (2007). k-means++: the advantages of careful seeding.
- Ayonrinde and Lindsey (2026). J++ Lens: Jacobian Filtering Enables More Faithful Workspace
  Lenses. https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr
- Bach et al. (2015). On pixel-wise explanations for non-linear classifier decisions by
  layer-wise relevance propagation. PLOS ONE.
- Belrose et al. (2023). Eliciting latent predictions from transformers with the tuned lens.
  https://arxiv.org/abs/2303.08112
- Blank and Bhatia. R-Lens (relevance propagation Jacobian lens).
  https://huggingface.co/camilablank/workspace-lenses
- Campello, Moulavi and Sander (2013). Density-based clustering based on hierarchical density
  estimates (HDBSCAN). McInnes, Healy and Astels (2017). hdbscan: hierarchical density based
  clustering. JOSS.
- Conneau et al. (2018). Word translation without parallel data (MUSE).
- d'Errico, Facco, Laio and Rodriguez (2021). Automatic topography of high-dimensional data sets
  by non-parametric density peak clustering (ADP). Information Sciences.
- Elazar et al. (2021). Measuring and improving consistency in pretrained language models
  (PARAREL). TACL.
- Elhage et al. (2021). A mathematical framework for transformer circuits.
  https://transformer-circuits.pub/2021/framework/index.html
- Facco, d'Errico, Rodriguez and Laio (2017). Estimating the intrinsic dimension of datasets by a
  minimal neighborhood information (TwoNN). Scientific Reports.
- Geiger, Lu, Icard and Potts (2021). Causal abstractions of neural networks. NeurIPS.
- Gemma Team (2024). Gemma 2: improving open language models at a practical size (logit soft-capping).
- Glielmo et al. (2022). DADApy: distance-based analysis of data-manifolds in Python. Patterns.
- Gurnee et al. (2026). The J-lens paper, Transformer Circuits Thread.
  https://transformer-circuits.pub/2026/workspace/index.html
- Hernandez et al. (2024). Linearity of relation decoding in transformer language models (LRE).
  ICLR. https://github.com/evandez/relations
- Huang et al. (2024). RAVEL: evaluating interpretability methods on disentangling language model
  representations. https://github.com/explanare/ravel
- McInnes, Healy and Melville (2018). UMAP: uniform manifold approximation and projection.
- Meng et al. (2022). Locating and editing factual associations in GPT (ROME, CounterFact).
- Merity et al. (2016). Pointer sentinel mixture models (WikiText-103).
- Miller (1995). WordNet: a lexical database for English.
- Nanda (2023). Attribution patching: activation patching at industrial scale.
  https://www.neelnanda.io/mechanistic-interpretability/attribution-patching
- nostalgebraist (2020). Interpreting GPT: the logit lens.
  https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens
- van der Maaten and Hinton (2008). Visualizing data using t-SNE. JMLR.
- Venna and Kaski (2001). Neighborhood preservation in nonlinear projection methods
  (trustworthiness).
