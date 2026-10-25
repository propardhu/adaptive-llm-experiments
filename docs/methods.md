# Experiment methods and complete results

## Experiment A — learning facts with adapter weights

The fixed dataset in `data.py` contains 12 development names and 24 different
final-test names. Each name receives one of six balanced color codes. Four
training templates present each fact. Two separate templates guide development
selection; three further templates form the final 72-probe evaluation.

An example training prefix is `Project Belvar uses the access code`, with target
` green`. A held-out template is `Access to Project Belvar requires the code`.
The true color comes from that split's fixed assignment. This is supervised
one-token answer adaptation from supplied examples, not unsupervised learning
from arbitrary conversations or an online decoder updating every generated token.

### Compared recipes

1. A rank-8 additive adapter on the vocabulary output projection.
2. Rank-8 adapters on `down_proj` in the final two internal MLP blocks.
3. The same internal adapters, plus a KL penalty matching the frozen model's
   next-token distribution on 48 general-language anchor prefixes.

The third loss is `answer_cross_entropy + lambda * KL(base || adapted)`.
There are 397,824 trainable output-adapter parameters, or 33,792 for the two
internal adapters. All pretrained parameters stay frozen. Adapters use
`B @ (A @ x)` with random `A` and initially zero `B`.

The training objective scores the answer token only. This differs from the first
study, which trained on every token in twelve repeated sentences. Therefore old
and new percentages are not a controlled head-to-head comparison. The old summary
is included under `previous_experiment` solely for context.

### Development selection and transparency

Each method tests learning rates 0.001 and 0.003 and evaluates checkpoints at
40, 80 and 160 updates. The protected method additionally tests KL weights 1 and
5. A development run uses initialization seed 101. The selection criterion is:

`development_fact_accuracy - 2 * (1 - development_retention_agreement)`.

Ties prefer an earlier checkpoint, then lower learning rate, then larger KL weight.
The selected configurations are written before the main final evaluation, which
uses seeds 7, 17 and 27. Full development results are retained. The tested range
is small and is not an exhaustive search. Unequal parameter counts and selected
training budgets prevent interpreting the initial recipe comparison as a clean
causal test of adapter location or KL alone.

A two-step implementation smoke test ran before the full experiment using the
same fixture, including the final prefixes. It returned zero new-fact completions
for every method and was used to verify cache/reset/reload behavior. No parameters,
data or selection criteria were changed in response to those scores. It is not
included as a research run. This is an exploratory local study, not an externally
preregistered benchmark.

### Follow-up matched ablation

After the primary test run, a matched unprotected control was added to isolate
the KL penalty in the chosen protected recipe. It uses the SAME rank, two layers,
learning rate 0.003, 160 steps, initializations and fact minibatches. The anchor
RNG draw is consumed even when the KL term is disabled, preserving fact-batch
order. There is no further tuning, selection or discarded seed. The protocol for
this follow-up is stored separately and its post-primary status is disclosed.

| Condition | Correct new-wording tokens out of 72 | Agreement on 48 unrelated prefixes | Expected answers out of 8 |
|---|---:|---:|---:|
| Frozen, no facts | 0 | 48 | 5 |
| Frozen, perfectly selected fact supplied | 46 | Not evaluated in that condition | Not evaluated |
| Selected output adapter, 40 steps | 6.67 average | 45.67 average | 5 |
| Selected internal adapter, 40 steps | 13 average | 24.67 average | 7 |
| Matched internal adapter, 160 steps | 27 average | 24 average | 2 |
| Protected internal adapter, 160 steps | 23.67 average | 44.67 average | 4 |

Final protected counts by seed are (24,44), (22,44), (25,46), where each pair
means (new-wording correct, unrelated agreement). Matched unprotected counts are
(31,21), (26,24), (24,27). Three seeds repeat the SAME facts and prefixes; they
are not three independent task datasets. Figures show means and min–max ranges,
not confidence intervals or statistical significance.

### What the scores mean

- Primary fact score: full-vocabulary greedy **next-token** exact match against a
  one-token color. It is not multiple-choice accuracy or full-answer quality.
- Stability score: agreement with the frozen model's greedy next token on 48
  held-out general-language prefixes. These differ from the 48 fitting anchors
  and 24 development retention prefixes. It is **not correctness**: an old answer
  can be wrong, and a changed answer can be better.
- A separate eight-prefix check compares with expected answers (capitals, an
  opposite, a time unit and arithmetic). The prefixes are part of the 48 test
  prefixes, but the expected answers do not enter the training or selection loss.
  This tiny diagnostic is not a broad capability benchmark.
- The relevant-fact baseline uses perfect fact selection. It does not implement
  or measure retrieval. Source facts are absent from adapted evaluation prompts.

### Correctness and persistence checks

The cached model path is compared with the original model before adaptation.
Only layers before the final two blocks are cached in the internal-adapter runs;
those layers remain frozen, so the prefix cache is valid for this architecture.
No adapted-layer KV cache is reused. Frozen base parameters are SHA-256 checked
before and after. Detaching the adapters restores baseline recall exactly.

`results/session_adapter.pt` stores the protected seed-7 adapter. The primary run
checks serialization/reconstruction within a process; `verify_reload.py` then
loads the pinned base model and that adapter in a **fresh process**. All 72
predictions and target probabilities matched exactly. This shows persistence
of one session's update, not safe merging or improvement across many sessions.

Recorded update times exclude loading, frozen-prefix construction, evaluation
and other overhead. They are not end-to-end serving benchmarks. This setup does
not establish lower cost than retrieval, context caching or ordinary inference.

## Experiment B — a memory prepared to learn

This is a tiny symbolic association task, **not a language model**. Twelve keys
receive six values, two of each value per episode. An encoder with 144 shared
parameters maps keys to normalized 12-dimensional representations. A temporary
6 × 12 matrix is the writable memory.

Every episode starts with zero memory. A genuine autograd gradient step on squared
association error writes the twelve supplied key/value pairs. Recall then queries
those same associations. The query answers are known from the supplied context;
the goal is memory recall, not predicting unseen labels without evidence.

Outer training optimizes the encoder through that write step, using recall
cross-entropy. It runs 600 steps, batch size 32, Adam learning rate 0.01, for seeds
11, 23 and 37. The inner update uses one step with learning rate 12 on a loss
averaged over the twelve writes. No hyperparameter search was performed.

For each seed, 1,000 evaluation assignments are generated with seed 98765, excluding
assignments seen in training and duplicates within the evaluation set. Each yields
12 recalls. The slow encoder is frozen and hash-checked during evaluation. The
gradient write is independently verified against its closed-form matrix expression.

| Condition | Recall averaged over three encoder seeds |
|---|---:|
| Trained encoder with no memory write | 16.7% |
| Random encoder with one gradient write | 87.7% |
| Meta-trained encoder with one gradient write | 100% |

This compares learned representations with their random initializations, not
TTT-E2E against a conventionally pretrained LLM. The trained memory has an outer
training cost that the random representation does not; this is not a compute-matched
efficiency result. The result establishes only that preparation can make writes
more useful on this deliberately simple task. The network has no natural-language
understanding, Transformer, attention, learned forgetting gate or momentum write.
It is not a miniature reproduction of the full Titans or TTT-E2E architecture.

## Why the official TTT systems were not run here

The reference runtime has no CUDA GPU and no installed JAX stack. A direct public
listing request to the official TTT-E2E checkpoint bucket returned HTTP 400:
`Bucket is a requester pays bucket but no user project provided.` No billing
project or paid compute was provisioned. The official repository documents GPU
dependencies and requester-pays checkpoints. Thus no paper checkpoint was evaluated.

The article treats In-Place TTT, TTT-E2E and Titans as published related methods,
with their training requirements and architectural differences stated explicitly.
It does not place paper-reported scores on the local experiment's charts.

## Verify and rebuild the publication

```bash
python analyze.py
python -m pip install -r requirements-publication.txt
python make_figures.py
python make_article.py
```

The analysis audits 2,784 stored LLM probe records plus symbolic-memory totals.
The DOCX was rendered and all five pages visually reviewed before delivery.
Quantitative charts are generated from the audited JSON; conceptual diagrams explain the mechanisms. Raw values, code,
protocols and the full set of tested recipes are retained.

## Related work

- https://huggingface.co/HuggingFaceTB/SmolLM2-135M
- https://arxiv.org/abs/2604.06169
- https://arxiv.org/abs/2512.23675
- https://arxiv.org/abs/2501.00663
- https://github.com/test-time-training/e2e

External sources checked 25 October 2026. See the model's own license and terms.
