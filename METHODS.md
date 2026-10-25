# Protocol and interpretation

## Research question

The study isolates whether a trainable neural memory can change its recall while
the shared controller remains unchanged, and what this separation enables when new
information is false. It does not test a pretrained LLM, natural-language understanding,
general intelligence, production serving, or automatic verification of source truth.

## Relationship to Titans

The study implements a related inner update: a neural memory learns key-to-value
associations with a squared-error gradient, momentum, and forgetting. An outer loop
prepares the controller, initial memory and update gates for later adaptation.
This follows the broad mechanism in Titans equations 12–14.

Our memory is a two-layer tanh network with normalized hidden features, 32 input
coordinates, 64 hidden units and eight outputs, without biases. Our gates consume a
32-dimensional key and the observed one-hot value. Learning rate is bounded in
(0, 2), momentum in (0, 0.9), and decay in (0, 0.02). These are our chosen parameters,
not a reproduction of a published Titans model configuration.

We share the query and write-key encoder, use symbolic one-hot value targets,
and do not include attention, Titans' architectural integration variants, or its
separate data-independent persistent-memory parameters. The experiment queries
the neural memory directly. It cannot establish how a full LLM would combine a
memory read with its pretrained knowledge or respond to conflicting documents.

Titans' test-time neural memory is itself a set of weights. Application-level
cross-session persistence still requires deciding whether to retain or save that
fast state. Reset, replay and source filtering are deployment controls demonstrated
here; they are not claimed as new research inventions or Titans' truth-verification
algorithm. Restoring weights without the momentum would reproduce reads, but could
change subsequent writes; our checkpoint retains both.

## Data and preparation

Each episode creates fresh normalized Gaussian 32-dimensional keys and a balanced
assignment to eight values. The observed key vector is used again at query time.
There is no semantic retrieval, paraphrase, held-out key recognition, or missing-key
task. New random assignments prevent fixed shared parameters from knowing the
episode's associations before observing its stream.

Each of seeds 11, 23 and 37 trains for a fixed 1,000 Adam steps, batch size 32,
learning rate 0.002, on 16-association streams. A single actual autograd update is
performed for each incoming association. Outer cross entropy scores recall after
the complete stream. The outer gradient differentiates through those updates.
Global gradient norm clipping is 1.0. A separate 128-stream development set is
logged every 100 steps; it does not select checkpoints or configurations.
The final training step is used for every seed. No final-test tuning was performed.

At evaluation, shared parameters and initial-state templates remain fixed. Each
independent stream receives its own copy of memory weights and momentum. Gradients
are enabled only for the actual fast-state update. The controller supplies detached
keys and gate values. SHA-256 checks cover every shared parameter before and after
all evaluations.

## Final protocol

All three trained controllers receive the same 200 independent test streams of
32 associations, generated with a separate random seed. This yields 6,400 unique
key/value probes, repeated across three controller seeds, or 19,200 predictions per
condition. Means and seed minimum–maximum ranges are reported; these ranges are
not confidence intervals and the repeated predictions are not independent samples.

1. Recall all 32 values after writing the clean stream.
2. Select eight distinct keys and append one wrong replacement per key. Each wrong
   value is guaranteed to differ from the original value.
3. Recall all 32 values after these false updates.
4. Append one correct replacement for each of those eight keys and recall again.
5. For the source-filter control, reject the false events entirely and retain the
   clean state. No write, momentum update or decay occurs for a rejected event.
6. Reset to the initial state and replay the original accepted stream in the same
   order. Check all fast weights, momentum and logits for exact equality.

The source-filter control has perfect, supplied provenance labels: original and
correction events are trusted; injected false events are untrusted. It is an idealized
control, not an automatic fact checker, learned gate or evidence of resilience to
a corrupted trusted source. Accepting a false event still permits it to influence
memory even if its source label says trusted.

The exact-key lookup baseline retains an observed value for every key identity and
uses last-write-wins replacement. Its clean, false-update and corrected accuracies
are 100%, 75% and 100%. It has no approximation interference between distinct keys.
It is not RAG and does not share a measured byte/latency budget with neural memory.

The untrained-memory baseline uses the same seed's controller and memory
initialization before any outer training. The no-write/reset baseline makes a
forced eight-class prediction; chance accuracy is 12.5%. There is no abstention
or confidence-threshold mechanism in the study.

## Capacity and persistence

An independently seeded capacity set tests 8, 16, 32 and 64 associations at the same
memory size. Each load has 200 streams per controller. The 32-item capacity set is
different from the primary test set, so its mean need not equal the primary result.
No width, gate or learning-rate configuration was tuned for these loads.

Each memory contains 2,560 float32 weights and 2,560 momentum entries: 20,480 bytes.
The shared controller has 3,708 trainable parameters during outer training, including
the initial memory template. Its parameters do not change during test-time writing.
Input storage, graph activations, replay logs and shared parameters are excluded
from the fast-state byte count. This is not a demonstration of a storage advantage.

One clean session per seed is saved. `verify_reload.py` runs in a fresh process,
reloads the saved controller and fast state, checks controller hashes, and exactly
reproduces 32 saved logits vectors and predictions per seed. It also checks that
resetting the memory changes these outputs. A separate-user check proves that
writing to a newly initialized state does not mutate an existing user's state.

## Validation and recorded measurements

The smoke test checks write-loss derivatives against finite differences, verifies
that per-episode writes are independent of batch size, checks shared-parameter
identity, and verifies finite outer meta-gradients. `audit.py` recomputes counts
from 134,400 stored neural-memory decisions, validates corruption indices and
values, and reconciles all headline aggregates and capacity counts.

Stored data auditing does not independently reproduce neural computations.
The fresh-process reload verifies a saved subset. A full rerun is necessary to
independently reproduce the complete training and evaluation measurements.

## References

- Titans: https://arxiv.org/abs/2501.00663
- In-Place TTT: https://arxiv.org/abs/2604.06169
- TTT-E2E: https://arxiv.org/abs/2512.23675

In-Place TTT reuses existing MLP output projections as fast weights and uses a
language-modeling-aligned local objective. TTT-E2E meta-trains a model to benefit
from next-token-prediction updates in selected MLPs. Neither method was implemented
or benchmarked here; the article gives a brief conceptual comparison only.
