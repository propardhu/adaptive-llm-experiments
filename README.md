# Titans Inspired Neural Memory Experiment

Can a separate neural memory learn new associations while shared parameters stay fixed?
This small CPU study tests recall, false updates, corrections, source filtering, reset,
checkpoint reload and memory capacity. It is inspired by the neural-memory update in
[Titans](https://arxiv.org/abs/2501.00663).

**The memory learns into its own weights.** Separating these weights from the shared
controller makes their lifetime explicit; it does not automatically make inputs true.
This is a synthetic associative-memory experiment, not a language model, an official
Titans implementation, or a comparison of complete LLM architectures.

Read [Giving AI a Memory We Can Manage](article.md) for the illustrated explanation,
or [METHODS.md](METHODS.md) for the full experimental protocol.

![Separate neural memory and the fixed controller](figures/01-where-learning-happens.png)

## Recorded results

| Condition | Recall |
|---|---:|
| No memory writes / reset to initial state | 12.5% |
| Neural memory before meta-training | 70.9% |
| Learned memory, clean stream | 92.0% |
| After false replacements for 8 of 32 facts | 65.8% |
| After one verified correction for each of those 8 facts | 87.2% |
| Reset and replay the clean stream | 92.0% |
| Perfect source filter rejects all injected false updates | 92.0% |
| Exact-key lookup, clean stream | 100.0% |

Three independent controller-training seeds each evaluated the same 200 held-out
32-association streams: 19,200 predictions per condition, not 600 independent test
streams. Exact recall uses eight possible values. Source filtering assumes supplied
source labels are perfect; no fact-checking model was built. Every key queried was
provided in the stream. The exact-key baseline stores the supplied mapping and uses
last-write-wins updates; it does not evaluate semantic search or RAG.

The shared controller hashes stayed unchanged. Reset/replay reproduced the full
clean fast state exactly. A fresh Python process reloaded one saved session per seed
and exactly reproduced all 96 saved logits vectors and predictions.

## Audit the stored predictions

No model download or PyTorch is needed for this check:

```bash
python3 audit.py --results results
```

This verifies 134,400 stored neural-memory decisions and their reported aggregates.
It checks stored data consistency; it does not rerun learning.

## Run locally

The reference run used Python 3.12.14, PyTorch 2.6.0+cpu, float32 and one CPU thread.
No pretrained-model download is required. The full reference run took about 102
seconds on the execution environment; this is not a hardware benchmark.

```bash
git clone https://github.com/propardhu/adaptive-llm-experiments.git
cd adaptive-llm-experiments
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python experiment.py --smoke
python experiment.py --output results-my-run
python verify_reload.py --results results-my-run
python audit.py --results results-my-run
```

Use a fresh output directory. The script refuses to overwrite completed results.
The smoke test checks finite-difference gradients, batch independence, fixed shared
parameters, and the outer meta-gradient. Only Linux CPU was tested.

To reproduce the published checkpoint reload independently:

```bash
python verify_reload.py --results results
```

## Design

The frozen-at-test controller encodes 32-dimensional random keys and supplies
input-dependent learning-rate, momentum and forgetting controls. A two-layer memory
with 2,560 fast weights learns key-to-value associations through squared-error
gradient updates. Outer training optimizes recall after a 16-association stream.

Memory weights and momentum together use 20 KiB per stream. This excludes the
shared controller, input vectors, activations, metadata and the optional replay log.
An exact table is a stronger baseline for this deliberately simple task. No memory,
token, latency or cost advantage over lookup or an LLM context window is established.

See [METHODS.md](METHODS.md) for the complete protocol, differences from Titans,
and the distinction between learned forgetting gates and source verification.

## Files and article

- `experiment.py`: learning mechanism, fixed protocol, training and evaluation.
- `verify_reload.py`: fresh-process persistence checks.
- `audit.py`: independent audit of saved predictions.
- `results/`: protocols, training curves, predictions, summaries and checkpoints.
- `figures/`: editable SVG and PNG figures.
- [Illustrated article](Learning-in-a-Separate-Neural-Memory.html): reading copy with embedded images.
- [Editable article](Learning-in-a-Separate-Neural-Memory.docx): Word copy with images.
- [Markdown article](article.md): repository-friendly reading copy.
- `make_figures.py`, `make_article.py`: artifact builders.

```bash
python -m pip install -r requirements-publication.txt
python make_figures.py
python make_article.py
```

Author: Partha Sai Guttikonda · partha_systems.ai

No separate project license has been selected. The cited research retains its own terms.
