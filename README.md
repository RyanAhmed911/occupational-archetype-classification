# occupational-archetype-classification
A multi-class text classification pipeline predicting professional domains from synthetic persona descriptions, built for the course CSE440: Natural Language Processing II.

## Live Demo

**https://farhanzarif98--persona-domain-classifier-ui.modal.run**

Paste a one-line description of someone's work and the model predicts which of 10 academic
domains they belong to, with confidence scores for the top 3.

The demo runs serverless on Modal and sleeps after 5 minutes of inactivity, so the first
load may take ~19 seconds while the container wakes. It is not broken — give it a moment.

## Repository layout

| Path | Contents |
|---|---|
| `06_23301529_24241095_24241213.ipynb` | Full pipeline: EDA, preprocessing, 10 model families × 3 configs, evaluation |
| `06_23301529_24241095_24241213.pdf` | Final written report |
| [`RESULTS.md`](RESULTS.md) | Experiments and analysis: main results, per-class behaviour, ablations, error analysis |
| `figures/` | Generated plots — class distribution, word clouds, confusion matrices, model comparisons |
| `deployment/` | The Gradio app and Modal deployment for the winning model |

## Results

BERT-base with light preprocessing wins, but the more interesting findings are about *why*:
preprocessing interacts with the representation rather than the task, hyperparameter
sensitivity concentrates in only two places, and adding gating to a SimpleRNN is worth
roughly **+42 F1 points** at identical width and dropout. `history` is the easiest class
across all ten models; `psychology` and `medicine` are the most confusable.

See [`RESULTS.md`](RESULTS.md) for the full analysis.

The best model is BERT-base fine-tuned with light preprocessing (lr=5e-5, batch 32, 3
epochs), reaching ~0.95 accuracy and weighted F1 on the validation set. See
[`deployment/README.md`](deployment/README.md) to run or redeploy it — note the trained
weights are distributed as a [release asset](../../releases), not committed to the repo.
