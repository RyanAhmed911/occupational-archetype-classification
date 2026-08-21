# Persona Domain Classifier 🎓

This model classifies academic domain from persona descriptions. It is `bert-base-uncased` fine-tuned for 10-way single-label classification on a persona dataset.

## Dataset

Sourced from Hugging Face `proj-persona/PersonaHub` (ElitePersonas part10), filtered to English and restricted to the top 10 domain classes. Balanced via undersampling to 17,760 rows, split 70% train / 15% validation / 15% test.

## Training Configuration

- Learning rate: 5e-5
- Batch size: 32
- Epochs: 3
- Max length: 150
- Validation accuracy: ~0.95
- Weighted F1: ~0.95

## Labels

- Biology
- Computer Science
- Education
- Engineering
- Environmental Science
- History
- Medicine
- Physics
- Psychology
- Public Health

## Preprocessing

Inputs undergo whitespace normalization only: consecutive whitespace is collapsed to single spaces, and leading/trailing whitespace is stripped. No lowercasing, stopword removal, or lemmatization is applied.

## Setup

The 418MB weights file exceeds GitHub's 100MB per-file limit, so it is not in this repo.
Fetch it from the release before running anything (everything else is already here):

```bash
curl -L -o bert_light_config2/model.safetensors \
  https://github.com/RyanAhmed911/occupational-archetype-classification/releases/download/v1.0/model.safetensors
```

Check it arrived at ~418MB — a failed download silently writes a small HTML error page.

## Running it

Requires **Python 3.10+** (Gradio no longer supports 3.9). From this directory:

```bash
pip install -r requirements.txt
python app.py
```

Deployed on [Modal](https://modal.com) (serverless, scales to zero):

```bash
modal setup                  # one-time, opens browser to sign in
modal deploy modal_app.py    # prints the permanent public URL
```

First request after ~5 minutes idle pays a cold start of roughly 19 seconds; warm
requests return in about 1 second.

`app.py` reads the weights path from the `MODEL_DIR` env var, defaulting to
`bert_light_config2` for local runs; `modal_app.py` sets it to the copy baked into
the container image.
