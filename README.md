# learn-llm

Self-directed study of how large language models work — from math intuition through training a complete GPT-2–style model.

## What this is

A personal learning repo, not a library or framework. Content is built around **Google Colab** and **Jupyter notebooks**, supplemented with markdown notes that explain concepts as I go.

**Goals:**

- Understand transformer / LLM fundamentals (attention, tokenization, training loop, sampling)
- Train a small GPT-2–style model end-to-end
- Load and experiment with open-source weights from [Hugging Face](https://huggingface.co/models)

## What's in here

| Kind | Description |
|------|-------------|
| **Notebooks** | Hands-on Colab / Jupyter work — architecture, training, inference |
| **Markdown docs** | Concept notes, workshop summaries, Q&A (e.g. `LLM-OVERVIEW.md`, `backprop.md`) |
| **Artifacts** | Checkpoints and datasets stay **local** (see `.gitignore`); pull pretrained weights from Hugging Face as needed |

Layout is intentionally loose and **will evolve** as topics get added or reorganized. Don't treat folder names as stable API.

## Getting started

1. Clone the repo.
2. Open a notebook in Colab (*File → Upload notebook*) or run locally with Jupyter / VS Code.
3. For pretrained models, use Hugging Face `transformers` — weights download to cache on first use (not committed here).
4. Read the overview docs for context before diving into notebooks.

## Local setup (optional)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install torch transformers datasets tokenizers jupyter
```

## What not to commit

Large binaries (`.pt`, `.safetensors`, checkpoints), datasets, Hugging Face cache, secrets, and notebook checkpoint dirs are all gitignored. If you need a specific model, document the HF repo ID in a notebook or note instead of checking in weights.

---

*Workshop reference: [Build Your Own LLM](https://go.justinangel.ai/neuralnet)*
