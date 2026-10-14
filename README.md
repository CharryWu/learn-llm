# learn-llm

Self-directed study of how large language models work — from math intuition through training a complete GPT-2–style model.

## What this is

A personal learning repo, not a library or framework. Content is built around **Google Colab** and **Jupyter notebooks**, supplemented with markdown notes that explain concepts as I go.

**Goals:**

- Understand transformer / LLM fundamentals (attention, tokenization, training loop, sampling)
- Train a small GPT-2–style model end-to-end
- Load and experiment with open-source weights from [Hugging Face](https://huggingface.co/models)

## What's in here

| Path | Description |
|------|-------------|
| [`LLM-OVERVIEW.md`](LLM-OVERVIEW.md) | **Start here.** One-page map of the workshop: NN basics → training → stability → text-to-numbers → attention → lifecycle → systems. Key formulas with a collapsible 🔍 symbol atlas, figures, and cross-links |
| [`LLM-OVERVIEW-zh.md`](LLM-OVERVIEW-zh.md) | Colloquial Chinese version of the overview |
| [`zhihu/`](zhihu/) | Local copies (with images) of the 猛猿 zhihu column: Transformer notes (PE, attention, LayerNorm, ResNet, tokenization), BERT, GPT-1/2/3, InstructGPT, Codex, RL/MDP, pipeline parallelism |
| [`notebook/`](notebook/) | Colab / Jupyter work — `00` pretraining, `01` inference & sampling |
| Concept notes | [`softmax.md`](softmax.md), [`cross-entropy.md`](cross-entropy.md), [`backprop.md`](backprop.md), [`likelihood.md`](likelihood.md), [`stochastic.md`](stochastic.md) |
| [`CODE-QA.md`](CODE-QA.md) | Running log of line-by-line code explanations from the workshop |
| Workshop materials | [`Build your own LLM Workshop Topic List.md`](<Build your own LLM Workshop Topic List.md>), [`Building intuition by Building LLMs.pdf`](<Building intuition by Building LLMs.pdf>) (slide deck) |
| Artifacts | Checkpoints and datasets stay **local** (see `.gitignore`); pull pretrained weights from Hugging Face as needed |

Layout is intentionally loose and **will evolve** as topics get added or reorganized. Don't treat folder names as stable API.

## Getting started

1. Clone the repo.
2. Open a notebook in Colab (*File → Upload notebook*) or run locally with Jupyter / VS Code.
3. For pretrained models, use Hugging Face `transformers` — weights download to cache on first use (not committed here).
4. Read [`LLM-OVERVIEW.md`](LLM-OVERVIEW.md) (or the [中文版](LLM-OVERVIEW-zh.md)) for context before diving into notebooks. Formulas render with KaTeX in the GitHub / VS Code / Cursor preview.

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
