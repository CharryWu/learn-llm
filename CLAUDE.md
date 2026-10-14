# CLAUDE.md

Personal LLM study repo (notes + Colab notebooks). Not a library: no build, no tests.

## Map

| Path | What |
| --- | --- |
| `LLM-OVERVIEW.md` | Main English overview: 7 sections, KaTeX formulas, collapsed 🔍 symbol atlas under each formula |
| `LLM-OVERVIEW-zh.md` | Colloquial Chinese version (same structure; "符号速读" lines instead of atlas tables) |
| `zhihu/` | Local copies of the 猛猿 zhihu column; figures in `zhihu/images/<sha1-12>.jpg` |
| `notebook/` | Workshop notebooks (`00_` pretraining, `01_` inference & sampling) |
| `*.md` at root | Concept notes (`softmax`, `cross-entropy`, `backprop`, `likelihood`, `stochastic`), `CODE-QA.md` log, workshop topic list |

## Conventions

- Markdown math is KaTeX (Cursor/GitHub preview): `$…$` inline, `$$` on its own lines for display. No inline `$$…$$`, no Chinese text in math without `\text{}`.
- Keep EN and ZH overviews in sync when changing structure, formulas, or figures.
- Link figures as `zhihu/images/…`; link zhihu articles with `<…>` (names have spaces, colons, parens).
- Chinese text uses half-width punctuation followed by a space (`中文, 中文`, `中文: 中文`, `中文 (note)`), not `，：（）`.
- Bold next to Chinese: don't end `**…**` with punctuation when CJK text follows directly (it won't render).
- Code fences need a language tag (`py`, `pycon`, `text`).
- markdownlint: only MD013 (line length) is tolerated; MD033 is disabled in `LLM-OVERVIEW.md` for `<details>`.
- Don't commit weights, datasets, or HF cache (see `.gitignore`).
- Commits: conventional one-liner, e.g. `docs(overview): …`.
