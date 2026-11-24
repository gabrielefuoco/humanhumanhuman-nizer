---
title: Humanhumanhuman Nizer
emoji: ✨
colorFrom: blue
colorTo: purple
sdk: gradio
pinned: false
app_port: 7860
---

# The Humanizer Pipeline

A Gradio app that scores a document sentence by sentence on the two signals AI-text detectors approximate — **perplexity** and **burstiness** — highlights the sentences that look machine-written, and rewrites them with minimal, controlled edits. Built for a hard constraint: the document is **LaTeX**, so a rewrite that breaks a formula, a citation or a technical term is worse than no rewrite at all.

## Table of contents

- [What it does](#what-it-does)
- [How it works](#how-it-works)
- [Decisions](#decisions)
- [Repository map](#repository-map)
- [Running it](#running-it)
- [Limitations](#limitations)

## What it does

1. **Scores.** Each sentence gets a perplexity value from a small local model, and the document gets a burstiness figure — the standard deviation of sentence lengths. Both are shown in the UI, with the sentences ranked and paginated.
2. **Locates.** Sentences below the predictability threshold are highlighted, so the user edits where it matters instead of rewriting the whole document.
3. **Suggests.** For the words inside those sentences, the app proposes synonyms — from Mistral for context-aware choices, from WordNet as an offline fallback — with a score per candidate.
4. **Rewrites.** A rewrite request goes out with the neighbouring sentences as context, and comes back as a **minimal edit**: one or two words replaced, nothing else touched.

## How it works

### Scoring

Perplexity comes from `Qwen/Qwen3.5-0.8B` running locally through `transformers`: the sentence is passed as both input and label, and perplexity is the exponential of the resulting loss. It is computed **per sentence**, not per document, because the useful output is a location, not a verdict.

Burstiness is the standard deviation of the sentence lengths in words. Uniform sentence length is the cheapest signal of machine writing, and the cheapest to fix.

Before any model runs, `apply_algorithmic_rules()` catches the mechanical tells — em-dashes, three-item lists, stock transitions — with regular expressions. A regex is deterministic and free; a language model is neither.

### Protecting the document

`latex_parser.py` masks five kinds of structure before anything else happens: the preamble, display blocks, inline math, citations and structural commands. Each becomes a placeholder like `[MATH_3]`, and a registry keeps the original text. All scoring and rewriting happens on the masked text; the placeholders are restored on the way out.

The rewrite prompt enforces the same rule from the model's side: the returned sentence must contain every placeholder exactly as received, and must never touch proper nouns or domain terms — *fractional Laplacian*, *overdetermined problem*, *domain*, *singularity*. A missing placeholder is a corrupted document, so the prompt warns that the system will break.

### The constraints that shape the UI

The app runs on a Hugging Face Space with **ZeroGPU**: GPU time is granted in fixed budgets (120 s for analysis, 60 s for actions), which is why scoring is paginated at **20 sentences per page** instead of processing a whole document in one call. Input is accepted as pasted text, `.docx` or `.tex`, and the result can be exported.

## Decisions

**Mask first, rewrite second.** The obvious design — send the text to a model and replace the output — destroys LaTeX. Masking turns an open-ended generation problem into a bounded substitution problem, and makes the invariant checkable: what went in comes out, placeholders included.

**Minimal edits instead of regeneration.** The prompt asks for at most one or two synonym substitutions and forbids touching the rest. Regeneration would rewrite the author's own phrasing; in a LaTeX document the wording is part of the content.

**Per-sentence scores instead of a document score.** A single perplexity number tells you nothing about where to intervene. Twenty sentences per page with a value each turns the metric into a to-do list.

**Context in the rewrite request.** The neighbours go with the sentence, because a synonym chosen in isolation is how a rewrite ends up grammatically perfect and logically disconnected.

**Two synonym sources with different costs.** Mistral when a key is available, WordNet when it is not: the app keeps working offline, degraded but not broken.

**A 0.8B model, not a large one.** Perplexity only needs a language model's probability distribution, not its intelligence, and a small model fits the Space's RAM. The rewriting model lives behind an API, where capacity actually matters.

## Repository map

```
app.py            Gradio UI, scoring, synonym search, rewrite calls, export
latex_parser.py   masks and restores preamble, math, citations, commands, comments
main.tex          a real 3,258-line mathematics document used as the working case
requirements.txt  gradio, spaces, transformers, torch, nltk, python-docx, regex
todo.md           working notes: the design conversation behind the pipeline
```

## Running it

On a Hugging Face Space: create a Gradio Space, push this repository, and set `MISTRAL_API_KEY` under *Settings → Variables and secrets*. The Qwen weights are downloaded on first start; scoring runs on the free ZeroGPU budget and falls back to CPU locally.

Locally:

```bash
pip install -r requirements.txt
export MISTRAL_API_KEY=...
python app.py
```

Without the key everything except the Mistral rewrite and the Mistral synonyms still works: `spaces.GPU` is stubbed out when the app is not on a Space.

## Limitations

| Item | Impact | Status |
|---|---|---|
| Perplexity is an indicator, not a verdict | it depends on the model, the language and the domain; another model would rank differently | by design |
| The tool optimises a proxy | low perplexity and high variance correlate with detectors, not with good writing | accepted, stated |
| Italian and English only | prompts and prompts' guards are written in these two languages | open |
| LaTeX masks are positional | a rewrite that reorders a sentence around a placeholder can still produce odd output | open |
| Whole-document scoring is paginated | a long document needs several clicks, forced by the ZeroGPU budget | accepted |
| No automated tests | the invariants are checked by eye on the exported document | open |

One note that belongs in the README rather than in a commit message: the honest use of this tool is rewriting your own text so it reads better. Optimising against a detector optimises against a proxy, and the proxy is weak in both directions — it flags non-native writers and classical authors as machine-written, and it accepts machine writing edited by hand as human.
