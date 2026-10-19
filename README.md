# RAG-history

A retrieval-augmented generation system over Indian history textbooks, built to measure
**how well retrieval works and where it fails**, not just putting a chat bot on top.

Questions it answers with numbers: Does retrieval find the right passage? Does the model
stay inside the retrieved text? Does it refuse when the corpus can't answer, or when the
question is ambiguous?

**Status:** Complete end to end. Includes Corpus, chunking, retrieval comparison (dense / BM25 / hybrid), grounded generation with Claude Sonnet, and scoring of both stages.

Results below.

---

## Pipeline at a glance

```
NCERT PDFs → text (OCR where scanned) → chapters → 2,859 chunks
                                                       │
question → BM25 top-5 ──────────────────────────────── ┘
              │
              └→ Claude Sonnet (answer only from passages, cite IDs, or abstain / ask to clarify) 
                  → scored
                      ├ rule checks (behaviour, 
                      citations)
                        └claim-level judge (Opus + Sonnet)
```

Only two steps call an LLM: **generation** and the **groundedness judges**. 
Chunking, retrieval and embeddings run locally.

---

## The corpus

12 NCERT history textbooks (classes 6–12) plus three Old NCERT volumes (R.S. Sharma,
Satish Chandra, Bipan Chandra): 27 PDFs, ~5,700 pages, 78 detected chapters,
**2,859 chunks** at a median ~357 tokens.

Quality is mixed on purpose: current NCERT books extract as clean text, the Old NCERT
volumes are scanned and OCR'd. Every chunk carries `source_type` (clean / OCR /
PUA-decoded) so results can be split by text quality.

---

## Reproducing

```bash
pip install requests pymupdf pytesseract pillow chromadb sentence-transformers rank-bm25 anthropic
# OCR also needs the Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki

python download_ncert.py          # fetch PDFs
python extract_text.py            # PDF -> text, OCR fallback
python detect_chapters.py         # detect chapter boundaries
python repair_chapter_index.py    # repair low-confidence boundaries
# apply the one manual correction below
python chunk_corpus.py            # chunk for embedding
python embed_chunks.py            # embed into a local Chroma collection
python backfill_gold_chunks.py    # generate eval_set.json with gold chunk IDs
python eval_retrieval.py          # score dense retrieval
python eval_hybrid.py --sweep     # compare dense / BM25 / fusion

export ANTHROPIC_API_KEY=...      # PowerShell: $env:ANTHROPIC_API_KEY = "..."
python generate_answers.py        # generate answers
python score_generation.py        # score generations
```

OCR takes ~5.5 s/page (25–30 min per scanned volume); embedding ~10 min on CPU.
Generation over all 49 questions takes a few minutes and ~170k input tokens; scoring adds
two judge calls per answered question.

**The one manual correction.** `chapter_index_repaired.csv` line 76, `chapter_title`:
`‘` → `Administrative Changes After 1858`. OCR left a stray quote mark ahead of the real
title.

---

## Known defects

| Defect | Location | Handling |
|---|---|---|
| `ijarah` extracted as `yarah` | `old_ncert_modern_india_bipan_chandra.txt` | Left in place — drives eval question X1 |
| `Zulfiqar Khan` also appears as `Zulfigar Khan` | same file | Left in place — drives eval question X2 |
| 14,339 chars in the Unicode Private Use Area | `class8_our_pasts_3.txt` | Decoded with a `-0xF000` offset |
| 11 chunks (0.4%) exceed the embedding model's 512-token limit | mostly OCR text | Accepted; tails truncated |

The OCR spellings are kept on purpose: a query for "ijarah" can't lexically match a corpus
that says "yarah". That is a real failure mode, so it stays as a live test case.

---

## Design decisions

- **Sentence-aware chunking, not recursive character splitting.** The PDFs are
  hard-wrapped at ~40 characters per line, so default splitters degrade into page-boundary
  cuts. Lines are de-wrapped into sentences first, then packed to a token budget.
- **Section-aware only where sections exist** (Class 11–12). Elsewhere, sentence packing.
- **Chapter repair without guessing.** Where a table of contents is parseable, chapters are
  aligned to it; where OCR broke the TOC, chapters are numbered by document order and held
  at `medium` confidence rather than forced.
- **BM25 is the retriever** used for generation, because it won the comparison below.
- **The model must choose a status** — `answer`, `abstain` or `clarify` — and cite passage
  IDs. That makes refusal measurable instead of something you have to read for.
- **The groundedness judge is not the generator.** Sonnet generates; **Opus** is the primary
  judge, and Sonnet also judges the same answers so the self-judging gap can be measured.
  Judges must quote the passage text behind every "supported" verdict, and quotes that are
  not in the passages are downgraded to *unverified*.

---

## Evaluation set

49 questions in four tiers — Easy, Medium, Hard, Trick — where difficulty means *retrieval*
difficulty (how many chunks hold the answer, lexical overlap, distractors), not exam
difficulty. Gold chunks are assigned by matching the literal evidence text, not by search
rank. Four questions are not meant to be answered: X3, X4 and X10 are absence /
false-premise questions (the right behaviour is to abstain) and X5 is ambiguous (the right
behaviour is to ask which revolt).

---

## Results

### Retrieval (k=5, 44 questions with reliable gold)

| Method | recall@5 | full@5 | MRR |
|---|---|---|---|
| Dense (bge-small) | 0.63 | 0.53 | 0.46 |
| **BM25** | **0.73** | **0.61** | **0.49** |
| Hybrid (RRF, best config) | 0.73 | 0.61 | 0.49 |

- **BM25 beats dense on every tier.** Questions hinge on rare proper nouns (*Nizam-ul-Mulk*,
  *sardeshmukhi*), where exact term matching outperforms a small embedding model.
- **Fusion didn't help.** The best fusion setting was BM25 alone; the default setting
  (α=0.5) lost five questions and gained none.
- **The OCR defect is visible in retrieval.** BM25 misses X1 (`ijarah` vs `yarah`); dense
  retrieval still finds it. Hybrid can drag that correct dense hit down.

### Generation (BM25 top-5 → Claude Sonnet, 49 questions, final run)

| | Result |
|---|---|
| Fabricated citations | **0 / 49** (every cited ID was in the retrieved set) |
| Abstain / clarify questions handled correctly | **4 / 4** (X3, X4, X10 abstained; X5 asked to clarify) |
| Answered when the gold passage was retrieved | **27 / 28** |
| When the gold passage was *not* retrieved (17 questions) | abstained on 8, answered on 9 |
| Behaviour matches expectation | 40 / 49 |

**Takeaway: retrieval is the bottleneck, not the model.** When the right passage reaches
the model it answers; when it doesn't, the model mostly refuses rather than inventing an
answer. Eight of the nine "false abstentions" followed a retrieval miss and are correct
refusals. The OCR defect carries through: X1 abstains because BM25 never retrieves the
`yarah` chunk.

### Groundedness (claims in answered responses, 36 answers)

| Judge | Role | Claim-level supported | Answers with every claim supported |
|---|---|---|---|
| claude-opus-5-5 | independent (primary) | 0.97 | 27 / 36 |
| claude-sonnet-5-5 | same model as generator | 0.98 | 30 / 36 |

The two judges agree on 29 of 36 answers. Averaged scores look close (gap +0.01), but at
the answer level (more fine grained) the self-judge was more lenient: in a 10-answer spot-check
(`spot_check.md`) the Opus judge matched the reviewer's verdict on 10/10 and Sonnet on 6/10,
missing two real errors 
<!-- (an attribution to Asoka the passage doesn't make, and two separate causes merged into one). -->

Typical unsupported claims are small overreaches rather than inventions: a causal link the
passage doesn't make, or the model's own synthesis stated as fact. By tier, claim-level
support was 0.99 easy, 0.96 medium, 0.97 hard, 0.94 trick.

---

## Limitations

- **Both judges are Claude models**, so errors common to the model family can pass both.
  The self-judge gap and spot-check measure part of this; a judge from another vendor would
  close it.
- **Groundedness is not correctness.** It checks the answer against the *retrieved* text.
  A faithful answer built on an incomplete retrieval still scores well (e.g. H5 omits the
  Class 12 section on the end of the Harappan civilisation because it was never retrieved).
- **Not fully deterministic.** The API/model used here rejected a temperature setting, so
  runs vary slightly (across runs, 38–40 of 49 behaviours matched; individual questions such
  as H9 flipped between answering and abstaining). Treat results as ±1–2 questions.
- **Small samples.** Four abstain/clarify questions, ten spot-checked answers, one final run.
- **Some gold labels are imperfect** (M8, M14, H6, H13, X5), which understates retrieval
  scores. The full list of biases is in `bias_list.md`.

---

## Future tasks

- **Better multi-hop retrieval.** Hard-tier full@5 is stuck at 0.36 across every retriever:
  questions need two chunks and get one. Query decomposition is the likely fix.
- **Cross-encoder reranking.** Built (`eval_rerank.py`) but not run; measured headroom
  is only ~0.10 recall.
- **Judge from another model family**, or an NLI-based checker, to remove the remaining
  shared-blind-spot risk.
- **Larger embedding model** to test whether the dense/BM25 gap is specific to bge-small.

---

## Layout

| File | Purpose |
|---|---|
| `download_ncert.py` | Fetch source PDFs, content-hash deduplication |
| `extract_text.py` | PDF → text, per-page OCR fallback |
| `detect_chapters.py` | Chapter boundary detection |
| `repair_chapter_index.py` | Repair low-confidence boundaries via TOC alignment |
| `chunk_corpus.py` | De-wrap, tag furniture, chunk with metadata |
| `embed_chunks.py` | Embed into Chroma |
| `backfill_gold_chunks.py` | Generate gold chunk IDs for the eval set |
| `eval_retrieval.py` | Dense retrieval scoring: recall@k, full@k, MRR per tier |
| `eval_hybrid.py` | BM25, fusion sweep, recall-at-depth |
| `eval_rerank.py` | Cross-encoder reranking over the retriever union |
| `generate_answers.py` | BM25 top-k → Claude; answer / abstain / clarify with citations |
| `score_generation.py` | Behaviour, citation and two-judge groundedness scoring; spot-check sample |
| `spot_check.md` | 10 sampled answers with judge verdicts and reviewer verdicts |
| `bias_list.md` | Known biases and threats to validity |
| `eval_questions.md` | Evaluation set, taxonomy, verification log |
| `CHANGELOG_corpus_prep.md` | Record of pipeline changes and manual corrections |