# RAG-history

A retrieval-augmented generation system over Indian history textbooks, built with
**retrieval quality as the subject** rather than as plumbing behind a chat interface.

The questions this repo tries to answer: does retrieval find the right passage, and how
do you know? What happens on queries that are ambiguous, multi-hop, or unanswerable from
the corpus? Where does it fail, why, and what measurably fixed it?

**Status:** corpus, chunking, embedding, and a scored retrieval baseline are done —
results below. Generation (calling an LLM over retrieved chunks) and groundedness scoring
are not built yet; see Future Tasks.

---

## The corpus

12 NCERT history textbooks (classes 6–12) plus three Old NCERT volumes
(R.S. Sharma, Satish Chandra, Bipan Chandra): 27 PDFs, ~5,700 pages, 78 detected
chapters, **2,859 chunks** at a median ~357 tokens.

Quality is deliberately mixed — current NCERT books extract as clean digital text, the
Old NCERT volumes are scanned and OCR'd. Every chunk carries `source_type`
(~1,612 clean / ~1,125 OCR / ~122 PUA-decoded), so retrieval can be scored separately for
clean and degraded text instead of averaged into one number that hides the difference.

---

## Reproducing

```bash
pip install requests pymupdf pytesseract pillow chromadb sentence-transformers rank-bm25
# OCR also needs the Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki

python download_ncert.py          # fetch PDFs (not committed)
python extract_text.py            # PDF -> text, OCR fallback for scanned volumes
python detect_chapters.py         # detect chapter boundaries
python repair_chapter_index.py    # repair low-confidence boundaries
# apply the one manual correction below
python chunk_corpus.py            # chunk for embedding
python embed_chunks.py            # embed into a local Chroma collection
python backfill_gold_chunks.py    # generate eval_set.json with gold chunk IDs
python eval_retrieval.py          # score dense retrieval
python eval_hybrid.py --sweep     # compare dense / BM25 / fusion
```

OCR runs at ~5.5 s/page (25–30 min per scanned volume). Embedding runs at ~5 chunks/s on
CPU (~10 min for the full corpus).

<!-- `chunk_corpus.py` must run after `repair_chapter_index.py`, or it silently chunks without
chapter metadata — check its first output line reads `Loaded chapter index: 78 chapters
across 8 books`.

If `sentence-transformers` fails to import (some Windows machines block scikit-learn's
compiled BLAS library via Application Control policy), `pip install fastembed` — scripts
fall back to an ONNX backend using the same model. Cross-encoder reranking has no
fastembed equivalent and needs `sentence-transformers` to run at all.-->

**The one manual correction.** `chapter_index_repaired.csv` line 76, `chapter_title`:
`‘` → `Administrative Changes After 1858`. OCR left a stray quote mark ahead of the real
title, and the title heuristic takes the first non-junk line after the chapter marker.

---

## Known defects

| Defect | Location | Handling |
|---|---|---|
| `ijarah` extracted as `yarah` | `old_ncert_modern_india_bipan_chandra.txt` L299, L767 | Left in place — drives eval question X1 |
| `Zulfiqar Khan` also appears as `Zulfigar Khan` | same file, L267–1625 | Left in place — drives eval question X2 |
| 14,339 chars in the Unicode Private Use Area | `class8_our_pasts_3.txt` | Decoded with a `-0xF000` offset |
| 11 chunks (0.4%) exceed the model's 512-token limit | mostly OCR text | Accepted; tails are truncated |

The first two are deliberately not corrected — a query for "ijarah" cannot lexically match
a corpus spelling it "yarah". These are structural failures chunk-size tuning cannot fix,
and the retrieval results below use them as a live test rather than a hypothetical one.

---

## Design decisions

**Chunking: sentence-aware fixed-size, not recursive character splitting.** Lines here
average 33–48 characters (hard-wrapped at PDF column width), and blank lines occur almost
only around page markers, so the usual `\n\n → \n → sentence → char` default degenerates
into page-boundary splitting. The pipeline de-wraps lines into sentences first, then packs
to a token budget with overlap at sentence granularity.

**Section-awareness only where sections exist.** Class 11–12 files carry the authors' own
numbered sections; other files fall back to sentence packing. `chunks_nosections.jsonl` is
a comparison set generated so the benefit can be tested, not assumed.

<!-- **Furniture is tagged, not deleted.** `class10` alone has 184 figure captions and 49 
glossary boxes. Deleting them is tempting and wrong — some gold answers live inside them.
One limitation found in practice: packing sometimes absorbs a glossary box into an
adjacent prose chunk, so `content_type` is not a fully reliable filter for isolated
furniture.-->

**Chapter repair: two methods, and a refusal to guess.** Detection left 54 low-confidence
rows, reduced to 3 unresolved. Where a table of contents is parseable, chapters are
aligned to it by order-preserving fuzzy matching. Where OCR decoupled TOC titles from page
numbers, forcing alignment would invent data, so those books are renumbered by document
order and held at `medium` confidence.

<!-- **Embedding model: bge-small-en-v1.5, not all-MiniLM-L6-v2.** MiniLM truncates at 256 
word-pieces against a 90th-percentile chunk near 399 tokens. bge-small handles 512 at
similar size, and is asymmetric — queries need an instruction prefix, passages don't — so
the prefix lives in Chroma's collection metadata and is read back at query time.-->

---

## Evaluation set

49 questions across four tiers — Easy, Medium, Hard, Trick Questions — where difficulty
means *retrieval* difficulty (chunk cardinality, lexical overlap, distractor density,
answer locality, specification, corpus defects), not exam difficulty.

<!-- Gold chunk IDs are assigned by evidence pattern (the literal text that must appear in a
chunk holding the answer), not by search rank — an early rank-based approach put the wrong
chunk first often enough to be unusable (e.g. "what is a stupa" ranked a passage about an
unrelated iron pillar above the actual definition). 38 of 49 questions verify
automatically this way; the rest need a human judgment call, mainly aggregation questions
with many equally valid answers.-->

The Trick tier includes absence questions, where the correct behaviour is to say the
corpus doesn't support the premise — scored later as *failure to abstain*, since a system
with no absence questions in its eval has no evidence it doesn't hallucinate.

---

## Retrieval results

Scored at k=5 against 44 questions with reliable gold chunks.

| Method | recall@5 | full@5 | MRR |
|---|---|---|---|
| Dense (bge-small) | 0.63 | 0.53 | 0.46 |
| **BM25** | **0.73** | **0.61** | **0.49** |
| Hybrid (RRF, best config) | 0.73 | 0.61 | 0.49 |

<!-- **BM25 beats dense retrieval on every tier.** The common advice — dense retrieval is the
strong leg, add BM25 as hybrid support — doesn't hold here. Answers hinge on distinctive
proper nouns and technical terms (*Nizam-ul-Mulk*, *sardeshmukhi*, *Ajit Singh*), exactly
where term-rarity scoring outperforms a 384-dim embedding average.

**Fusion doesn't help ranking.** A full sweep over fusion weight and the RRF constant
found no configuration beating BM25 alone — the best result *is* BM25 alone (α=0). The
default fusion settings (α=0.5, k=60) actively hurt: zero questions gained, five lost,
because a chunk ranked ~10th by both retrievers can outscore one ranked 1st by only one

**The OCR-defect questions behave as designed, but only under the right retriever.**
Dense retrieval finds X1/X2 anyway (their chunks are still semantically about
revenue-farming even with `ijarah` misspelled as `yarah`) — so it does *not* demonstrate
the defect. BM25 misses them, confirming the defect breaks lexical matching specifically.
Poorly-tuned fusion (α=0.5) drags a correct dense hit down to a miss, which is itself a
useful cautionary result about fusion defaults.

**Reranking headroom is small and not worth pursuing yet.** recall@50 minus recall@5 is
0.10 for BM25 (0.84 → potential ceiling) and 0.16 for dense. A cross-encoder reranker was
built (`eval_rerank.py`) but not run to conclusions — with ~10 points of ceiling and no
generation layer yet, effort belongs elsewhere first (see Future Tasks).-->

---

## Future tasks

**Fix 5 broken gold assignments (M8, M14, H6, H13, X5).** These score as retrieval
failures but aren't — each question's evidence pattern matches 20–270 chunks
corpus-wide (H13's "which periods span multiple books" matches 272), and an arbitrary
subset was frozen as "gold." Retrieval returning a different, equally valid chunk gets
penalized. Fix: either narrow these to questions with an identifiable single answer, or
score them by different means than exact chunk-ID match (e.g. topic/book-level credit).

**Build the generation layer.** Only retrieval has been measured — there are no generated
answers yet. Send the top-k chunks (BM25, per the results above) to the Claude API and
score:
- *Groundedness*: is the answer supported by the retrieved text, not hallucinated?
- *Abstention*: on X3/X4/X10 (absence/false-premise questions), does it correctly say
  the corpus doesn't support the premise, rather than confabulating?

**Reranker: built, deliberately not pursued further.** `eval_rerank.py` reranks the union
of BM25 + dense candidates with a cross-encoder. Measured headroom (0.10) is small, and
with generation unscored, a few more points of retrieval recall isn't the highest-value
next step. Revisit after the generation layer exists, if hard-tier `full@k` (currently
0.36 — multi-hop questions finding one of two required chunks and stopping) is still the
binding constraint.

**Query decomposition for multi-hop.** The hard tier's real gap is full@k stuck at 0.36
across every retriever tried. That's a retrieval-composition problem, not a ranking one —
decomposing a multi-hop question into sub-queries is the more promising fix than
reranking.

**Test whether a larger embedding model closes the dense/BM25 gap.** The result above is
specific to bge-small; worth one comparison run with a larger model before concluding
dense retrieval is categorically weaker on this corpus.

---

## Layout

| File | Purpose |
|---|---|
| `download_ncert.py` | Fetch source PDFs, content-hash deduplication |
| `extract_text.py` | PDF → text, per-page OCR fallback, page markers preserved |
| `detect_chapters.py` | Chapter boundary detection (four strategies) |
| `repair_chapter_index.py` | Repair low-confidence boundaries via TOC alignment |
| `chunk_corpus.py` | De-wrap, tag furniture, chunk with metadata |
| `embed_chunks.py` | Embed into Chroma |
| `backfill_gold_chunks.py` | Generate gold chunk IDs for the eval set |
| `eval_retrieval.py` | Score dense retrieval: recall@k, full@k, MRR per tier |
| `eval_hybrid.py` | BM25, fusion sweep, recall-at-depth (reranker headroom) |
| `eval_rerank.py` | Cross-encoder reranking over the retriever union |
| `eval_questions.md` | Evaluation set, taxonomy, verification log |