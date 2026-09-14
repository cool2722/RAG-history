# RAG-history

A retrieval-augmented generation system over Indian history textbooks, built with
**retrieval quality as the subject of the project** rather than as plumbing behind a
chat interface. The corpus is Indian history, chosen as the starting point, but nothing in the pipeline is corpus-specific, so it can be pointed at another domain.

The questions this repo is actually trying to answer:

- Does retrieval find the right passage, and how do you know?
- What happens on queries that are ambiguous, multi-hop, or unanswerable from the corpus?
- Where does it fail, why, and what measurably fixed it?

**Status:** corpus preparation and the evaluation question set are complete. Embedding,
retrieval and scoring are in progress. There are no retrieval numbers yet.

---

## Corpus

12 NCERT history textbooks (classes 6–12) plus three Old NCERT volumes
(R.S. Sharma, Satish Chandra, Bipan Chandra), acquired as PDFs and processed to text.

| | |
|---|---|
| Source PDFs | 27 files, ~5,700 pages |
| Extracted text | 31 files |
| Detected chapters | 78 |
| Chunks | 2,859 (median ~357 estimated tokens) |
| By source type | ~1,612 clean / ~1,125 OCR / ~122 PUA-decoded |

The corpus is deliberately mixed in quality. The current NCERT books extract as clean
digital text; the Old NCERT volumes are scanned and OCR'd. Every chunk carries a
`source_type` field so retrieval quality can be reported separately for clean and
degraded text instead of averaged into a single number that hides the difference.

Source PDFs and generated vector stores are gitignored — large, regenerable from these
scripts, and in the case of the PDFs not ours to redistribute.

---

## Reproducing the corpus

```bash
python download_ncert.py          # fetch PDFs (not committed)
python extract_text.py            # PDF -> text, OCR fallback for scanned volumes
python detect_chapters.py         # detect chapter boundaries
python repair_chapter_index.py    # repair low-confidence boundaries
# then apply the one manual correction below in case of Indian history corpus
python chunk_corpus.py            # chunk for embedding
python chunk_corpus.py --no-sections --out chunks_nosections.jsonl
```

Requirements: `requests`, `pymupdf`, `pytesseract`, `pillow`, plus the Tesseract binary
itself for the OCR step. OCR runs at roughly 5.5 seconds per page, so the three scanned
volumes take around 25–30 minutes each; the clean books take seconds.

`chunk_corpus.py` must be run after `repair_chapter_index.py`, since it reads
`chapter_index_repaired.csv` for chapter metadata. If that file is missing it silently
falls back to chunking without chapter metadata — check that its first line of output
reads `Loaded chapter index: 78 chapters across 8 books`.

### The one manual correction

`chapter_index_repaired.csv`, line 76, `chapter_title` field:

| | |
|---|---|
| Before | `‘` |
| After | `Administrative Changes After 1858` |

The chapter boundary is correct — the source has `CHAPTER IX` at line 7452. But the OCR
emitted a stray quotation mark on line 7454, before the real title on line 7456, and the
title-recovery heuristic takes the first non-junk line after the marker:

```
7452  CHAPTER IX
7454  ‘                                    <- OCR artifact, taken as the title
7456  Administrative Changes After 1858    <- the actual title
```

Loosening the junk filter to catch this risked new false positives elsewhere, so the
single affected cell is corrected by hand. Only `chapter_title` was changed;
`start_line`, `end_line` and `start_page` are as the script produced them.

---

## Known corpus defects

Three verified defects, handled differently on purpose:

| Defect | Location | Handling |
|---|---|---|
| `ijarah` extracted as `yarah` | `old_ncert_modern_india_bipan_chandra.txt`, lines 299 and 767 | Left in place — drives eval question X1 |
| `Zulfiqar Khan` also appears as `Zulfigar Khan` | same file, lines 267–1625 | Left in place — drives eval question X2 |
| 14,339 characters in the Unicode Private Use Area (U+F000–U+F0FF) | `class8_our_pasts_3.txt` | Decoded with a `-0xF000` offset |

The first two are **deliberately not corrected**. A query for "ijarah" cannot lexically
match a corpus that spells it "yarah", and a query for "Zulfiqar Khan" misses every
chunk containing the corrupted variant. These are structural retrieval failures that no
amount of chunk-size tuning fixes, which makes them the right basis for a measured
before/after: run the eval against the raw text, apply a targeted fix (variant map or
fuzzy matching), re-run the same two questions, report both numbers.

The PUA issue is different in kind and is corrected automatically. Those characters are
not degraded text but a font-encoding artifact — the chapter headings are not merely
misspelled, they are unreadable and unsearchable until decoded. Leaving them in would
remove content from the corpus rather than test anything. `source_type` marks which
chunks came from that file.

---

## Design decisions and trade-offs

### Chunking: sentence-aware fixed-size, not recursive character splitting

The common default splits on `\n\n` → `\n` → sentence → character. Measured against this
corpus, that is wrong:

| File | Avg. line length | Blank lines not adjacent to a page marker |
|---|---|---|
| `kehs103` | 38 | 0 |
| `lehs203` | 33 | 1 |
| `class6_our_pasts_1` | 33 | 1 |
| `class10_india_and_contemporary_world_2` | 48 | 17 |

Lines average 33–48 characters because the text is hard-wrapped at PDF column width, and
blank lines occur almost only around page markers. Splitting on `\n\n` therefore
degenerates into page-boundary splitting, and splitting on `\n` cuts every six words
mid-sentence. The pipeline de-wraps lines back into sentences first — including rejoining
hyphen-split words, common in the OCR'd volumes — and packs sentences to a token budget
with overlap at sentence granularity, so no chunk begins mid-sentence.

### Section-awareness only where sections exist

The Class 11–12 files carry the authors' own numbered sections (`1. Beginnings`,
`2. Subsistence Strategies`) and these are used as hard chunk boundaries. Other files
have none, and fall back to pure sentence packing. Rather than assert that structural
chunking is better, `chunks_nosections.jsonl` is generated as a comparison set so the
claim can be tested on the eval rather than assumed.

### Furniture is tagged, not deleted

Textbook pages interleave prose with captions, glossary boxes, source extracts and
exercises — `class10` alone has 184 figure captions and 49 glossary boxes. Deleting them
is tempting and wrong: the gold answer for eval question E10 (`sardeshmukhi`) lives in a
Class 7 glossary box. Each chunk instead carries a `content_type`, so furniture can be
filtered or boosted at retrieval time and reported on separately.

### Chapter repair: two methods, chosen per book, and a refusal to guess

Chapter detection produced 54 low-confidence rows. These were not 54 independent problems
but five recurring patterns, four of them mechanically fixable — reduced to 3 unresolved
rows, which are listed explicitly in `repair_report.txt` rather than quietly dropped.

- **TOC sequence alignment** (`class9`, `old_ncert_modern_india`) — order-preserving
  alignment of detected chapters against the book's own table of contents, scored by
  fuzzy title similarity. A garbage title still resolves correctly from its *position*;
  an unmatched TOC entry exposes a genuinely missing chapter. For `class9` an independent
  page-offset cross-check agreed at offset 12 on 6 of 7 chapters.
- **Order-only repair** (`old_ncert_ancient_india`, `old_ncert_medieval_india`) — their
  TOCs are unusable because OCR decoupled titles from page numbers. Forcing alignment
  here would invent data, so these are renumbered by document order with titles recovered
  from nearby headings, and held at `medium` confidence because nothing independently
  corroborates them.

Three behaviours were removed during development after producing wrong output: a
permissive TOC fallback that scraped the copyright page (`ALL RIGHTS RESERVED`,
`Phone : 011-26562708`) as chapters; repairing books that had no low-confidence rows, which
made an already-coherent index worse; and accepting a recovered chapter that landed 17
lines before its neighbour.

Chapter confidence is propagated into chunk metadata as `chapter_confidence` rather than
flattened away, so uncertainty in the boundaries stays visible downstream instead of
silently becoming fact.

### Evaluation set: verified, with the failures recorded

49 questions across four tiers — Easy, Medium, Hard, and Trick Questions — where
difficulty is defined by *retrieval* difficulty (chunk cardinality, lexical overlap,
distractor density, answer locality, specification, corpus defects) rather than exam
difficulty. A question that is hard for a student can be trivial for a retriever, and
vice versa.

Every question was verified against the actual corpus in two passes. That process found
real errors, which are recorded rather than quietly corrected: five questions had the
wrong source book, one was reworded after the corpus turned out to use different
vocabulary than assumed, and **one question (X8) was retired entirely** because the
cross-source disagreement it assumed does not exist in the corpus. Gold chunk IDs are
recorded so retrieval can be scored independently of generation.

The Trick tier includes absence questions, where the correct behaviour is to say the
corpus does not support the premise. A system with no absence questions in its eval has
no evidence that it does not hallucinate, so these are scored as *failure to abstain*
rather than merely as wrong answers.

### Embedding model

Chunks are packed to ~400 estimated tokens, with a 90th percentile near 399.
`all-MiniLM-L6-v2` truncates at 256 word-pieces and would silently cut the tail off a
large fraction of chunks, so a 512-token model (`bge-small-en-v1.5`) is used instead —
still CPU-viable, at some cost in speed.

---

## Repository layout

| File | Purpose |
|---|---|
| `download_ncert.py` | Fetch source PDFs, with content-hash deduplication |
| `extract_text.py` | PDF → text, per-page OCR fallback, page markers preserved |
| `detect_chapters.py` | Chapter boundary detection (four strategies) |
| `repair_chapter_index.py` | Repair low-confidence boundaries via TOC alignment |
| `chunk_corpus.py` | De-wrap, tag furniture, chunk with metadata |
| `eval_questions.md` | Evaluation set, difficulty taxonomy, verification log |
| `CHANGELOG_corpus_prep.md` | Full record of pipeline changes and manual corrections |