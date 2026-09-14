# RAG Evaluation Question Set — NCERT History Corpus

## Part 1: What actually makes a question hard?

The most common mistake in building a RAG eval set is importing **exam difficulty**
instead of **retrieval difficulty**. They are not the same thing, and conflating them
produces an eval set that looks rigorous but measures nothing useful.

A question that is hard for a *student* ("Assess the causes of Mughal decline") may be
trivial for a retriever: it is topically distinctive, the relevant chapter is obvious,
and any chunk from that chapter will do. Conversely, a question that is trivial for a
student ("What was ijarah?") can be *impossible* for a retriever if the corpus spells
the word differently than the question does.

So difficulty here is defined along retrieval-relevant axes:

| Axis | What varies |
|---|---|
| **Chunk cardinality** | How many distinct chunks must be retrieved for the answer to be complete? |
| **Lexical overlap** | Does the question share surface words with the source text, or must the embedding bridge a paraphrase/synonym gap? |
| **Distractor density** | How many chunks *look* relevant but don't contain the answer? |
| **Answer locality** | Is the answer a contiguous span, or scattered/aggregated across a passage that a chunk boundary may bisect? |
| **Specification** | Is the question unambiguous, or does it need disambiguation before retrieval is even well-defined? |
| **Corpus defects** | Does the answer sit in OCR-degraded text where the surface form is corrupted? |

### The four tiers

**EASY — "is the pipeline plugged in?"**
Single chunk. High lexical overlap: the question reuses the distinctive proper nouns or
technical terms that appear verbatim in the source. Low distractor density: the topic
occurs in essentially one place in the corpus. Answer is a contiguous span.
*Diagnostic value:* near-zero for showing sophistication, but essential as a smoke test.
If easy questions fail, something is broken (bad chunking, wrong embedding model,
index not populated) and no amount of reranking will save you. Expect ~95%+ recall@5.
**These are not there to be impressive. They are there to falsify "the system works".**

**MEDIUM — "is it retrieval, or just keyword matching?"**
Still single-chunk, but one of the easy conditions is deliberately removed:
- the question paraphrases rather than quotes (tests whether embeddings earn their keep), or
- the topic recurs across several chunks and only one has the answer (tests ranking, not just recall).
*Diagnostic value:* this is where pure BM25 and pure vector search start to visibly
diverge. A system that aces Easy and fails Medium is doing lexical matching with extra steps.

**HARD — "does it compose?"**
At least one of:
- **Multi-hop:** the answer requires chaining two facts that never co-occur in one chunk.
- **Aggregation:** the answer is a list/comparison assembled from a passage that chunking may have split.
- **High distractor density:** two or more chunks are near-identical in topic and vocabulary; only one is correct.
*Diagnostic value:* this tier is where chunk size, overlap, and reranking show measurable
effect. It is also where retrieval and generation failures start to look alike from the
outside — which is precisely why your eval must score them **separately**. A wrong answer
here could mean the right chunk wasn't retrieved, or that it was retrieved and the model
ignored it. Those need different fixes.

**TRICK QUESTIONS — "does it fail honestly?"**
The tier that distinguishes a portfolio project from a tutorial. At least one of:
- **Ambiguity:** the question is underspecified and *should* trigger clarification or a hedged answer rather than a confident wrong one.
- **Cross-source conflict:** two books in the corpus cover the same event with different framing or emphasis; a good answer represents both rather than silently picking one.
- **Corpus defect:** the answer lives in OCR-degraded text where the surface form is corrupted, so lexical retrieval structurally cannot find it.
- **Absence / negative:** the corpus genuinely does not contain the answer. The correct behaviour is to say so. **This is the single most under-tested case in amateur RAG projects** — a system with no absence questions in its eval has no evidence it doesn't hallucinate.
- **Trap chunk:** an obviously topically-matching chunk exists that does *not* answer the question, adjacent to a less-obvious one that does.

A note on tier balance: resist the urge to make the set mostly Hard/Trick Questions because
they feel more impressive. Without the Easy tier you cannot distinguish "my reranker is
sophisticated" from "my baseline was broken and I never noticed."

---

## Part 2: A defect worth building an eval around

While inspecting the OCR output of `old_ncert_modern_india_bipan_chandra.pdf`, two real
defects showed up on a single page. Both are **verified**, not hypothetical, and both are
excellent hard cases because they are *structural* failures rather than tuning problems:

1. **Corrupted technical term.** The text reads `"...that of yarah or revenue-farming"`.
   The actual historical term is **ijarah**. Any user asking "what was ijarah?" gets zero
   lexical matches, and the embedding is unlikely to bridge a corrupted token either.
2. **Inconsistent proper-noun spelling within the same page.** The OCR produces both
   `Zulfiqar Khan` and `Zulfigar Khan` (q→g). A query for the correct spelling will miss
   any chunk that happens to contain the corrupted variant.

These give you a genuine, documentable "here is a query that failed, here is the
diagnosis, here is what I changed" narrative — fuzzy matching, OCR post-correction, a
synonym/variant map, or character-n-gram retrieval as candidate fixes. That is exactly
the artifact that survives an interview follow-up.

---

## Part 3: The 50 questions

**Status column:** `VERIFIED` = I read the underlying source text and confirmed the answer
exists. `CANDIDATE` = plausible given the book's known scope, but **you must confirm
against your extracted text before it enters the eval set.**

Do not skip that verification. A question whose "gold answer" came from a model's prior
knowledge rather than your corpus will silently score your system on the wrong thing.

### EASY (12)

| # | Question | Target | Tests | Status |
|---|---|---|---|---|
| E1 | Which two brothers were given the offices of wazir and mir bakshi after Farrukh Siyar's victory? | Old NCERT Modern | Distinctive proper nouns, single chunk | VERIFIED |
| E2 | In which month and year did Nizam-ul-Mulk relinquish his office and march south? | Old NCERT Modern | Exact date, contiguous span | VERIFIED |
| E3 | Who was given the title Maharaja and appointed Governor of Gujarat under Jahandar Shah? | Old NCERT Modern | Single fact, high lexical overlap | VERIFIED |
| E4 | Who was the Chief Advisor for the Class XII *Themes in Indian History* textbook? | Class 12 front matter | Corpus-hygiene probe (see note below) | VERIFIED |
| E5 | What is the title of Theme One in *Themes in Indian History Part I*? | Class 12 Part I | Structural metadata retrieval | VERIFIED |
| E6 | In which year was Jahandar Shah defeated at Agra? | Old NCERT Modern | Date extraction | VERIFIED |
| E7 | What material were Harappan seals typically made from? | `class6_our_pasts_1.txt` (line ~1955) | Single-fact, distinctive term | VERIFIED |
| E8 | What was the jizyah? | `old_ncert_medieval_india_satish_chandra.txt` (line ~9173, printed p.108) | Definition lookup | VERIFIED |
| E9 | Where was the capital of the Vijayanagara empire located? | `lehs203.txt` (Theme 7, **Part II** — corrected from earlier guess of Part III) | Named-entity lookup | VERIFIED |
| E10 | What does the term *sardeshmukhi* refer to? | `class7_our_pasts_2.txt` (line ~7173, glossary box — **not** Old NCERT Modern as originally guessed) | Technical term definition | VERIFIED |
| E11 | Which Mughal ruler's reign lasted roughly thirty years from 1719 to 1748? | Old NCERT Modern | Date-range fact | VERIFIED |
| E12 | What is a stupa? | `class6_our_pasts_1_c00118` ("the word stupa means a mound") | Definition; 25 chunks mention stupas, so more distractor density than a typical Easy | VERIFIED |

> **On E4/E5:** these target front matter, not history. They are deliberate *corpus
> hygiene probes*. If your system answers them, front matter (acknowledgements, publication
> team, contents pages) is in your index and is competing with real content for retrieval
> slots. Many pipelines should arguably strip it. Either choice is defensible — but you
> should know which one you made, and these two questions are how you find out.

### MEDIUM (14)

| # | Question | Target | Tests | Status |
|---|---|---|---|---|
| M1 | Why did Zulfiqar Khan reverse Aurangzeb's policies toward Hindu chieftains? | Old NCERT Modern | Paraphrase gap — text states policy, not labelled "why" | VERIFIED |
| M2 | What system replaced fixed-rate land revenue collection, and why was it harmful to peasants? | Old NCERT Modern | Causal link stated indirectly; also hits the `yarah` defect | VERIFIED |
| M3 | How did the Saiyid brothers attempt to win over Rajput princes? | Old NCERT Modern | Aggregation of 2-3 named actions in one passage | VERIFIED |
| M4 | What made the Emperor distrust his own wazir despite not dismissing him? | Old NCERT Modern | Narrative causation, not a stated fact | VERIFIED |
| M5 | Why do historians consider oral sources controversial as historical evidence? | Cl. 12 Part I intro | Synonym gap ("oral evidence" vs "oral sources") | VERIFIED |
| M6 | What changed about the *conception of a source* as history broadened? | Cl. 12 Part I intro | Abstract paraphrase | VERIFIED |
| M7 | What kinds of sources would a historian need to study rebels, and why are official records insufficient? | Cl. 12 Part I intro | Two-part answer in one passage | VERIFIED |
| M8 | How was Harappan town planning organised? | `class6_our_pasts_1.txt` + `lehs101.txt` (both confirmed substantial coverage) | Distractor density — topic in 2+ books | VERIFIED |
| M9 | What was the significance of Alexander Cunningham's encounter with a Harappan seal? | Cl. 12 Part I intro | Requires understanding, not keyword match | VERIFIED |
| M10 | How did the Bhakti and Sufi traditions differ in their approach to devotion? | `lehs202.txt` (Theme 6) | Comparison within a chapter | LOCATION VERIFIED — write gold answer only after reading full chapter; this is synthesis, not a single quote |
| M11 | What role did zamindars play in Mughal agrarian society? | `lehs204.txt` (Theme 8, 54 mentions) | Recurring term, one authoritative passage | VERIFIED |
| M12 | Why did NCERT rationalise textbook content in this edition? | `kehs1ps.txt`, `lehs1ps.txt`, `lehs2ps.txt`, `lehs3ps.txt` | Appears identically in 4 books — **not literally every book**, corrected: this boilerplate is specific to the Class 11-12 ZIP-sourced front matter and absent from the Vajiram-sourced whole-book files | VERIFIED |
| M13 | What distinguishes 'Themes in Indian History' in scope from the Class IX-X books? | Cl. 12 Part I intro | Cross-referential, stated obliquely | VERIFIED |
| M14 | What were the main features of the Mauryan administration? | `old_ncert_ancient_india_rs_sharma.txt` (multiple passages — **not** Cl. 6 or Cl. 12 T2 as originally guessed, both had zero substantive hits) | Aggregation across a section | VERIFIED |

### HARD (14)

| # | Question | Target | Tests | Status |
|---|---|---|---|---|
| H1 | The Deccan's chauth and sardeshmukhi were granted to the Marathas twice under different arrangements. How did the 1711 arrangement differ from the later Saiyid agreement? | Old NCERT Modern | **Two near-identical chunks, same vocabulary** — classic reranker test | VERIFIED |
| H2 | List every concession Zulfiqar Khan made to non-Muslim chiefs, and identify the one group he continued to suppress. | Old NCERT Modern | Aggregation + a negative clause at passage end (chunk-boundary risk) | VERIFIED |
| H3 | Nizam-ul-Mulk both served as wazir and later founded an independent state. What position did he hold, and what does the text give as his motive for leaving? | Old NCERT Modern | Multi-hop across a page | VERIFIED |
| H4 | Why did the murder of Farrukh Siyar ultimately weaken rather than strengthen the Saiyid brothers? | Old NCERT Modern | Multi-hop: consequence stated separately from event | VERIFIED |
| H5 | How does each book describe the end of the Harappan urban phase? | `class6_our_pasts_1.txt` (line ~2195, brief, no word "decline" used — says "rulers lost control") + `lehs101.txt` (line ~30, uses "decline" explicitly, much more detail) | Cross-book, different depth **and vocabulary** — reworded from original to not assume both use the word "decline" | VERIFIED, reworded |
| H6 | Compare how 1857 is framed in the Old NCERT *Modern India* versus Class XII Theme Ten. | `old_ncert_modern_india_bipan_chandra.txt` (68 mentions) + `lehs302.txt` (Theme 10, 59 mentions) | Cross-book, different scholarly eras | VERIFIED |
| H7 | Which themes in *Themes in Indian History* were authored by contributors based outside India? | Cl. 12 front matter | Multi-hop: name→affiliation→theme number | VERIFIED |
| H8 | What connects the Harappan seals discussed in Theme One with the "historian's craft" argument in the book's introduction? | Cl. 12 Part I | Cross-chapter conceptual link | VERIFIED |
| H9 | Trace how a single Mughal revenue practice contributed to both peasant oppression and state financial decline. | Old NCERT Modern | Multi-hop causal chain, 2 passages | VERIFIED |
| H10 | Which Rajput ruler appears in accounts of both Jahandar Shah's and the Saiyid brothers' conciliation policies? | Old NCERT Modern | Entity resolution across two passages (Ajit Singh) | VERIFIED |
| H11 | How did Gandhi's methods in Civil Disobedience differ from earlier nationalist strategies? | `lehs303.txt` (Theme 11, 13 mentions) + `class8_our_pasts_3.txt` (79 mentions, richer source) | Cross-book synthesis | VERIFIED |
| H12 | What evidence do historians use to reconstruct caste and kinship in early societies, and what are its limits? | `lehs103.txt` (Theme 3, line ~163 — **not** `lehs102.txt` as originally guessed, which had zero hits) | Two-part, second half often in a separate section | VERIFIED |
| H13 | Across the corpus, which periods are covered by more than one book, and how do treatments differ in depth? | Whole corpus — measured: Mughal 16 books, 1857 13, Gandhi 13, Harappan 11, Gupta 11, Vijayanagara 9, Mauryan 9, Delhi Sultanate 6 | Aggregation over many chunks; depth gap is real (Mauryan: 87 chunks in Old NCERT Ancient vs 13 in Class 6) | VERIFIED |
| H14 | Why did the Mughal artillery and imperial prestige not prevent decline under Muhammad Shah? | Old NCERT Modern | Concessive reasoning; the "why not" is implicit | VERIFIED |

### TRICK QUESTIONS (10)

| # | Question | Target | Tests | Status |
|---|---|---|---|---|
| X1 | What was *ijarah* and how did it affect peasants? | Old NCERT Modern | **OCR defect: corpus spells it `yarah`.** Lexical retrieval structurally fails | VERIFIED |
| X2 | What policies did Zulfiqar Khan pursue regarding jagirs and mansabdars? | Old NCERT Modern | **OCR defect: name appears as both `Zulfiqar` and `Zulfigar`** on the same page | VERIFIED |
| X3 | How was the Harappan script deciphered? | Cl. 12 T1 | **Absence question** — the script is undeciphered; correct behaviour is to say the corpus doesn't support the premise | CANDIDATE |
| X4 | What does the corpus say about the Indus Valley's trade with Mesopotamia during the Gupta period? | Cl. 12 | **False-premise absence** — anachronistic; must not confabulate | CANDIDATE |
| X5 | "What did the textbook say about the revolt?" | Whole corpus — ≥5 distinct referents measured: Revolt of 1857 (27 chunks), Santhal (17), Deccan Riots (6), Munda/Birsa (3), European peasant revolts (2) | **Ambiguity** — genuinely underspecified; expected behaviour is `clarify`, not a confident pick | VERIFIED |
| X6 | Who was 'the Emperor' who intrigued against his own ministers? | Gold: `old_ncert_modern_india_bipan_chandra_c00009` → **Muhammad Shah** (r. 1719–1748). Trap: `..._c00004` ("the Emperor too did not give him...", about nobles working against Zulfiqar Khan) matches the phrasing but does not answer | **Trap chunk** — two chunks in the same chapter match; only one answers | VERIFIED |
| X7 | How many pages does the Class XII Part I textbook devote to Theme Two? | Two independent paths: `lehs1ps_c00012` (contents listing: THEME TWO 28, THEME THREE 53) **or** `lehs102_c00000` (opens at 28) + `lehs103_c00000` (opens at 53). Answer 53−28 = 25 | Arithmetic over retrieved metadata, not extraction; has a cross-check path | VERIFIED |
| X8 | ~~Do the Old NCERT and current NCERT books agree on the causes of Mughal decline?~~ **RETIRED — see note below.** | — | — | FAILED VERIFICATION |
| X9 | What is the price printed on the Class XII Part I textbook, and in which reprint year? | Gold: `lehs1ps_c00000` (₹105.00 + "April 2026 Vaishakha 1948"). Distractors: `lehs2ps_c00000` and `lehs3ps_c00001` show the same ₹105.00 for Parts II and III | Multi-field metadata retrieval + disambiguation across three near-identical front-matter chunks | VERIFIED |
| X10 | Summarise what the corpus says about Ramachandra Guha's account of the nationalist movement. | 6 chunks, three different roles: `lehs1ps_c00009` (front matter, listed as Theme 11 contributor), `class9_..._c00213` (cited as author of *A Corner of a Foreign Field* in the **cricket** chapter), `class9_..._c00122` (cited in the **forest** chapter) | **Trap, stronger than designed** — the corpus never gives Guha's account of the nationalist movement; the retrievable Guha content is about cricket and forests | VERIFIED |

---

## Part 3.5: Verification log

This is the evidence behind the `VERIFIED` labels above — not just an assertion, but
what was actually checked, so this is falsifiable by anyone reading the repo.

**Method:** grep for distinctive terms/names in the actual extracted `.txt` files, then
read the surrounding lines in context to confirm the fact is genuinely stated (not just
that a keyword co-occurs nearby). Verified against your full uploaded corpus (31 files)
on the date these questions were checked.

**Corrections found during verification** (originally-guessed source book was wrong):

| # | Originally guessed | Actual verified source |
|---|---|---|
| E9 | Cl. 12 Theme 7, "Part III" | `lehs203.txt` — Theme 7 is in **Part II** |
| E10 | Old NCERT Modern | `class7_our_pasts_2.txt` — a glossary box, not the Old NCERT text |
| M11 | Cl. 12 T8 (right book, unconfirmed) | Confirmed: `lehs204.txt`, 54 mentions |
| M14 | Cl. 6 / Cl. 12 T2 | Neither had relevant content — actual source is `old_ncert_ancient_india_rs_sharma.txt` |
| H12 | `lehs102.txt` | Zero hits in that file — actual source is `lehs103.txt` |

**Questions retired because verification failed** (kept here instead of silently
deleted, so the failure itself is on record):

- **X8** ("do Old NCERT and Cl. 12 T8 agree on Mughal decline causes?") — `lehs204.txt`
  (Theme 8, "Peasants, Zamindars and the State") has **zero** mentions of "decline."
  It's a structural/agrarian analysis, not a decline narrative — there is no comparable
  content to conflict with the Old NCERT chapter. The premise of the question doesn't
  hold against the actual corpus. This is itself a useful thing to have found: it's a
  reminder that a plausible-sounding cross-source question still needs the same
  line-by-line check as any other.

**Rewording made necessary by verification:**

- **H5** originally assumed both Class 6 and Class 12 use the word "decline" for the
  end of Harappan urban life. Class 6 does cover it, but with different language
  ("rulers lost control," sites "abandoned") — no occurrence of "decline" at all. Reworded
  to ask about "the end of the Harappan urban phase" so a good retriever isn't
  penalized for a wording mismatch that isn't the actual point of the question.

**Second verification pass — all remaining questions now checked.** E12, H13, X5, X6,
X7, X9 and X10 were verified against `chunks.jsonl` (2,909 chunks) rather than the raw
`.txt` files, which has the advantage that gold chunk IDs fall out of the same search.
All seven survived; none had to be retired. Two turned out stronger than designed:

- **X6** — the gold answer is Muhammad Shah (`..._c00009`), and there is a genuine trap
  chunk two chunks earlier (`..._c00004`) that matches the question's phrasing
  ("the Emperor too did not give him...") while being about Zulfiqar Khan's enemies.
- **X10** — Ramachandra Guha appears in three unrelated roles across the corpus
  (Theme 11 contributor in front matter; cited author in the Class 9 cricket chapter;
  cited again in the Class 9 forest chapter). The corpus contains no Guha account of
  the nationalist movement at all, so the correct behaviour is closer to `abstain`
  than `answer`.

(M13 is solid — drawn from the "Defining the Focus of Study" passage in the Class 12
front matter, which was read in full.)



For each question you keep, record:

```json
{
  "id": "H1",
  "tier": "hard",
  "question": "...",
  "gold_answer": "...",
  "gold_chunk_ids": ["lehs101_c14", "lehs101_c15"],
  "gold_source_files": ["old_ncert_modern_india_bipan_chandra.txt"],
  "gold_pages": [3, 4],
  "tests": ["multi_hop", "distractor_density"],
  "expected_behaviour": "answer",
  "verified_by_hand": true,
  "notes": "..."
}
```

Two fields do the heavy lifting and are worth being strict about:

- **`gold_chunk_ids`** must be filled in *after* chunking, by looking at your actual index —
  not guessed. This is what lets you score retrieval independently of generation
  (recall@k, MRR) instead of only judging the final prose.
- **`expected_behaviour`** should be one of `answer` / `abstain` / `clarify`. Absence
  questions (X3, X4) expect `abstain`; ambiguous ones (X5) expect `clarify`. Scoring a
  confident answer to X3 as "wrong" is much weaker than scoring it as
  "**failed to abstain**" — the latter is a hallucination metric, and it is the number
  most worth putting in your README.

### Suggested reporting shape

Report retrieval and generation separately, broken out by tier:

| Tier | n | Recall@5 | MRR | Groundedness | Correct abstentions |
|---|---|---|---|---|---|
| Easy | 12 | | | | n/a |
| Medium | 14 | | | | n/a |
| Hard | 14 | | | | n/a |
| Trick questions | 10 | | | | x/4 |

A flat aggregate number ("87% accuracy") hides exactly the information that makes this a
portfolio piece. The per-tier breakdown is the result; the aggregate is a footnote.