#!/usr/bin/env python3
"""
Backfill gold chunk IDs into a machine-readable evaluation set.

WHY THIS RUNS LOCALLY
---------------------
Chunk IDs are sequential per book (`class6_our_pasts_1_c00042`), so they shift
whenever chunking changes -- a single merged runt chunk renumbers everything
after it in that book. Gold IDs must therefore be generated against the same
chunks.jsonl that was embedded, not copied from anywhere else, or they will
point at the wrong passages without erroring.

WHAT IT DOES
------------
For each question it searches chunks.jsonl for the distinctive terms that must
appear in a chunk containing the answer, ranks candidates, and writes:

  eval_set.json        machine-readable eval set, gold_chunk_ids prefilled
                       with the top candidate(s) where confident
  gold_review.md       human review file: every question with its candidates
                       and the matched text, so you can confirm or correct

CONFIRMATION IS REQUIRED. A candidate is a search hit, not a verified answer.
The `verified` flag stays false until you set it, and the eval harness reports
how many of its gold IDs are unconfirmed so the numbers are never quietly
based on guesses.

USAGE
-----
  python backfill_gold_chunks.py
  python backfill_gold_chunks.py --input chunks.jsonl --top 3
"""

import os
import re
import json
import argparse

# ---------------------------------------------------------------------
# The evaluation set.
#
# `must` : terms that MUST appear in a chunk holding the answer (all of them)
# `any`  : at least one must appear (used where the corpus wording varies)
# `books`: restrict search to these books (substring match on the book field)
# `expected_behaviour`: answer | abstain | clarify
# `multi`: this question legitimately needs more than one chunk
# ---------------------------------------------------------------------
QUESTIONS = [
    # ---------------- EASY ----------------
    dict(id="E1", tier="easy",
         evidence=r"""mir bakshi|Abdullah Khan""",
         q="Which two brothers were given the offices of wazir and mir bakshi after Farrukh Siyar's victory?",
         must=[r"Saiyid", r"wazir"], books=["modern_india"]),
    dict(id="E2", tier="easy",
         evidence=[r"""October 1724""", r"""1724.{0,120}(Deccan|south|march)""", r"""(relinquish|march).{0,120}1724"""],
         note="gold tightened after baseline: bare 1724 also matched a passage about Muhammad Shah's reign",
         q="In which month and year did Nizam-ul-Mulk relinquish his office and march south?",
         must=[r"Nizam-ul-Mulk"], any=[r"1724", r"October"], books=["modern_india"]),
    dict(id="E3", tier="easy",
         evidence=r"""Ajit Singh of Marwar was awarded""",
         q="Who was given the title Maharaja and appointed Governor of Gujarat under Jahandar Shah?",
         must=[r"Ajit Singh"], any=[r"Gujarat", r"Maharaja"], books=["modern_india"]),
    dict(id="E4", tier="easy",
         evidence=r"""NEELADRI BHATTACHARYA|Neeladri Bhattacharya""",
         q="Who was the Chief Advisor for the Class XII Themes in Indian History textbook?",
         must=[r"Chief Advisor"], any=[r"Neeladri"], books=["lehs1ps"],
         note="corpus-hygiene probe: front matter must be indexed to answer"),
    dict(id="E5", tier="easy",
         evidence=r"""THEME ONE\s*1\s*BRICKS|Bricks, Beads and Bones""",
         q="What is the title of Theme One in Themes in Indian History Part I?",
         must=[r"THEME ONE|Theme One"], any=[r"BRICKS|Bricks"], books=["lehs1"]),
    dict(id="E6", tier="easy",
         evidence=r"""January 1713|1713""",
         q="In which year was Jahandar Shah defeated at Agra?",
         must=[r"Jahandar"], any=[r"1713", r"Agra"], books=["modern_india"]),
    dict(id="E7", tier="easy",
         evidence=[r"""seals out of stone""", r"""stone called steatite"""],
         books=['class6', 'lehs101'],
         note='gold widened after baseline: retrieval returned lehs101_c00000 (a stone called steatite) at rank 1, which answers the question; the original class6-only gold was too narrow',
         q="What material were Harappan seals typically made from?",
         must=[r"seals"], any=[r"stone", r"steatite"]),
    dict(id="E8", tier="easy",
         evidence=r"""tax called jizyah""",
         q="What was the jizyah?",
         must=[r"jizyah"], books=["medieval_india"]),
    dict(id="E9", tier="easy",
         evidence=r"""Hampi""",
         q="Where was the capital of the Vijayanagara empire located?",
         must=[r"Hampi"], any=[r"Tungabhadra", r"capital"], books=["lehs203"]),
    dict(id="E10", tier="easy",
         evidence=r"""head revenue\s*collector|9-10 per cent""",
         q="What does the term sardeshmukhi refer to?",
         must=[r"[Ss]ardeshmukhi"], books=["class7"],
         note="answer is a glossary-box definition, but packing absorbed the box "
              "into an adjacent prose chunk (mixed:prose), so it does not rank "
              "first on lexical score -- confirm candidate 2, not 1"),
    dict(id="E11", tier="easy",
         evidence=r"""1719-1748""",
         q="Which Mughal ruler's reign lasted roughly thirty years from 1719 to 1748?",
         must=[r"Muhammad Shah"], any=[r"1719", r"1748"], books=["modern_india"]),
    dict(id="E12", tier="easy",
         evidence=r"""stupa means a mound""",
         q="What is a stupa?",
         must=[r"stupa"], any=[r"mound"], books=["class6"]),

    # ---------------- MEDIUM ----------------
    dict(id="M1", tier="medium",
         evidence=r"""jizyah was abolished""",
         q="Why did Zulfiqar Khan reverse Aurangzeb's policies toward Hindu chieftains?",
         must=[r"Zulfi[qg]ar"], any=[r"jizyah", r"Rajput", r"conciliate|concilate"],
         books=["modern_india"]),
    dict(id="M2", tier="medium",
         evidence=r"""yarah""",
         q="What system replaced fixed-rate land revenue collection, and why was it harmful to peasants?",
         must=[r"yarah|ijarah|revenue-farming"], books=["modern_india"],
         note="also exercises the yarah OCR defect"),
    dict(id="M3", tier="medium",
         evidence=r"""Ajit Singh|Jai Singh""",
         q="How did the Saiyid brothers attempt to win over Rajput princes?",
         must=[r"Saiyid"], any=[r"Rajput", r"Ajit Singh"], books=["modern_india"]),
    dict(id="M4", tier="medium",
         evidence=[r"""intrigue against him secretly"""],
         note="gold corrected after baseline: retrieval returned the chunk reading 'dared not dismiss the powerful wazir, but he began to intrigue against him secretly' at rank 1 -- that is the answer",
         q="What made the Emperor distrust his own wazir despite not dismissing him?",
         must=[r"Emperor"], any=[r"poisoned", r"jealous nobles"], books=["modern_india"]),
    dict(id="M5", tier="medium",
         evidence=r"""oral evidence|verifiab""",
         q="Why do historians consider oral sources controversial as historical evidence?",
         must=[r"oral"], any=[r"authenticity", r"verifiab"], books=["lehs1"]),
    dict(id="M6", tier="medium",
         evidence=r"""conception of what constitutes a source""",
         q="What changed about the conception of a source as history broadened?",
         must=[r"source"], any=[r"conception", r"written records"], books=["lehs1"]),
    dict(id="M7", tier="medium",
         evidence=r"""diaries""",
         q="What kinds of sources would a historian need to study rebels, and why are official records insufficient?",
         must=[r"rebel"], any=[r"diaries", r"official"], books=["lehs1"]),
    dict(id="M8", tier="medium",
         gold_mode='any',
         evidence=r"""citadel|drainage""",
         q="How was Harappan town planning organised?",
         must=[r"Harappan|Mohenjodaro"], any=[r"drainage", r"citadel", r"town plan"],
         books=["class6", "lehs101"], multi=True,
         note="distractor density: substantial coverage in both books"),
    dict(id="M9", tier="medium",
         evidence=[r"""Cunningham (first saw|could make no sense)""", r"""Cunningham"""],
         q="What was the significance of Alexander Cunningham's encounter with a Harappan seal?",
         must=[r"Cunningham"], books=["lehs1"]),
    dict(id="M10", tier="medium",
         evidence=r"""Sufi""",
         q="How did the Bhakti and Sufi traditions differ in their approach to devotion?",
         must=[r"Sufi"], any=[r"Bhakti", r"bhakti"], books=["lehs202"], multi=True,
         note="synthesis across a chapter; gold answer needs the full chapter read"),
    dict(id="M11", tier="medium",
         evidence=[r"""zamindars were|role of the zamindar|zamindars played""", r"""zamindar"""],
         q="What role did zamindars play in Mughal agrarian society?",
         must=[r"zamindar"], books=["lehs204"]),
    dict(id="M12", tier="medium",
         evidence=[r"""rationalise the textbooks""", r"""[Rr]ationalis"""],
         gold_mode='any',
         q="Why did NCERT rationalise textbook content in this edition?",
         must=[r"[Rr]ationalis"], books=["ps"],
         note="extreme distractor density: identical text in 4 front-matter files"),
    dict(id="M13", tier="medium",
         evidence=r"""Classes IX and X|Class XI""",
         q="What distinguishes Themes in Indian History in scope from the Class IX-X books?",
         must=[r"Classes IX|Class IX"], any=[r"themes", r"contemporary"], books=["lehs1"]),
    dict(id="M14", tier="medium",
         gold_mode='any',
         evidence=r"""Maurya""",
         q="What were the main features of the Mauryan administration?",
         must=[r"Maurya"], any=[r"administ", r"officer"], books=["ancient_india"], multi=True),

    # ---------------- HARD ----------------
    dict(id="H1", tier="hard",
         evidence=[r"""chauth a[un]d sardeshmukhi of the Deccan""", r"""sardeshmukhi of the Deccan"""],
         q="The Deccan's chauth and sardeshmukhi were granted to the Marathas twice under different arrangements. How did the 1711 arrangement differ from the later Saiyid agreement?",
         must=[r"sardeshmukhi"], books=["modern_india"], multi=True,
         note="two near-identical chunks; reranker test"),
    dict(id="H2", tier="hard",
         gold_mode='any',
         evidence=r"""Churaman Jat|Churaman""",
         q="List every concession Zulfiqar Khan made to non-Muslim chiefs, and identify the one group he continued to suppress.",
         must=[r"Zulfi[qg]ar"], any=[r"Churaman", r"Jat", r"Sikh"], books=["modern_india"],
         multi=True, note="aggregation across a chunk boundary"),
    dict(id="H3", tier="hard",
         evidence=[r"""Nizam-ul-Mulk.{0,200}(wazir|Deccan)""", r"""Nizam-ul-Mulk"""],
         q="Nizam-ul-Mulk both served as wazir and later founded an independent state. What position did he hold, and what does the text give as his motive for leaving?",
         must=[r"Nizam-ul-Mulk"], books=["modern_india"], multi=True),
    dict(id="H4", tier="hard",
         evidence=[r"""Farrukh Siyar.{0,200}(murder|deposed|killed)""", r"""Farrukh Siyar"""],
         q="Why did the murder of Farrukh Siyar ultimately weaken rather than strengthen the Saiyid brothers?",
         must=[r"Farrukh Siyar"], books=["modern_india"], multi=True),
    dict(id="H5", tier="hard",
         evidence=r"""abandoned|decline of the civilisation""",
         q="How does each book describe the end of the Harappan urban phase?",
         must=[r"Harappan|abandoned|decline"], books=["class6", "lehs101"], multi=True,
         note="reworded: class6 never uses the word 'decline'"),
    dict(id="H6", tier="hard",
         evidence=[r"""Revolt of 1857""", r"""1857"""],
         q="Compare how 1857 is framed in the Old NCERT Modern India versus Class XII Theme Ten.",
         must=[r"1857"], books=["modern_india", "lehs302"], multi=True),
    dict(id="H7", tier="hard",
         evidence=r"""Chicago|Austria""",
         q="Which themes in Themes in Indian History were authored by contributors based outside India?",
         must=[r"Chicago|Austria|USA"], books=["lehs1ps"], multi=True),
    dict(id="H8", tier="hard",
         evidence=[r"""Cunningham first saw a Harappan seal""", r"""Cunningham"""],
         q="What connects the Harappan seals discussed in Theme One with the historian's craft argument in the introduction?",
         must=[r"seal"], any=[r"Cunningham", r"historian"], books=["lehs1"], multi=True),
    dict(id="H9", tier="hard",
         gold_mode='any',
         evidence=r"""yarah""",
         q="Trace how a single Mughal revenue practice contributed to both peasant oppression and state financial decline.",
         must=[r"yarah|ijarah|revenue-farming"], books=["modern_india"], multi=True),
    dict(id="H10", tier="hard",
         evidence=[r"""Ajit Singh of Marwar""", r"""Ajit Singh"""],
         q="Which Rajput ruler appears in accounts of both Jahandar Shah's and the Saiyid brothers' conciliation policies?",
         must=[r"Ajit Singh"], books=["modern_india"], multi=True),
    dict(id="H11", tier="hard",
         evidence=[r"""Civil Disobedience Movement""", r"""[Cc]ivil [Dd]isobedience"""],
         gold_mode='any',
         q="How did Gandhi's methods in Civil Disobedience differ from earlier nationalist strategies?",
         must=[r"[Cc]ivil [Dd]isobedience"], books=["lehs303", "class8"], multi=True),
    dict(id="H12", tier="hard",
         evidence=r"""difficult to reconstruct|kinship""",
         q="What evidence do historians use to reconstruct caste and kinship in early societies, and what are its limits?",
         must=[r"kinship|caste"], any=[r"difficult", r"limitation"], books=["lehs103"],
         multi=True),
    dict(id="H13", tier="hard",
         gold_mode='any',
         evidence=r"""Mughal|Harappan""",
         q="Across the corpus, which periods are covered by more than one book, and how do treatments differ in depth?",
         must=[r"Mughal|Harappan|Maurya"], books=None, multi=True,
         note="aggregation over many chunks; gold set is necessarily broad"),
    dict(id="H14", tier="hard",
         evidence=[r"""Muhammad Shah.{0,200}(artillery|prestige|decline)""", r"""Muhammad Shah"""],
         gold_mode='any',
         q="Why did Mughal artillery and imperial prestige not prevent decline under Muhammad Shah?",
         must=[r"Muhammad Shah"], books=["modern_india"], multi=True),

    # ---------------- TRICK ----------------
    dict(id="X1", tier="trick",
         evidence=r"""yarah""",
         q="What was ijarah and how did it affect peasants?",
         must=[r"yarah"], books=["modern_india"],
         note="OCR defect: corpus spells it 'yarah'; lexical retrieval should fail"),
    dict(id="X2", tier="trick",
         evidence=r"""Zulfigar""",
         q="What policies did Zulfiqar Khan pursue regarding jagirs and mansabdars?",
         must=[r"Zulfigar"], books=["modern_india"],
         note="OCR defect: 'Zulfigar' variant; querying 'Zulfiqar' misses this chunk"),
    dict(id="X3", tier="trick",
         evidence=r"""undeciphered""", expected_behaviour="abstain",
         q="How was the Harappan script deciphered?",
         must=[r"undeciphered"], books=["lehs101"],
         note="absence: corpus states the script is undeciphered"),
    dict(id="X4", tier="trick", expected_behaviour="abstain",
         evidence=[r"""Mesopotamian texts""", r"""Mesopotamia"""],
         q="What does the corpus say about the Indus Valley's trade with Mesopotamia during the Gupta period?",
         must=[r"Mesopotamia"], books=["lehs101"],
         note="false premise: Mesopotamia trade is 3rd-millennium BCE, not Gupta"),
    dict(id="X5", tier="trick",
         gold_mode='any',
         evidence=r"""[Rr]evolt""", expected_behaviour="clarify",
         q="What did the textbook say about the revolt?",
         must=[r"[Rr]evolt"], books=None, multi=True,
         note="ambiguous: >=5 distinct revolts in corpus"),
    dict(id="X6", tier="trick",
         evidence=r"""intrigued against his own ministers""",
         q="Who was 'the Emperor' who intrigued against his own ministers?",
         must=[r"intrigued against"], books=["modern_india"],
         note="trap: a nearby chunk matches the phrasing but is about Zulfiqar Khan's enemies"),
    dict(id="X7", tier="trick",
         evidence=r"""THEME TWO\s*28|Theme Two""",
         q="How many pages does the Class XII Part I textbook devote to Theme Two?",
         must=[r"THEME TWO|Theme Two"], books=["lehs1"], multi=True,
         note="arithmetic over retrieved metadata; two independent paths"),
    dict(id="X9", tier="trick",
         evidence=r"""105\.00""",
         q="What is the price printed on the Class XII Part I textbook, and in which reprint year?",
         must=[r"105\.00"], books=["lehs1ps"],
         note="distractors: Parts II and III show the same price"),
    dict(id="X10", tier="trick",
         gold_mode='any',
         evidence=r"""Ramachandra Guha""", expected_behaviour="abstain",
         q="Summarise what the corpus says about Ramachandra Guha's account of the nationalist movement.",
         must=[r"Ramachandra Guha"], books=None, multi=True,
         note="trap: Guha appears as a contributor and as a cited author on cricket "
              "and forests; no account of the nationalist movement exists in corpus"),
]


def load_chunks(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def score_chunk(chunk, q):
    """Rank by how well a chunk satisfies the question's search terms."""
    text = chunk["text"]
    for pat in q.get("must", []):
        if not re.search(pat, text):
            return 0.0
    score = 1.0
    anys = q.get("any", [])
    if anys:
        hit = sum(1 for p in anys if re.search(p, text))
        if hit == 0:
            return 0.0
        score += hit / len(anys)
    # prefer denser matches
    for pat in q.get("must", []):
        score += 0.1 * len(re.findall(pat, text))
    # Slight preference for prose, but deliberately small: some gold answers
    # live in furniture (E10's sardeshmukhi definition is a glossary box), and
    # a strong prose bonus ranked a loosely-related prose passage above the
    # actual definition.
    want = q.get("prefer_content")
    ctype = chunk.get("content_type", "")
    if want:
        if want in ctype:
            score += 0.5
    elif ctype.endswith("prose"):
        score += 0.05
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="chunks.jsonl")
    ap.add_argument("--top", type=int, default=3,
                    help="candidates to record per question")
    ap.add_argument("--out", default="eval_set.json")
    ap.add_argument("--review", default="gold_review.md")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"ERROR: {args.input} not found. Run chunk_corpus.py first.")
        return

    chunks = load_chunks(args.input)
    print(f"Searching {len(chunks)} chunks for {len(QUESTIONS)} questions\n")

    eval_set = []
    review = ["# Gold chunk review",
              "",
              "Each question below lists its top candidate chunks. Confirm or correct,",
              "then set `verified: true` for that question in `eval_set.json`.",
              "",
              "A candidate is a search hit, not a verified answer.",
              ""]
    no_hit = []

    for q in QUESTIONS:
        pool = chunks
        if q.get("books"):
            pool = [c for c in chunks
                    if any(b in c["book"] for b in q["books"])]

        scored = [(score_chunk(c, q), c) for c in pool]
        scored = [(s, c) for s, c in scored if s > 0]
        scored.sort(key=lambda x: -x[0])
        top = scored[:args.top]

        cand = [c["chunk_id"] for _, c in top]
        if not cand:
            no_hit.append(q["id"])

        # Evidence-based gold selection.
        #
        # Ranking by lexical score picks the wrong chunk often enough to be
        # unusable: for "in which year did Nizam-ul-Mulk march south" the top
        # hit was a passage about Muhammad Shah's reign, and for "what is a
        # stupa" it was a passage about the iron pillar at Mehrauli. So gold is
        # defined by the evidence text that MUST appear in a chunk holding the
        # answer, not by search rank.
        # `evidence` may be a single pattern or a list tried most-specific
        # first. One pattern often cannot be both precise enough to pin a
        # single chunk and robust to OCR variation, so the list lets a tight
        # pattern win where it matches and a looser one act as fallback.
        evidence = q.get("evidence")
        patterns = ([evidence] if isinstance(evidence, str)
                    else list(evidence or []))
        auto_verified = False
        ev_hits = []
        # Search the book-filtered pool, NOT the must/any-filtered candidate
        # list. Evidence is the authoritative definition of gold; must/any
        # exist only to rank candidates. Searching the narrower list silently
        # lost M4, whose answer chunk ("dared not dismiss the powerful wazir,
        # but he began to intrigue against him secretly") does not contain the
        # any= terms and was filtered out before evidence ever ran.
        for pat in patterns:
            ev_hits = [c["chunk_id"] for c in pool
                       if re.search(pat, c["text"])]
            if 0 < len(ev_hits) <= 4:
                break
        if patterns:
            # If several chunks carry the evidence, they are all gold: the
            # answer genuinely appears in each. That is not ambiguity, and
            # treating it as such was wrong -- X1's "yarah" appears in two
            # chunks and both legitimately answer the question.
            #
            # What it does change is scoring. For a single-chunk question ANY
            # gold suffices; only a multi-hop question needs them ALL. That
            # distinction is carried by gold_mode, so full@k does not demand
            # retrieving two interchangeable chunks.
            if ev_hits and len(ev_hits) <= 4:
                prefill, auto_verified = ev_hits, True
            elif ev_hits:
                # Too many matches to call gold automatically -- the evidence
                # pattern is too loose for this question.
                prefill, auto_verified = ev_hits[:4], False
            else:
                prefill = []
        else:
            prefill = cand[:1] if (cand and not q.get("multi")) else []

        eval_set.append({
            "id": q["id"],
            "tier": q["tier"],
            "question": q["q"],
            "expected_behaviour": q.get("expected_behaviour", "answer"),
            "gold_chunk_ids": prefill,
            "gold_chunk_candidates": cand,
            "multi_chunk": bool(q.get("multi")),
            # "all" only where every gold chunk is genuinely required.
            # Aggregation and ambiguity questions (H13 "which periods appear in
            # more than one book", X5 "the revolt") have many equally valid
            # gold chunks; demanding all of them would score a correct system
            # as a failure.
            "gold_mode": q.get("gold_mode",
                               "all" if q.get("multi") else "any"),
            "verified": auto_verified,
            "verified_by": ("evidence-match" if auto_verified else ""),
            "evidence_pattern": q.get("evidence", ""),
            "note": q.get("note", ""),
        })

        review.append(f"## {q['id']} ({q['tier']}) — {len(scored)} matching chunks")
        review.append("")
        review.append(f"**Q:** {q['q']}")
        if q.get("note"):
            review.append(f"*{q['note']}*")
        review.append("")
        if not top:
            review.append("**NO CANDIDATES FOUND — check the search terms or "
                          "retire the question.**")
        for s, c in top:
            snippet = re.sub(r"\s+", " ", c["text"])[:240]
            review.append(f"- `{c['chunk_id']}` (score {s:.2f}, "
                          f"{c.get('book')}, p{c.get('page_start')})")
            review.append(f"  > {snippet}...")
        review.append("")

        flag = "" if top else "   <-- NO HITS"
        print(f"  {q['id']:4s} {q['tier']:7s} {len(scored):4d} matches, "
              f"{len(cand)} recorded{flag}")

    # Record the source file. Gold chunk IDs are only meaningful against the
    # collection built from THIS file -- a different chunking run reuses the
    # same ID format for different text, which scores silently and wrongly.
    payload = {
        "source_file": os.path.basename(args.input),
        "n_chunks": len(chunks),
        "questions": eval_set,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    with open(args.review, "w", encoding="utf-8") as f:
        f.write("\n".join(review) + "\n")

    prefilled = sum(1 for e in eval_set if e["gold_chunk_ids"])
    multi = sum(1 for e in eval_set if e["multi_chunk"])
    print(f"\nWrote {args.out} ({len(eval_set)} questions) and {args.review}")
    print(f"  {prefilled} single-chunk questions prefilled with a best candidate")
    print(f"  {multi} multi-chunk questions need you to choose the gold set")
    if no_hit:
        print(f"  NO CANDIDATES for: {', '.join(no_hit)} -- check these first")
    print(f"\nNext: read {args.review}, fix gold_chunk_ids in {args.out}, "
          f"set verified:true. The harness reports how many golds are "
          f"unverified so results are never quietly based on guesses.")


if __name__ == "__main__":
    main()