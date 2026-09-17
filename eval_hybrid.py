#!/usr/bin/env python3
"""
Hybrid retrieval: BM25 + dense vectors, fused with Reciprocal Rank Fusion.

WHY
---
The dense-only baseline failed on exactly the queries dense retrieval is known
to handle badly:

  E10  "What does the term sardeshmukhi refer to?"   rare technical term
  E2   "In which month and year did ... march south"  date lookup
  E3   "Who was given the title Maharaja ..."         specific proper nouns

All three are cases where an exact token match is the strongest available
signal and cosine similarity over a 384-dim embedding washes it out. BM25
scores them on term rarity instead, which is the complementary failure mode.

An unexpected baseline result shapes this too. The OCR-defect questions X1 and
X2 PASSED under dense retrieval (ranks 1 and 2), contradicting the prediction
that they would fail. The reason: the defect (`ijarah` extracted as `yarah`,
`Zulfiqar` also appearing as `Zulfigar`) breaks LEXICAL matching, but the
chunk is still semantically about revenue-farming and peasant oppression, so
the embedding finds it anyway.

That makes X1/X2 a test of whether adding BM25 *degrades* them: BM25 cannot
match a token the corpus spells differently, so if the fusion over-weights the
lexical side those two should fall. Watch them.

FUSION
------
Reciprocal Rank Fusion rather than score normalisation:

    score(d) = sum over retrievers of  1 / (k_rrf + rank(d))

BM25 scores and cosine similarities are not on a comparable scale, and
normalising them requires choosing a scheme that itself becomes a tuning
parameter. RRF uses only ranks, has one constant, and is hard to get subtly
wrong.

USAGE
-----
    pip install rank-bm25
    python eval_hybrid.py                      # compare dense / bm25 / hybrid
    python eval_hybrid.py --alpha 0.7          # weight dense higher
    python eval_hybrid.py --failures
"""

import os
import re
import json
import argparse
from collections import defaultdict

DEFAULT_DB = "chroma_db"
DEFAULT_COLLECTION = "chunks__bge_small_en_v1_5"
TIER_ORDER = ["easy", "medium", "hard", "trick"]
K_RRF = 60          # standard RRF constant; results are insensitive to it
CANDIDATE_DEPTH = 50  # how deep each retriever goes before fusion


def tokenize(text):
    """Lowercase word tokens. Deliberately simple: no stemming, so an OCR
    variant like 'Zulfigar' stays a distinct token from 'Zulfiqar' and the
    defect's effect on lexical matching remains visible rather than papered
    over by aggressive normalisation."""
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf_fuse(rankings, weights=None, k=K_RRF):
    """
    rankings: list of ranked id-lists, best first.
    weights:  optional per-retriever weight.
    """
    weights = weights or [1.0] * len(rankings)
    scores = defaultdict(float)
    for ranked, w in zip(rankings, weights):
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] += w / (k + rank)
    return [cid for cid, _ in sorted(scores.items(), key=lambda x: -x[1])]


def get_query_encoder(model_name, prefix, backend="auto"):
    def wrap(fn):
        return lambda texts: fn([prefix + t for t in texts])

    if backend in ("auto", "sentence-transformers"):
        try:
            from sentence_transformers import SentenceTransformer
            m = SentenceTransformer(model_name)
            return wrap(lambda ts: m.encode(ts, normalize_embeddings=True,
                                            convert_to_numpy=True,
                                            show_progress_bar=False).tolist())
        except ImportError:
            if backend == "sentence-transformers":
                raise
            print("sentence-transformers unavailable, using fastembed")
    from fastembed import TextEmbedding
    m = TextEmbedding(model_name=model_name)
    return wrap(lambda ts: [v.tolist() for v in m.embed(ts)])


def score_one(retrieved, gold, mode, k):
    top = retrieved[:k]
    hits = [i for i, c in enumerate(top) if c in gold]
    first = hits[0] + 1 if hits else None
    if mode == "all":
        full = 1.0 if gold.issubset(set(top)) else 0.0
    else:
        full = 1.0 if hits else 0.0
    return {
        "recall_at_k": 1.0 if hits else 0.0,
        "full_at_k": full,
        "mrr": 1.0 / first if first else 0.0,
        "first_rank": first,
        "retrieved": top,
    }


def summarise(all_results, k):
    methods = list(all_results.keys())
    print(f"\n{'='*78}")
    print(f"RETRIEVAL COMPARISON @ k={k}")
    print("=" * 78)
    header = f"{'tier':8s} {'n':>3s}"
    for m in methods:
        header += f" | {m:>18s}"
    print(header)
    print(f"{'':8s} {'':>3s}" + "".join(
        f" | {'rec  full  mrr':>18s}" for _ in methods))
    print("-" * 78)

    tiers = TIER_ORDER + ["ALL"]
    for tier in tiers:
        rows = {}
        for m in methods:
            rs = [r for r in all_results[m]
                  if tier == "ALL" or r["tier"] == tier]
            rows[m] = rs
        n = len(rows[methods[0]])
        if not n:
            continue
        line = f"{tier:8s} {n:3d}"
        for m in methods:
            rs = rows[m]
            line += (f" | {sum(x['recall_at_k'] for x in rs)/n:4.2f} "
                     f"{sum(x['full_at_k'] for x in rs)/n:5.2f} "
                     f"{sum(x['mrr'] for x in rs)/n:5.2f}")
        print(line)

    # the questions that motivated hybrid search
    print("\nQuestions that failed under dense retrieval:")
    dense = {r["id"]: r for r in all_results.get("dense", [])}
    for qid in ("E2", "E3", "E10"):
        if qid not in dense:
            continue
        cells = []
        for m in methods:
            r = next((x for x in all_results[m] if x["id"] == qid), None)
            cells.append(f"{m}={r['first_rank'] if r and r['first_rank'] else 'miss'}")
        print(f"  {qid}: " + ", ".join(cells))

    print("\nOCR-defect questions (does BM25 drag them down?):")
    for qid in ("X1", "X2"):
        cells = []
        for m in methods:
            r = next((x for x in all_results[m] if x["id"] == qid), None)
            if r:
                cells.append(f"{m}={r['first_rank'] if r['first_rank'] else 'miss'}")
        if cells:
            print(f"  {qid}: " + ", ".join(cells))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-set", default="eval_set.json")
    ap.add_argument("--chunks", default="chunks.jsonl")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="weight on the dense retriever (0=BM25 only, 1=dense only)")
    ap.add_argument("--k-rrf", type=int, default=K_RRF,
                    help="RRF constant. Lower values let a single strong rank-1 "
                         "hit dominate; the default 60 rewards documents both "
                         "retrievers rank moderately well, which is right when "
                         "the retrievers correlate but can bury a rank-1 "
                         "lexical hit when they do not. Try 10-20 if BM25 finds "
                         "an answer that hybrid then loses.")
    ap.add_argument("--backend", default="auto")
    ap.add_argument("--depth-curve", action="store_true",
                    help="report recall at several depths. This decides "
                         "whether a reranker is worth building: a reranker can "
                         "only reorder what retrieval already returned, so the "
                         "gap between recall@5 and recall@50 is its ceiling.")
    ap.add_argument("--sweep", action="store_true",
                    help="grid-search alpha and k_rrf instead of scoring one "
                         "config, and print the curve")
    ap.add_argument("--failures", action="store_true")
    ap.add_argument("--out", default="eval_hybrid_results.json")
    args = ap.parse_args()

    for f in (args.eval_set, args.chunks):
        if not os.path.exists(f):
            print(f"ERROR: {f} not found.")
            return

    raw = json.load(open(args.eval_set, encoding="utf-8"))
    if isinstance(raw, dict):
        eval_set, eval_source = raw.get("questions", []), raw.get("source_file", "")
    else:
        eval_set, eval_source = raw, ""

    chunks = [json.loads(l) for l in open(args.chunks, encoding="utf-8") if l.strip()]
    if eval_source and os.path.basename(args.chunks) != eval_source:
        print(f"ERROR: eval set was built from '{eval_source}' but --chunks is "
              f"'{os.path.basename(args.chunks)}'. Chunk IDs are not comparable "
              f"across chunking runs.")
        return

    import chromadb
    client = chromadb.PersistentClient(path=args.db)
    try:
        coll = client.get_collection(args.collection)
    except Exception:
        print(f"ERROR: collection '{args.collection}' not found. Available: "
              f"{[c.name for c in client.list_collections()]}")
        return

    meta = coll.metadata or {}
    coll_source = meta.get("source_file", "")
    if eval_source and coll_source and os.path.basename(coll_source) != eval_source:
        print(f"ERROR: eval set from '{eval_source}', collection from "
              f"'{coll_source}'. Not comparable.")
        return

    model = meta.get("model", "BAAI/bge-small-en-v1.5")
    prefix = meta.get("query_prefix", "")
    print(f"Collection: {args.collection} ({coll.count()} chunks)")

    # ---- BM25 index ----
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("ERROR: pip install rank-bm25")
        return
    print(f"Building BM25 index over {len(chunks)} chunks...")
    ids = [c["chunk_id"] for c in chunks]
    bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])

    questions = [e for e in eval_set if e.get("gold_chunk_ids")]
    print(f"Scoring {len(questions)} questions\n")

    encode = get_query_encoder(model, prefix, args.backend)
    qvecs = encode([e["question"] for e in questions])

    results = {"dense": [], "bm25": [], "hybrid": []}
    cached = []   # (question, qvec, dense_ranking, bm25_ranking) for the sweep

    for e, v in zip(questions, qvecs):
        gold = set(e["gold_chunk_ids"])
        mode = e.get("gold_mode", "all" if e.get("multi_chunk") else "any")

        dense_ranked = coll.query(query_embeddings=[v],
                                  n_results=CANDIDATE_DEPTH,
                                  include=[])["ids"][0]

        scores = bm25.get_scores(tokenize(e["question"]))
        bm25_ranked = [ids[i] for i in
                       sorted(range(len(ids)), key=lambda i: -scores[i])
                       [:CANDIDATE_DEPTH]]

        fused = rrf_fuse([dense_ranked, bm25_ranked],
                         weights=[args.alpha, 1.0 - args.alpha],
                         k=args.k_rrf)

        cached.append((e, v, dense_ranked, bm25_ranked))

        for name, ranked in (("dense", dense_ranked),
                             ("bm25", bm25_ranked),
                             ("hybrid", fused)):
            row = {"id": e["id"], "tier": e["tier"],
                   "question": e["question"],
                   "verified": e.get("verified", False),
                   "gold_mode": mode,
                   "gold_chunk_ids": sorted(gold),
                   "note": e.get("note", "")}
            row.update(score_one(ranked, gold, mode, args.k))
            results[name].append(row)

    if args.depth_curve:
        print(f"\n{'='*72}")
        print("RECALL AT DEPTH  (reranker headroom)")
        print("=" * 72)
        depths = [5, 10, 20, 50]
        print(f"{'method':8s} " + " ".join(f"{'@'+str(dd):>7s}" for dd in depths)
              + f" {'headroom':>10s}")
        print("-" * 72)
        for name in ("dense", "bm25"):
            recs = []
            for dd in depths:
                hit = 0
                for e, v, dr, br in cached:
                    ranked = dr if name == "dense" else br
                    gold = set(e["gold_chunk_ids"])
                    if gold & set(ranked[:dd]):
                        hit += 1
                recs.append(hit / len(cached))
            print(f"{name:8s} " + " ".join(f"{r:7.2f}" for r in recs)
                  + f" {recs[-1]-recs[0]:10.2f}")
        # Union coverage. The sweep showed fusion does not help RANKING at
        # k=5, but that is a different question from COVERAGE at k=50: if the
        # two retrievers miss different questions, feeding the union into a
        # reranker beats reranking either alone.
        for dd in (50,):
            u = 0
            for e, v, dr, br in cached:
                gold = set(e["gold_chunk_ids"])
                if gold & (set(dr[:dd]) | set(br[:dd])):
                    u += 1
            print(f"{'union':8s} " + " ".join(
                f"{'':>7s}" for _ in depths[:-1])
                + f"{u/len(cached):7.2f}"
                + f" {'(ceiling)':>10s}")

        # Which questions have no gold anywhere in either top-50? A reranker
        # cannot reach these; they are chunking or query-rewriting problems.
        dead = [e["id"] for e, v, dr, br in cached
                if not set(e["gold_chunk_ids"]) & (set(dr[:50]) | set(br[:50]))]

        print("-" * 72)
        print("headroom = recall@50 - recall@5. That is the most a perfect")
        print("reranker over the top 50 could add. If it is near zero, the")
        print("problem is retrieval coverage, not ranking, and a reranker")
        print("cannot fix it.")
        if dead:
            print(f"\nNO GOLD IN EITHER TOP-50 ({len(dead)} questions): "
                  f"{', '.join(dead)}")
            print("A reranker cannot help these. They are coverage failures --")
            print("wrong gold, bad chunk boundaries, or vocabulary the query")
            print("and corpus do not share.")
        return

    if args.sweep:
        # The first run showed hybrid at alpha=0.5, k_rrf=60 gaining ZERO
        # questions over BM25 alone while losing five -- pure dilution. The
        # right fusion weight is an empirical question about this corpus, not
        # a default to inherit, so sweep it and show the curve.
        print(f"\n{'='*72}")
        print("FUSION SWEEP (recall@k / full@k / MRR over all questions)")
        print("=" * 72)
        print(f"{'alpha':>6s} {'k_rrf':>6s} | {'recall':>7s} {'full':>7s} {'MRR':>7s}")
        print("-" * 72)
        best = None
        for a in (0.0, 0.2, 0.3, 0.5, 0.7, 0.8, 1.0):
            for krrf in (5, 10, 20, 60):
                rows = []
                for e, v, dr, br in cached:
                    gold = set(e["gold_chunk_ids"])
                    mode = e.get("gold_mode",
                                 "all" if e.get("multi_chunk") else "any")
                    fused = rrf_fuse([dr, br], weights=[a, 1.0 - a], k=krrf)
                    rows.append(score_one(fused, gold, mode, args.k))
                n = len(rows)
                rec = sum(r["recall_at_k"] for r in rows) / n
                ful = sum(r["full_at_k"] for r in rows) / n
                mrr = sum(r["mrr"] for r in rows) / n
                print(f"{a:6.1f} {krrf:6d} | {rec:7.2f} {ful:7.2f} {mrr:7.2f}")
                if best is None or rec > best[0]:
                    best = (rec, ful, mrr, a, krrf)
        print("-" * 72)
        print(f"best recall@{args.k}: {best[0]:.2f} at alpha={best[3]}, "
              f"k_rrf={best[4]} (full@k {best[1]:.2f}, MRR {best[2]:.2f})")
        print("\nalpha=1.0 is dense only, alpha=0.0 is BM25 only. If neither "
              "interior value beats both ends, fusion is not helping on this "
              "corpus and the honest reporting is to say so.")
        return

    summarise(results, args.k)

    if args.failures:
        print(f"\n{'='*78}\nHYBRID FAILURES\n{'='*78}")
        for r in results["hybrid"]:
            if r["recall_at_k"] == 0.0:
                d = next(x for x in results["dense"] if x["id"] == r["id"])
                b = next(x for x in results["bm25"] if x["id"] == r["id"])
                print(f"\n[{r['id']}] {r['tier']}  "
                      f"(dense={d['first_rank'] or 'miss'}, "
                      f"bm25={b['first_rank'] or 'miss'})")
                print(f"  Q: {r['question'][:84]}")
                print(f"  gold: {', '.join(r['gold_chunk_ids'][:3])}")

    json.dump({"collection": args.collection, "k": args.k,
               "alpha": args.alpha, "results": results},
              open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nDetail -> {args.out}")


if __name__ == "__main__":
    main()