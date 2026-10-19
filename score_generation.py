#!/usr/bin/env python3
"""
Score generation_results.json (output of generate_answers.py).

Retrieval and generation are scored separately. Three layers, cheapest first:

1. DETERMINISTIC checks (free, no API):
   behaviour   status == expected_behaviour (answer / abstain / clarify)
   citations   every cited ID (field and inline [id]) was actually retrieved;
               a cited ID that was never shown to the model is fabricated
   gold_in_ctx did retrieval put the gold chunk(s) in front of the model?
               (separates "model failed" from "retrieval failed")
   unsupported numbers and capitalised words in the answer that appear in NONE
               of the retrieved passages: a cheap hallucination tripwire
               (noisy: paraphrase and OCR variants cause false alarms)

2. LLM JUDGE (one API call per answered question): splits the answer into
   atomic claims and labels each supported / unsupported by the passages.
   groundedness = supported claims / total claims.
   Caveat: if the judge is the same model as the generator it may favour its
   own phrasing (self-preference). Use --judge-model to pick another model.
   Judge scores are a signal, not ground truth; hence layer 3.

3. MANUAL SPOT-CHECK: writes spot_check.md with a seeded sample of answers
   next to their passages, for a human to verify. Report how many you checked
   and how often you agreed with the judge. That agreement rate is what makes
   the judge number credible.

NOT scored: answer correctness. eval_set.json stores gold chunks, not gold
answer text, so "is the answer right" is only approximated by gold_in_ctx +
groundedness.

USAGE
    python score_generation.py                  # all layers
    python score_generation.py --no-judge       # deterministic only, free
    python score_generation.py --judge-model claude-opus-5-5
    python score_generation.py --spot 12 --seed 1
"""

import re
import sys
import json
import random
import argparse
from collections import defaultdict

from generate_answers import (DEFAULT_MODEL, call_claude, make_client, parse_json)

TIER_ORDER = ["easy", "medium", "hard", "trick"]

JUDGE_SYSTEM = """You are a strict fact-checking judge. You are given PASSAGES and an ANSWER that was written from them.

1. Split the ANSWER into atomic factual claims (ignore citations, hedges and meta-statements like "the passages do not say").
2. For each claim decide whether the PASSAGES, read allowing for OCR spelling errors, directly support it.
   - "supported": stated in or clearly entailed by the passages
   - "unsupported": not stated in the passages (even if true in the real world), or contradicted by them

Respond with a single JSON object and nothing else:
{"claims": [{"claim": "<text>", "verdict": "supported" | "unsupported", "reason": "<short>"}]}"""


# ---------------------------------------------------------------- deterministic

def inline_ids(answer):
    return set(re.findall(r"\[([A-Za-z0-9_]+_c\d{5})\]", answer or ""))


def norm(s):
    return re.sub(r"[^a-z0-9 ]+", " ", s.lower())


def unsupported_entities(answer, texts):
    """Numbers and capitalised mid-sentence words in the answer found in no passage."""
    if not answer:
        return []
    body = re.sub(r"\[[A-Za-z0-9_]+_c\d{5}\]", " ", answer)
    corpus = " " + norm(" ".join(texts)) + " "
    found = set(re.findall(r"\b\d{3,4}\b", body))
    # capitalised words not at sentence start
    for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{3,})\b", body):
        found.add(m.group(1))
    return sorted(w for w in found if f" {norm(w).strip()} " not in corpus)


def gold_in_context(rec):
    gold = set(rec.get("gold_chunk_ids") or [])
    if not gold:
        return None
    got = gold & set(rec["retrieved_ids"])
    return len(got) == len(gold) if rec.get("gold_mode") == "all" else bool(got)


def deterministic(rec):
    ans = rec.get("answer") or ""
    cited = set(rec.get("citations") or []) | inline_ids(ans)
    retrieved = set(rec["retrieved_ids"])
    fabricated = sorted(cited - retrieved)
    answered = rec.get("status") == "answer"
    return {
        "behaviour_ok": rec.get("status") == rec["expected_behaviour"],
        "fabricated_citations": fabricated,
        "uncited_answer": answered and not cited,
        "gold_in_ctx": gold_in_context(rec),
        "unsupported_entities": unsupported_entities(ans, rec["retrieved_texts"].values())
                                if answered else [],
    }


# ---------------------------------------------------------------- judge

def judge(client, model, rec):
    passages = "\n\n".join(f"[{i}]\n{t}" for i, t in rec["retrieved_texts"].items())
    user = f"PASSAGES\n\n{passages}\n\nANSWER\n\n{rec['answer']}"
    text, ti, to = call_claude(client, model, JUDGE_SYSTEM, user, max_tokens=1500)
    parsed = parse_json(text)
    if not parsed or not isinstance(parsed.get("claims"), list) or not parsed["claims"]:
        return None
    claims = [c for c in parsed["claims"] if isinstance(c, dict)]
    n_sup = sum(1 for c in claims if c.get("verdict") == "supported")
    return {"claims": claims, "n_claims": len(claims), "n_supported": n_sup,
            "groundedness": n_sup / len(claims)}


# ---------------------------------------------------------------- reporting

def pct(num, den):
    return f"{num}/{den}" if den else "-"


def report(rows):
    print(f"\n{'tier':<8}{'n':>3}  {'behav':>7} {'cite_ok':>8} {'gold_ctx':>9} "
          f"{'grounded':>9} {'flag_ent':>9}")
    groups = defaultdict(list)
    for r in rows:
        groups[r["tier"]].append(r)
    tiers = [t for t in TIER_ORDER if t in groups] + [t for t in groups if t not in TIER_ORDER]
    for t in tiers + ["ALL"]:
        rs = rows if t == "ALL" else groups[t]
        beh = sum(r["det"]["behaviour_ok"] for r in rs)
        cite_ok = sum(not r["det"]["fabricated_citations"] for r in rs)
        g = [r["det"]["gold_in_ctx"] for r in rs if r["det"]["gold_in_ctx"] is not None]
        gr = [r["judge"]["groundedness"] for r in rs if r.get("judge")]
        ent = sum(bool(r["det"]["unsupported_entities"]) for r in rs if r["status"] == "answer")
        n_ans = sum(r["status"] == "answer" for r in rs)
        gtxt = f"{sum(gr) / len(gr):.2f}" if gr else "-"
        print(f"{t:<8}{len(rs):>3}  {pct(beh, len(rs)):>7} {pct(cite_ok, len(rs)):>8} "
              f"{pct(sum(g), len(g)):>9} {gtxt:>9} {pct(ent, n_ans):>9}")

    # abstention / clarification, split by expected behaviour
    print("\nBehaviour confusion (rows = expected, cols = got):")
    kinds = ["answer", "abstain", "clarify", None]
    print(f"{'':<10}" + "".join(f"{str(k):>9}" for k in kinds))
    for e in ["answer", "abstain", "clarify"]:
        rs = [r for r in rows if r["expected_behaviour"] == e]
        if rs:
            print(f"{e:<10}" + "".join(f"{sum(r['status'] == k for r in rs):>9}" for k in kinds))
    false_abs = [r["id"] for r in rows if r["expected_behaviour"] == "answer" and r["status"] == "abstain"]
    missed = [r["id"] for r in rows if r["expected_behaviour"] == "abstain" and r["status"] != "abstain"]
    print(f"\nFalse abstentions on answerable questions: {false_abs or 'none'}")
    print(f"Failed to abstain (hallucination risk):    {missed or 'none'}")

    # model vs retrieval failure on answerable questions
    ans_rows = [r for r in rows if r["expected_behaviour"] == "answer"
                and r["det"]["gold_in_ctx"] is not None]
    no_ctx = [r for r in ans_rows if not r["det"]["gold_in_ctx"]]
    ctx_but_abstain = [r["id"] for r in ans_rows if r["det"]["gold_in_ctx"] and r["status"] != "answer"]
    print(f"\nAnswerable questions where retrieval missed gold: {len(no_ctx)}/{len(ans_rows)} "
          f"({[r['id'] for r in no_ctx]})")
    print(f"Gold was in context but model did not answer:     {ctx_but_abstain or 'none'}")

    pf = [r["id"] for r in rows if not r.get("parse_ok", True)]
    print(f"Parse failures / loose parses (harness, not model, problems): {pf or 'none'}")

    # behaviour conditional on retrieval: the number that isolates the model
    print("\nOn answerable questions, model behaviour by retrieval outcome:")
    for label, flag in (("gold in context", True), ("gold NOT in context", False)):
        rs = [r for r in rows if r["expected_behaviour"] == "answer"
              and r["det"]["gold_in_ctx"] is flag]
        c = defaultdict(int)
        for r in rs:
            c[str(r["status"])] += 1
        print(f"  {label:<20} n={len(rs):<3} " + "  ".join(f"{k}={v}" for k, v in sorted(c.items())))

    fab = [(r["id"], r["det"]["fabricated_citations"]) for r in rows if r["det"]["fabricated_citations"]]
    print(f"Fabricated citations: {fab or 'none'}")
    bad = [(r["id"], [c for c in r["judge"]["claims"] if c.get("verdict") != "supported"])
           for r in rows if r.get("judge") and r["judge"]["groundedness"] < 1]
    if bad:
        print("\nAnswers with unsupported claims (judge):")
        for i, cs in bad:
            for c in cs:
                print(f"  {i}: {c.get('claim')}  -- {c.get('reason')}")


def write_spot_check(rows, n, seed, path="spot_check.md"):
    pool = [r for r in rows if r["status"] == "answer"]
    random.Random(seed).shuffle(pool)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Manual spot-check ({min(n, len(pool))} answers, seed {seed})\n\n"
                "For each: is every claim in the answer supported by the passages? "
                "Fill in `Human verdict` (all supported / some unsupported) and note "
                "whether you agree with the judge.\n\n")
        for r in pool[:n]:
            j = r.get("judge")
            f.write(f"---\n\n## {r['id']} ({r['tier']})\n\n**Q:** {r['question']}\n\n"
                    f"**Answer:** {r['answer']}\n\n")
            if j:
                f.write(f"**Judge:** {j['n_supported']}/{j['n_claims']} supported\n\n")
                for c in j["claims"]:
                    f.write(f"- [{c.get('verdict')}] {c.get('claim')}\n")
                f.write("\n")
            f.write("**Human verdict:** \n\n**Agree with judge:** \n\n<details><summary>Passages</summary>\n\n")
            for i, t in r["retrieved_texts"].items():
                f.write(f"**{i}**\n\n{t}\n\n")
            f.write("</details>\n\n")
    print(f"Spot-check file -> {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="generation_results.json")
    ap.add_argument("--out", default="generation_scores.json")
    ap.add_argument("--judge-model", default=DEFAULT_MODEL)
    ap.add_argument("--no-judge", action="store_true")
    ap.add_argument("--spot", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(args.results, encoding="utf-8") as f:
        rows = json.load(f)
    if not rows:
        print("No results to score.")
        sys.exit(1)

    prior = {}
    try:
        with open(args.out, encoding="utf-8") as f:
            prior = {r["id"]: r for r in json.load(f)}
    except (OSError, ValueError):
        pass

    client = None if args.no_judge else make_client()
    for r in rows:
        r["det"] = deterministic(r)
        if args.no_judge or r.get("status") != "answer" or not r.get("answer"):
            continue
        old = prior.get(r["id"])
        if old and old.get("judge") and old.get("judge_model") == args.judge_model \
                and old.get("answer") == r["answer"]:
            r["judge"], r["judge_model"] = old["judge"], old["judge_model"]
            continue
        try:
            r["judge"] = judge(client, args.judge_model, r)
            r["judge_model"] = args.judge_model
            if r["judge"] is None:
                print(f"  {r['id']}: judge output unparseable")
        except Exception as e:
            print(f"  {r['id']}: judge API error: {e}")

    if len(rows) <= 5:  # small test run: show per-question detail
        for r in rows:
            print(f"\n--- {r['id']} expected={r['expected_behaviour']} got={r['status']}")
            print("  deterministic:", json.dumps(r["det"]))
            if r.get("judge"):
                for c in r["judge"]["claims"]:
                    print(f"  judge [{c.get('verdict')}] {c.get('claim')}")
    report(rows)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"\nScores -> {args.out}")
    write_spot_check(rows, args.spot, args.seed)
    if client and args.judge_model == DEFAULT_MODEL:
        print("Note: judge model == generator model (self-preference bias possible). "
              "Spot-check spot_check.md by hand.")


if __name__ == "__main__":
    main()