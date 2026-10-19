# Known biases and threats to validity

Where the evaluation could mislead, and what was done about each.

## Measured in the final run

| Bias | What was found |
|---|---|
| **Self-preference / leniency** | Sonnet generates, and Sonnet also judged as a second judge so the gap could be measured. Averaged claim-level scores were close (0.98 self vs 0.97 independent, gap +0.01), but at the answer level the judges disagreed on 7 of 36 answers. In the 10-answer spot-check the independent judge (Opus) matched the reviewer 10/10, the self-judge (Sonnet) 6/10: it missed two real errors (M14, H4) and over-flagged two others. The averaged gap hides this; the per-answer disagreement shows it. |
| **Shared blind spots** | Both judges are Claude models. Mitigation: judges must quote the passage text behind each "supported" verdict, and quotes not in the passages are downgraded to *unverified*. **Limit found:** this checks that the quote *exists*, not that it *supports the claim*. Sonnet's quotes for M14 and H4 were real but did not contain the claim (no mention of Asoka; nothing about finances). Only the independent judge caught those. A judge from another vendor would close the remaining gap. |
| **Non-determinism** | The API/model rejected a temperature setting, so temperature could not be fixed. Behaviour matches ranged 38–40 / 49 across runs, and H9 flipped between answering and abstaining. Results are good to about ±1–2 questions. |

## Documented, not fixed

| Bias | Effect on the numbers |
|---|---|
| **Grounded ≠ correct** | Groundedness checks the answer against the *retrieved* passages only. An answer faithful to an incomplete retrieval still scores well. H5 is fully supported but omits the Class 12 section on the end of the Harappan civilisation, because BM25 never retrieved it. |
| **Survivorship in groundedness** | Computed only on questions the model chose to answer (36 of 49). A model that abstains more looks more grounded by default. |
| **Claim granularity** | Claim-level support (0.97) looks near ceiling, but only 27 / 36 answers (75%) had every claim supported. Claim-level scores hide that a quarter of answers contain at least one overreach. |
| **Question-author bias** | Questions and evidence patterns were written with the target passage in view, so questions reuse its rare words. This favours lexical retrieval and may partly explain BM25 > dense. Untested. |
| **Same family throughout** | The questions were drafted with Claude's help, Claude generates, and Claude judges. Agreement between Claude components is weaker evidence than agreement with an independent human. The spot-check reviewer should be stated; if AI-assisted, its agreement with the judges is not independent evidence. |
| **Label noise** | Some gold labels are imperfect (M8, M14, H6, H13, X5 match many chunks; X4 was once mislabeled). This understates retrieval scores and inflates "false abstention" counts. |
| **Small samples** | Only 4 abstain/clarify questions and 10 spot-checked answers. "Never failed to abstain" is a 4-question claim. |
| **Single run** | One final run, no repeated runs, so no variance estimate beyond the run-to-run differences observed. |
| **Prompt leakage (fixed)** | An early system prompt named the `yarah`/`ijarah` OCR pair, which leaked the X1 answer. Removed before the reported runs. |
| **Position effects** | Passages are given to the model in BM25 rank order. Order effects on the answer were not tested. |