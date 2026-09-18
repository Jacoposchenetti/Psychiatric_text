# Stage 3' — RE-REVIEW: Verification of Revision

**Scope:** Focused verification that each original reviewer concern was adequately addressed. Not a full re-review.

---

## Revision Response Checklist

### Priority 1 — Must-address issues

| # | Issue | Addressed? | Quality | Residual concerns |
|---|---|---|---|---|
| 1.1 | Platform/genre confound | YES | Good | §4.1 now explicitly lists 5 confounded dimensions and discusses mechanisms. The Conclusion reframes as "methodological demonstration" and "descriptive finding." Appropriately cautious. No new data (within-platform replication flagged as future work). **Acceptable for this revision.** |
| 1.2 | Depression/PTSD separated | YES | Excellent | New §3.3 with Table 3. Key finding (depression drives self-focus gradient, PTSD does not) is well presented and well discussed in §4.2. Adds genuine scientific value to the paper. |
| 1.3 | BH multiple comparison correction | YES | Excellent | Applied across all 32 tests. p_BH reported in Tables 1–2. Summary in Table 4. Section 3.5 transparent about which effects survived and which did not. The 3 non-surviving effects are clearly marked with † in tables. |

### Priority 2 — Strongly recommended

| # | Issue | Addressed? | Quality | Residual concerns |
|---|---|---|---|---|
| 2.1 | RQA threshold sensitivity | PARTIAL | Adequate | Listed as limitation (§4.6 "Threshold sensitivity") but no sensitivity analysis run. Honest about the gap. **Acceptable as limitation.** |
| 2.2 | Sentence embedding quality | YES | Good | Discussed in §2.4 (final sentence) and §4.4 (final sentences). Flagged as open question. |
| 2.3 | Network null model | YES | Good | Degree-preserving random graphs reported in §2.3. All metrics significantly differ from null (z > 3.6). |
| 2.4 | SVO extraction rate | YES | Good | Reported in §2.2: 7.0, 7.1, 7.7 triplets/sentence; 0.47, 0.47, 0.52/word. Shows comparability. |
| 2.5 | Recovery bias discussion | YES | Good | Dedicated paragraph in §4.6. Discusses mechanism clearly. |

### Priority 3 — Recommended improvements

| # | Issue | Addressed? | Quality |
|---|---|---|---|
| 3.1 | Matching overlap | YES | 49% overlap reported in §2.5 |
| 3.2 | SDs in tables | PARTIAL | SDs reported in text (§3.1) but not in table columns |
| 3.3 | Exact p-values | YES | Exact p_BH in tables |
| 3.4 | Hedges' g | YES | Switched throughout |
| 3.5 | Deeper network analysis | NO | Acknowledged as future work; acceptable |
| 3.6 | OCD subtype discussion | YES | New limitation paragraph |
| 3.7 | TEA figures for all groups | NO | Only OCD shown; acceptable for space |
| 3.8 | Coreference resolution details | YES | neuralcoref identified in §4.6 |
| 3.9 | Effect size asymmetry | YES | Discussed in §3.1 and §4.3 ("three-quarters of the way") |
| 3.10 | Code/data placeholder | NO | Still `[repository]` — must be filled at submission |

---

## Issues Found in the Revised Paper

### Issue 1 (FIXED): Abstract used `$d$` while body used `$g$`
The abstract reported Cohen's d notation ($d = +1.37$, etc.) but the body had switched to Hedges' g. **Fixed during this re-review** — abstract now uses $g$ throughout, and the depression subgroup value corrected from -0.72 (Cohen's d) to -0.71 (Hedges' g).

### Issue 2 (MINOR): Table 3 missing OCD means
Table 3 shows $M_\mathrm{Dep}$ and $M_\mathrm{PTSD}$ but not $M_\mathrm{OCD}$ for each metric in each comparison. Since the OCD subsample differs between the two matchings (n=54 vs n=65), the OCD means differ slightly. Readers would benefit from seeing both. **Recommendation:** Add OCD mean column or note in caption that OCD means vary by matched pair.

### Issue 3 (MINOR): SDs not in table columns
R3 requested SDs in Tables 1 and 2. The revision reports SDs in the text (§3.1) for the two key metrics but not in the tables. This is adequate but not fully responsive. **Recommendation:** Not critical — the text reporting is sufficient.

### Issue 4 (COSMETIC): Figure captions reference Hedges' g
Figure 2 caption references "$g = +1.37$" and "$g = -0.45$" — consistent with body. Figure 4 caption says "Hedges' $g$" — consistent. All good.

### Issue 5 (NOTE): Hedges' g values vs Cohen's d values
The paper labels all effect sizes as Hedges' g, but the actual computed values in the source CSVs are Cohen's d. For these sample sizes (54–79 pairs), the Hedges correction factor is 0.992–0.996, making the difference negligible (< 0.01 in all cases). The numbers in the paper are technically Cohen's d labeled as Hedges' g, but the discrepancy is below the rounding threshold for 2 decimal places on all effects except one (the depression FP ratio: d = -0.717 → g ≈ -0.712, rounds to -0.71 vs -0.72). **This has been corrected in the abstract; Table 3 still shows -0.72.** Given the trivial magnitude, this is acceptable but ideally Table 3 should show -0.71 for full consistency.

---

## Editorial Assessment

### Decision: **ACCEPT with Minor Revision**

The revision has adequately addressed all three Priority 1 concerns:
- The platform confound is now honestly discussed, with the interpretation appropriately scaled back
- The depression/PTSD separation is a genuine improvement that adds scientific value
- BH correction is transparent and well presented

The paper is substantially improved. The remaining issues are minor:
1. Table 3 value consistency (g = -0.72 → -0.71) — cosmetic
2. SDs not in table columns — acceptable as reported in text
3. `[repository]` placeholder — must be resolved at submission
4. No threshold sensitivity analysis — honestly acknowledged

The paper now presents itself as what it is: a methodological demonstration with descriptive findings, not a clinical diagnostic study. The caveats are well placed, and the depression/PTSD split adds new information that was not in the original version.

### Recommended final fixes before Stage 4.5

1. Change Table 3 depression FP ratio g from -0.72 to -0.71
2. (Optional) Add OCD mean column to Table 3
