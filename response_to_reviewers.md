# Response to Reviewers

## Reviewer 1 (Computational Linguistics / NLP)

**R1.1: spaCy model choice (en_core_web_lg vs transformer)**
We acknowledge this limitation and now discuss it explicitly in Section 4.6 (Limitations, "NLP pipeline" paragraph). We report SVO extraction rates across groups (0.47, 0.47, 0.52 triplets per word for OCD, cancer, depression/PTSD respectively), showing that extraction rates are comparable and any parsing noise is distributed rather than systematically biased. A full rerun with `en_core_web_trf` remains as future work.

**R1.2: Sentence embedding quality**
We now discuss this in both the RQA methods section (Section 2.4, final sentence) and the Discussion (Section 4.4, final sentences). We acknowledge that spaCy's 300d average word vectors are coarser than dedicated sentence transformers and flag this as an open question.

**R1.3: Cosine threshold sensitivity**
We now mention this explicitly as a limitation (Section 4.6, "Threshold sensitivity" paragraph). We did not conduct a formal multi-threshold analysis; the robustness of the gradient to threshold choice is flagged for future work.

**R1.4: Coreference resolution**
We now specify the module used (spaCy's neuralcoref) in Section 4.6.

**R1.5: Data availability placeholder**
We acknowledge this. The `[repository]` placeholder will be replaced with the actual URL at submission.

**R1.6 (minor): "Propensity-score-like matching" framing**
Changed to "nearest-neighbor matching on word count" (Section 2.5).

---

## Reviewer 2 (Clinical Psychology / OCD)

**R2.1: Conflation of depression and PTSD (PRIORITY 1)**
We conducted a full supplementary analysis separating the two subreddits. Results are reported in the new Section 3.3 ("Supplementary Analysis: Depression vs. PTSD Separated") with Table 3. Key finding: the self-focus gradient is driven primarily by depression (g = −0.72, p < .001) and does not reach significance for PTSD (g = −0.32, p = .11). The RQA gradient generalizes more broadly, though effects are larger for depression. We discuss the differential pattern in the new Section 4.2.

**R2.2: Recovery bias**
This is now a dedicated paragraph in Limitations (Section 4.6, "Recovery bias") rather than a brief mention. We discuss how recovery stage may partially explain the gradient, independent of disorder type.

**R2.3: Missing clinical variables**
Now addressed more explicitly in the "Unverified diagnoses" limitation paragraph.

**R2.4: OCD subtypes**
Added a dedicated limitation paragraph (Section 4.6, "OCD subtype heterogeneity").

**R2.5: "Ego-dystonic" interpretation is speculative**
We now label the centrality-profile interpretation as "interpretive rather than confirmatory" (Section 4.3) and note it is based on small representative samples.

**R2.6 (minor): DSM-5 reclassification detail**
Corrected to specify the "Obsessive-Compulsive and Related Disorders" chapter (Section 4.5).

---

## Reviewer 3 (Methodology / Statistics)

**R3.1: Multiple comparison correction (PRIORITY 1)**
We applied Benjamini-Hochberg FDR correction across all 32 TEA + RQA tests. Results: 16 of 19 originally significant effects survive at FDR = .05. The three that do not are all small effects at the margins (TEA triplets/word, graph density, Lmax). All effects with |g| > 0.5 survive. BH-corrected p-values are now reported in Tables 1 and 2, with a summary in Table 4 and Section 3.5.

**R3.2: Matching overlap**
Now reported in Section 2.5: "Of the 65 OCD texts matched with cancer stories and the 79 matched with depression/PTSD, 32 appeared in both comparisons (49% overlap)."

**R3.3: SDs in tables**
SDs are now reported for key metrics in the Results text (Section 3.1).

**R3.4: Hedges' g**
Changed from Cohen's d to Hedges' g throughout, as recommended for the moderate sample sizes.

**R3.5: Exact p-values**
Exact p-values are now reported in the BH-corrected results file; tables show exact values where possible and use < .001 only when p rounds to 0.000.

**R3.6: Effect size asymmetry**
We now discuss the asymmetry explicitly in Section 3.1 ("The gradient"): "The effect sizes are asymmetric: OCD is much farther from cancer (g = +1.37) than from depression/PTSD (g = −0.45)," and in Section 4.3: "On a linear scale from cancer to depression, OCD falls about three-quarters of the way toward the depression end."

---

## Reviewer 4 (Network Science)

**R4.1: Shallow network analysis**
We acknowledge this concern. The present paper uses TEA networks primarily as a feature-extraction framework, and we agree that deeper network analysis (community detection, motif analysis, sentiment flow) would be a valuable extension. We have framed the current contribution more clearly as a first application rather than an exhaustive network analysis.

**R4.2: Centrality analysis on representative samples**
Now clearly labeled as "interpretive rather than confirmatory" and "based on small representative samples" (Section 3.4 and Discussion 4.3).

**R4.3: Null model for network metrics**
We computed degree-preserving random graph comparisons for density, clustering, and LSCC across all three groups (1,000 randomizations per network). All three metrics differ significantly from the null model (z > 3.6, p < .001). This is now reported in Section 2.3.

**R4.4: Tripartite structure unexploited**
We acknowledge this as a limitation of using standard graph metrics on tripartite structures. This is an important direction for future work.

**R4.5: Edge weight distribution**
Not included in this revision; flagged for future work.

**R4.6 (minor): Node/edge counts**
SVO extraction rates per group are now reported in Section 2.2.

---

## Reviewer 5 (Devil's Advocate)

**R5.1: Confounded comparisons (PRIORITY 1)**
We substantially revised the Discussion to address this. The new Section 4.1 ("The Gradient and Its Limits") now explicitly lists the five confounded dimensions (disorder, platform, genre, curation, narrative stance), discusses specific mechanisms by which each could produce the observed pattern, and reframes the clinical interpretation as one of several possibilities. The Conclusion now characterizes the study as a "methodological demonstration" and a "descriptive finding that motivates within-platform replication."

**R5.2: Cherry-picked gradient**
We now report the BH correction results transparently (Tables 1-2, Table 4) and acknowledge that only the subset of metrics with medium-to-large effects in both directions define the gradient.

**R5.3: RQA on sentence embeddings is non-standard**
We added a note in Section 2.4 clarifying that we use RQA terminology as "convenient labels for specific recurrence-matrix properties, without implying that the underlying sentence sequences are dynamical systems in the technical sense."

**R5.4: No healthy baseline**
Added as a limitation (Section 4.6, "No healthy baseline").

**R5.5: "Intermediate position" is not a finding / effect size asymmetry**
We now discuss the asymmetry explicitly (Sections 3.1, 4.3) and avoid implying equidistance. The Conclusion no longer frames "intermediate position" as the central claim but instead focuses on the convergence of two independent methods.

**R5.6: SVO extraction rate**
Now reported in Section 2.2: 7.0, 7.1, and 7.7 triplets per sentence (0.47, 0.47, 0.52 per word) for OCD, cancer, depression/PTSD respectively.

---

## Summary of Changes

| Revision item | Priority | Status |
|---|---|---|
| Platform/genre confound discussion | P1 | Addressed (§4.1 rewritten) |
| Depression/PTSD separated | P1 | Addressed (new §3.3, Table 3, §4.2) |
| BH multiple comparison correction | P1 | Addressed (Tables 1-2, Table 4, §3.5) |
| SVO extraction rate | P2 | Addressed (§2.2) |
| Recovery bias discussion | P2 | Addressed (§4.6 dedicated paragraph) |
| Network null model | P2 | Addressed (§2.3) |
| RQA terminology caveat | P2 | Addressed (§2.4) |
| Matching overlap | P3 | Addressed (§2.5) |
| Hedges' g | P3 | Addressed (all tables, all text) |
| Effect size asymmetry | P3 | Addressed (§3.1, §4.3) |
| OCD subtype discussion | P3 | Addressed (§4.6) |
| Centrality caveat | P3 | Addressed (§3.4, §4.3) |
| Coreference module identified | P3 | Addressed (§4.6) |
| DSM-5 chapter name | P3 | Addressed (§4.5) |
| Code/data placeholder | P3 | Acknowledged |
