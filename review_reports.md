# Stage 3 — REVIEW: Simulated Peer Review

**Paper:** "Cognitive-Linguistic Network Structure of OCD Personal Narratives: A Three-Group Comparison Using TEA Networks and Recurrence Quantification Analysis"

**Target venue:** Computational Linguistics / Digital Health journal (e.g., *Journal of Biomedical Informatics*, *Computers in Human Behavior*, *Digital Health*)

---

## Reviewer 1 — Computational Linguistics / NLP

**Expertise:** Natural Language Processing, text mining, dependency parsing

**Overall assessment:** The paper applies two computational methods (TEA networks and RQA) to illness narratives and reports a consistent gradient. The methods are well described and the results are clearly presented. However, several technical concerns weaken the contribution.

### Strengths
1. The TEA framework is applied to a novel domain (mental health narratives), extending its validated use in political and educational text.
2. The combination of network-based (TEA) and dynamical-systems-based (RQA) methods is methodologically creative and produces converging evidence.
3. The length-matching procedure is well designed and addresses the most obvious confound (text length).
4. Figures are informative and the forest plot (Figure 4) effectively summarizes the effect structure.

### Weaknesses
1. **spaCy model choice.** The authors acknowledge using `en_core_web_lg` rather than a transformer-based model. For dependency parsing accuracy on informal text (Reddit posts, blog entries), transformer models outperform statistical models by 2–4% UAS. The authors should either rerun with `en_core_web_trf` or provide a robustness check showing that parsing errors do not systematically differ across groups.
2. **Sentence embedding quality.** RQA uses spaCy's average word vectors (300d) as sentence embeddings. These are known to be inferior to dedicated sentence transformers (e.g., all-MiniLM-L6-v2). The choice should be justified or tested for sensitivity.
3. **Cosine threshold calibration.** The threshold (0.923) was calibrated on a pooled sample to target ~10% RR. This is reasonable, but the paper does not report how sensitive the results are to threshold choice. A sensitivity analysis (e.g., thresholds from 0.90 to 0.95) would strengthen the RQA findings.
4. **Coreference resolution.** The paper mentions coreference resolution in passing (Section 2.2) but does not specify which module was used or its accuracy on informal text. Coreference errors could inflate or deflate first-person agent ratios.
5. **Data availability.** The GitHub URL is a placeholder (`[repository]`). For reproducibility, the code and processed data should be available at submission.

### Minor issues
- Section 2.4: "propensity-score-like matching" is an unusual framing for nearest-neighbor matching on a single covariate. Consider simply calling it nearest-neighbor length matching.
- The paper could cite prior applications of RQA to text (e.g., Lichtenstein et al., or Angus et al.) beyond the dynamical systems origin papers.

**Recommendation:** Major revision. The results are interesting but the robustness of the NLP pipeline needs verification.

**Confidence:** 4/5

---

## Reviewer 2 — Clinical Psychology / OCD

**Expertise:** OCD phenomenology, cognitive models of OCD, clinical assessment

**Overall assessment:** The clinical framing is generally appropriate, and the finding that OCD narratives show intermediate self-focus and recurrence fits well with existing cognitive models. However, the paper makes some clinical claims that are either underdeveloped or insufficiently cautious.

### Strengths
1. The three-group design is clinically motivated and the choice of cancer (physical illness) and depression/PTSD (affective disorder) as comparison groups is well justified.
2. The interpretation of the self-focus gradient in terms of ego-dystonic vs. ego-syntonic symptom experience (Section 4.2) is clinically interesting and goes beyond surface-level description.
3. The paper avoids overclaiming diagnostic utility, which is appropriate for unverified clinical samples.

### Weaknesses
1. **Conflation of depression and PTSD.** The third group pools depression and PTSD posts. These are clinically distinct conditions with different cognitive profiles. PTSD involves intrusive memories (more similar to OCD) while depression involves ruminative self-focus. Pooling them obscures potentially important differences. At minimum, a supplementary analysis separating depression from PTSD should be reported, or the rationale for pooling should be stronger.
2. **Recovery bias.** The OCD corpus comes from a recovery-focused blog, meaning narrators are likely in remission or actively recovering. The depression/PTSD posts come from active support-seeking on Reddit. This confound (recovery vs. active distress) could partially explain the gradient: less recurrence and self-focus in OCD might reflect recovery stage, not disorder type. This is mentioned in the limitations but deserves more prominence.
3. **Missing clinical variables.** No information is available about symptom severity, comorbidities, medication status, or illness duration. These are standard limitations in computational clinical work, but the paper should be more explicit about what the gradient might actually be measuring (disorder type? distress level? platform norms?).
4. **OCD subtypes.** OCD is highly heterogeneous (contamination, checking, symmetry, intrusive thoughts, etc.). Different subtypes might produce different narrative patterns. The paper should acknowledge this and discuss whether subtype variation might affect the results.
5. **The "ego-dystonic" interpretation (Section 4.2)** is speculative. The observation that "I" shares agentive space with "it" in OCD narratives is interesting, but the paper does not test this directly (e.g., by coding what "it" refers to). This should be clearly labeled as interpretive, not empirical.

### Minor issues
- The DSM-5 reference (Section 4.4) is appropriate but the paper should note that OCD was reclassified as part of an "Obsessive-Compulsive and Related Disorders" chapter, not simply removed from anxiety disorders.
- "Intrusion-driven (rather than mood-driven)" in the abstract is a useful distinction but could be briefly defined for readers outside clinical psychology.

**Recommendation:** Major revision. Address the depression/PTSD pooling concern and strengthen the clinical discussion.

**Confidence:** 4/5

---

## Reviewer 3 — Methodology / Statistics

**Expertise:** Quantitative methods, effect size estimation, matching designs

**Overall assessment:** The statistical approach is straightforward and generally appropriate for an exploratory study. The focus on effect sizes over p-values is commendable. Several methodological issues should be addressed.

### Strengths
1. Effect sizes (Cohen's d) are reported throughout and are the focus of interpretation, which is appropriate for exploratory work with modest sample sizes.
2. The length-matching procedure is well described and the quality checks (Cohen's d < 0.15 on word count) are reassuring.
3. The consistent directionality across 8 metrics is correctly identified as strong evidence against a null pattern.
4. The paper does not overclaim statistical significance given the lack of multiple comparison correction.

### Weaknesses
1. **Multiple comparisons.** The paper reports 9 TEA metrics × 2 comparisons + 7 RQA metrics × 2 comparisons = 32 tests. The justification for not correcting ("exploratory" + "triangulation") is weak. At minimum, the paper should report how many tests survive a Benjamini-Hochberg correction at FDR = 0.05. My estimate is that the large effects (d > 0.8) would survive, but the small-to-medium effects (density, TTR, some RQA metrics in the OCD vs. Dep/PTSD comparison) might not.
2. **Matching efficiency.** The matching produced 65 pairs (OCD vs. Cancer) and 79 pairs (OCD vs. Dep/PTSD) from starting pools of 152 and 152/200. This is 43% and 52% matching efficiency. How many OCD texts were matchable in both comparisons? If different subsets of OCD texts are used in the two comparisons, the gradient claim (Cancer < OCD < Dep/PTSD) rests on partially overlapping OCD samples, which weakens the inference.
3. **Tolerance band.** The 30–35% tolerance for length matching is wide. For a 1,400-word OCD text matched to a cancer text, this allows a control text of 910–1,820 words. Residual length effects within this band could contribute to the observed differences, especially for RQA metrics (longer texts mechanically have more sentence pairs). A regression-based approach (e.g., ANCOVA with word count as covariate) on the matched pairs would address this.
4. **Independence assumption.** Mann-Whitney U assumes independent observations. If some OCD stories come from the same author (the blog may feature repeat contributors), this is violated. Was author identity checked?
5. **Cohen's d with pooled SD.** Cohen's d assumes equal variances, which may not hold across groups with very different text characteristics. Consider reporting Hedges' g (bias-corrected) or Glass's delta.

### Minor issues
- Report exact p-values rather than p < .001 or p < .05 where possible.
- The matching seed (42) should be accompanied by a statement about whether results are robust to other seeds.
- Table 1: report also the SD for each group's means to allow readers to judge distributional overlap.

**Recommendation:** Minor revision. The core findings are robust (the large effects will survive any correction), but the statistical presentation should be tightened.

**Confidence:** 4/5

---

## Reviewer 4 — Network Science

**Expertise:** Graph theory, cognitive networks, complex networks analysis

**Overall assessment:** The TEA network analysis is adequately described but underexploits the network structure. The paper essentially reduces the graphs to a small set of summary statistics, missing opportunities to use the network topology more deeply.

### Strengths
1. The use of TEA networks preserves syntactic structure (who did what to whom) rather than reducing text to bag-of-words features. This is a meaningful advance over frequency-based approaches.
2. The first-person agent ratio metric is well defined and theoretically motivated.
3. The centrality analysis (Section 3.3) begins to explore the qualitative network structure, which is more informative than summary statistics alone.

### Weaknesses
1. **Shallow network analysis.** Nine summary metrics are computed, but the rich graph structure is largely ignored. The paper could analyze:
   - Community structure: Do OCD narratives have different modular organization than the other groups?
   - Motif analysis: What are the most common agent-event-target triplets in each group?
   - Sentiment flow: Do negative-valence agents connect to different event types across groups?
   These would make the network analysis genuinely network-analytic rather than metric-extractive.
2. **Centrality analysis on representative samples.** Section 3.3 mentions "representative samples" for the centrality analysis but does not specify how many texts or how they were selected. If this is 3 texts per group (as the previous work seems to suggest), the centrality rankings are too noisy to interpret. Aggregate centrality across all texts in each group, or report confidence intervals.
3. **No null model for network metrics.** TEA network metrics (density, clustering, LSCC) are compared across groups without a null model. A random graph with the same degree sequence would establish whether the observed values are structurally meaningful or simply reflect differences in text length/vocabulary. The length matching helps but does not fully substitute for a network-level null model.
4. **Bipartite/tripartite structure unexploited.** TEA graphs have three node types (Agent, Event, Target). This tripartite structure implies specific constraints on connectivity that standard metrics (density, clustering) do not capture well. Metrics designed for bipartite or k-partite networks would be more appropriate.
5. **Edge weights.** The paper reports edge repetition index and mean edge weight but does not analyze the weight distribution. Heavy-tailed weight distributions are common in cognitive networks and could differ across groups.

### Minor issues
- Report the number of nodes and edges per group (mean ± SD) as basic network descriptors.
- Figure 1 (TEA network) is visually informative but only shown for OCD. Including a comparable figure for each group would allow visual comparison.
- The GraphML files should be made available for reanalysis.

**Recommendation:** Major revision. The network analysis needs deepening; the paper currently treats networks as feature extractors rather than analyzing them as networks.

**Confidence:** 3/5

---

## Reviewer 5 — Devil's Advocate

**Expertise:** Critical methodological evaluation

**Overall assessment:** The paper's central claim — that OCD sits "between" cancer and depression/PTSD on narrative measures — is internally consistent but rests on a fragile empirical foundation. Several structural problems limit what can actually be concluded.

### The core problem: confounded comparisons

The three corpora differ in at least five dimensions simultaneously:

| | OCD | Cancer | Dep/PTSD |
|---|---|---|---|
| **Disorder type** | Mental (OC) | Physical | Mental (affective) |
| **Platform** | Blog | Interview site | Reddit |
| **Genre** | Unsolicited essay | Edited interview transcript | Social media post |
| **Curation** | Curated by site | Professionally edited | Unedited |
| **Narrative stance** | Recovery focus | Diagnosis/treatment | Active distress |

Length matching addresses one confound (word count) but leaves the other four untouched. The observed gradient could equally reflect:
- **Platform norms:** Reddit posts are more self-focused than blog posts regardless of clinical content. This is well documented in the CMC literature.
- **Editorial intervention:** Cancer interview transcripts are edited by professionals, which could reduce repetitiveness and self-focus.
- **Recovery vs. active distress:** Recovered OCD narrators vs. actively distressed Reddit posters.

The paper cannot distinguish the "disorder type" explanation from these alternatives, which makes the clinical interpretation (Section 4) speculative at best.

### Additional concerns

1. **Cherry-picked gradient.** Only 8 of 16+ metrics show the bidirectional gradient. The paper acknowledges this but frames it as "convergence." An alternative reading: the metrics that happen to be significant are the ones that define the gradient, and the rest are noise. A formal test (e.g., a multivariate approach or an omnibus test) would be more convincing than counting metrics post hoc.
2. **RQA on sentence embeddings is not standard.** RQA was developed for continuous dynamical systems, and its application to discrete sentence-level embeddings with cosine similarity thresholding is non-standard. The meaning of "determinism" and "laminarity" in this context is not the same as in the time-series literature. The paper should be more cautious about interpreting these metrics with the dynamical-systems vocabulary.
3. **No baseline.** What does the gradient look like for non-illness personal narratives (e.g., travel blogs, hobby posts)? Without a healthy baseline, we cannot tell whether the cancer end of the gradient represents "normal" narrative structure or its own clinical pattern.
4. **Single-method TEA extraction.** The TEA pipeline depends entirely on rule-based SVO extraction from dependency parses. How many sentences fail to produce any SVO triple? If the extraction rate differs across groups (plausible given genre differences), this is a systematic bias.
5. **"Intermediate position" is not a finding.** Any three ordered means will have one in the middle. The interesting question is whether the intervals are meaningful — i.e., whether OCD is equidistant from cancer and depression/PTSD, or closer to one pole. Effect size asymmetry (d = +1.37 vs. d = −0.45 for self-focus) suggests OCD is much closer to depression/PTSD than to cancer, which complicates the "intermediate" framing.

### What would strengthen the paper
- A within-platform comparison (e.g., OCD posts from Reddit vs. depression/PTSD posts from Reddit) to isolate disorder from platform.
- SVO extraction rate reported per group.
- A multivariate test (e.g., MANOVA or discriminant analysis) rather than parallel univariate tests.
- A healthy-narrative baseline.

**Recommendation:** Major revision. The results are internally consistent but the inferential chain from data to clinical interpretation has too many gaps.

**Confidence:** 4/5

---

# Editorial Decision

## Summary of reviews

| Reviewer | Expertise | Recommendation | Key concern |
|---|---|---|---|
| R1 | Computational Linguistics | Major revision | NLP pipeline robustness (model choice, embeddings, threshold sensitivity) |
| R2 | Clinical Psychology | Major revision | Depression/PTSD pooling, recovery bias, missing clinical nuance |
| R3 | Methodology/Statistics | Minor revision | Multiple comparisons, matching overlap, variance assumptions |
| R4 | Network Science | Major revision | Shallow network analysis, no null model, unexploited graph structure |
| R5 | Devil's Advocate | Major revision | Confounded comparisons (platform/genre/curation), interpretive overclaiming |

## Decision: **MAJOR REVISION**

The paper presents an interesting and well-motivated comparison of illness narratives using two complementary computational methods. The consistent gradient (Cancer < OCD < Depression/PTSD) on self-focus and thematic recurrence is a real pattern in the data. However, four reviewers raise major concerns that must be addressed before the paper can be accepted:

1. **Confound management** (R2, R5): The platform/genre confound is the most serious issue. At minimum, the paper must either (a) include a within-platform control (OCD Reddit posts) or (b) substantially temper the clinical interpretation and reframe the contribution as a methodological demonstration.
2. **NLP pipeline validation** (R1): Sensitivity analyses for the spaCy model and RQA threshold are needed.
3. **Network analysis depth** (R4): The TEA analysis should go beyond summary metrics.
4. **Statistical refinement** (R3): Apply FDR correction, report SDs, address matching overlap.
5. **Clinical nuance** (R2): Separate depression from PTSD, address recovery bias more prominently.

---

# Revision Roadmap

## Priority 1 — Must address (rejection risk if ignored)

| # | Issue | Reviewers | Suggested action |
|---|---|---|---|
| 1.1 | **Platform/genre confound** | R2, R5 | Add a supplementary OCD-from-Reddit sample (even small, e.g., r/OCD) to test whether the gradient holds within a single platform. If not feasible, rewrite Discussion §4.1 and §4.4 to clearly separate "disorder-type" from "platform-type" explanations and present the clinical interpretation as one of several possibilities, not the primary conclusion. |
| 1.2 | **Depression/PTSD pooling** | R2 | Report supplementary analysis separating r/depression from r/ptsd. At minimum, show that the gradient holds for each subreddit separately. If it does not, revise the Discussion. |
| 1.3 | **Multiple comparison correction** | R3 | Apply Benjamini-Hochberg FDR correction to all 32 tests. Report which survive. Rewrite Section 3.4 and Conclusion to focus on surviving effects. |

## Priority 2 — Strongly recommended

| # | Issue | Reviewers | Suggested action |
|---|---|---|---|
| 2.1 | **RQA threshold sensitivity** | R1 | Run RQA at thresholds 0.90, 0.91, ..., 0.95 and report whether the gradient holds across all thresholds. A supplementary table or figure is sufficient. |
| 2.2 | **Sentence embedding quality** | R1 | Either rerun RQA with a sentence transformer (all-MiniLM-L6-v2) and compare, or argue why spaCy embeddings are sufficient for the pairwise similarity task. |
| 2.3 | **Network null model** | R4 | For each TEA metric, compute the metric on degree-preserving random graphs and report z-scores. This can be a supplementary table. |
| 2.4 | **SVO extraction rate** | R5 | Report mean SVO extraction rate (triples per sentence) per group. If rates differ across groups, discuss implications. |
| 2.5 | **Recovery bias discussion** | R2 | Move recovery bias from a brief limitations mention to a dedicated paragraph in Discussion, discussing how recovery stage might inflate or deflate the observed gradient. |

## Priority 3 — Recommended improvements

| # | Issue | Reviewers | Suggested action |
|---|---|---|---|
| 3.1 | **Matching overlap** | R3 | Report what fraction of OCD texts appear in both comparison pairs. If it is low, discuss. |
| 3.2 | **SDs in tables** | R3 | Add SD columns to Tables 1 and 2. |
| 3.3 | **Exact p-values** | R3 | Report exact p-values where possible. |
| 3.4 | **Hedges' g** | R3 | Use Hedges' g instead of Cohen's d (bias-corrected for small samples). |
| 3.5 | **Deeper network analysis** | R4 | Add one of: community detection, motif analysis, or sentiment flow analysis. This is a strong enhancement but not strictly required for the current contribution. |
| 3.6 | **OCD subtype discussion** | R2 | Add a paragraph in Limitations discussing subtype heterogeneity. |
| 3.7 | **TEA figures for all groups** | R4 | Add cancer and depression/PTSD TEA network figures (supplementary). |
| 3.8 | **Coreference resolution details** | R1 | Specify which coreference module was used and its expected accuracy. |
| 3.9 | **"Intermediate" framing** | R5 | Discuss effect size asymmetry (much closer to Dep/PTSD than to Cancer on self-focus). Avoid implying equidistance. |
| 3.10 | **Code/data availability** | R1 | Replace `[repository]` placeholder with actual URL. |

## Estimated revision effort

- **Priority 1**: 1–2 weeks (new analyses + rewriting)
- **Priority 2**: 1 week (supplementary analyses + additional reporting)
- **Priority 3**: 2–3 days (text revisions + minor additions)
- **Total**: ~3 weeks
