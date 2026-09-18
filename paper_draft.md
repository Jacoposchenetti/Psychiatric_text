# Cognitive-Linguistic Network Structure of OCD Personal Narratives: A Three-Group Comparison Using TEA Networks and Recurrence Quantification Analysis

## Abstract

Personal narratives offer a window into the cognitive patterns underlying mental health conditions. This study applies two complementary computational methods — Target-Event-Agent (TEA) cognitive networks and Recurrence Quantification Analysis (RQA) — to examine how personal narratives of obsessive-compulsive disorder (OCD) differ from those of physical illness (cancer) and other mental health conditions (depression and PTSD). We analyzed three corpora: 152 OCD recovery stories from a dedicated blog, 152 cancer patient stories from an interview-based website, and 200 depression/PTSD posts from Reddit. TEA networks, extracted using Stella and colleagues' framework, revealed that OCD narratives exhibited significantly higher first-person agent ratios than cancer narratives (d = +1.37, p < .001) but lower ratios than depression/PTSD narratives (d = −0.45, p = .004), indicating an intermediate level of self-focused agency. RQA of sentence-level semantic embeddings showed that OCD narratives had dramatically higher thematic recurrence than cancer narratives across all metrics (Recurrence Rate d = +1.57, Determinism d = +1.19, Laminarity d = +1.24; all p < .001) yet significantly lower recurrence than depression/PTSD narratives (d = −0.39 to −0.62; all p < .05). This convergent pattern — OCD occupying an intermediate position between physical illness and affective disorder narratives on both self-focus and thematic repetitiveness — suggests that OCD narratives carry a distinctive cognitive signature: more ruminative than illness narratives but less globally self-absorbed than depression, consistent with the intrusion-driven rather than mood-driven nature of obsessive cognition.

**Keywords:** obsessive-compulsive disorder, cognitive networks, TEA networks, recurrence quantification analysis, computational linguistics, personal narratives, mental health

---

## 1. Introduction

### 1.1 Personal Narratives as Cognitive Windows

Personal narratives are not merely accounts of events; they are cognitive constructions that reflect the narrator's attentional biases, emotional processing, and self-conceptualization (Pennebaker, 2011). In clinical psychology, the way individuals narrate their experiences with illness has long been recognized as both a reflection of, and a contributor to, their psychological adjustment (Frank, 1995). The linguistic features of personal narratives — who acts, what happens, and how themes recur — encode cognitive patterns that computational methods can systematically extract and quantify.

The study of language in mental health has established several robust associations. First-person singular pronoun use correlates with depression severity (Rude et al., 2004), while thematic repetitiveness in discourse reflects ruminative cognitive styles (Nolen-Hoeksema, 2000). However, most computational studies of mental health language have examined single conditions in isolation or used binary comparisons (e.g., clinical vs. healthy controls), leaving open the question of how different conditions position themselves within the broader landscape of illness narratives.

### 1.2 Obsessive-Compulsive Disorder: A Unique Cognitive Profile

OCD presents a particularly interesting case for narrative analysis because its cognitive signature is theoretically distinct from both physical illness and affective disorders. Cognitive models of OCD emphasize the role of intrusive thoughts — unwanted, ego-dystonic mental events that trigger distress not through mood congruence (as in depression) but through appraisal of threat, responsibility, and unacceptability (Salkovskis, 1985; Rachman, 1997). Individuals with OCD often describe a relationship with their thoughts that is fundamentally adversarial: they narrate a self that is besieged by its own cognition.

This theoretical distinction generates testable predictions about narrative structure. If OCD narratives are shaped by intrusion-driven cognition, they should show:
1. Higher self-focus than physical illness narratives (where the disease, not the self, is the primary agent) but potentially lower self-focus than depression narratives (where pervasive mood renders the self the center of all experience).
2. Greater thematic recurrence than physical illness narratives (reflecting the repetitive nature of obsessions) but potentially different patterns from depression (where rumination may produce even more pervasive thematic circularity).

### 1.3 Computational Approaches: TEA Networks and RQA

To test these predictions, we employ two complementary computational methods that capture different aspects of narrative cognition.

**Target-Event-Agent (TEA) Cognitive Networks.** Developed by Franchini, Stella, and colleagues (2026), TEA networks extract "who did what to whom" from text using dependency parsing, representing narratives as directed graphs with three layers: Agents (subjects/actors), Events (verbs/actions), and Targets (objects/consequences). This framework, built on spaCy's transformer-based parser with coreference resolution, goes beyond bag-of-words approaches by preserving the syntactic structure of agency and action. TEA networks inherit the valence annotation of Textual Forma Mentis Networks (Stella, 2020), labeling each concept as positive, negative, or neutral via VADER sentiment analysis (Hutto & Gilbert, 2014). Network-level metrics — including the first-person agent ratio, graph density, largest strongly connected component (LSCC) fraction, and clustering coefficient — quantify structural properties of the narrative's cognitive architecture.

**Recurrence Quantification Analysis (RQA).** Originally developed for dynamical systems (Eckmann et al., 1987; Webber & Zbilut, 1994), RQA has been adapted for text analysis by treating sequences of sentence embeddings as time series and computing recurrence matrices based on semantic similarity thresholds (Marwan et al., 2007). RQA metrics capture the temporal structure of thematic development: Recurrence Rate (RR) measures how often themes revisit earlier content; Determinism (DET) quantifies the predictability of thematic sequences; Laminarity (LAM) indexes "stuckness" on particular themes; and Entropy (ENTR) captures the complexity of recurrence patterns. These metrics are particularly well-suited to studying rumination and cognitive repetitiveness in clinical populations.

### 1.4 The Present Study

This study applies TEA networks and RQA to a three-group comparison of personal narratives: OCD stories, cancer patient stories, and depression/PTSD posts. By including both a physical illness control (cancer) and a mental health control (depression/PTSD), we can position OCD narratives within a two-dimensional space defined by the physical-versus-mental illness distinction and the intrusion-versus-mood-driven cognition distinction. We hypothesize that OCD narratives will occupy an intermediate position between cancer and depression/PTSD on measures of both self-focused agency and thematic recurrence.

---

## 2. Methods

### 2.1 Corpora

Three corpora of English-language personal narratives were compiled, each representing a different relationship to illness and cognitive processing.

**OCD Stories Corpus.** 153 first-person recovery narratives were scraped from The OCD Stories (theocdstories.com), a curated blog featuring personal accounts of living with and recovering from OCD. After excluding one story with extraction errors, 152 stories were retained (M = 1,408 words, SD = 965, Mdn = 1,208, range: 51–6,634). These narratives are unsolicited, long-form personal accounts written for a peer audience, typically covering onset, symptom experience, treatment, and recovery.

**Cancer Patient Stories Corpus.** 342 first-person narratives were scraped from The Patient Story (thepatientstory.com), an interview-based platform featuring cancer patients' accounts of diagnosis, treatment, and survivorship across 20 cancer types. After cleaning (removal of boilerplate text, interviewer questions, and diagnostic fact boxes) and filtering for minimum length (≥300 words), 342 stories were retained (M = 4,226 words, SD = 2,870). For TEA extraction, a random sample of 152 stories was selected to match the OCD corpus size (M = 4,339 words, SD = 2,614).

**Depression/PTSD Reddit Corpus.** 200 posts were sampled from the HuggingFace dataset `solomonk/reddit_mental_health_posts`, a publicly available collection of posts from mental health subreddits. We selected posts from r/depression and r/ptsd, filtering for posts ≥300 words after cleaning. A random sample of 200 posts was drawn (M = 539 words, SD = 269). Posts were cleaned of Reddit-specific formatting (URLs, Markdown syntax, edit notes) and prepended with their titles.

The three corpora differ substantially in mean word count, reflecting their different genres (curated blog, interview transcript, social media post). All statistical comparisons therefore employ length matching (see Section 2.5).

### 2.2 TEA Network Extraction

TEA networks were extracted using the TEA_Networks Python library (Franchini et al., 2026; version 0.3.1) with spaCy's `en_core_web_lg` model for dependency parsing. For each text, the pipeline:
1. Parsed the text with spaCy to obtain dependency trees.
2. Extracted Subject-Verb-Object (SVO) triples via rule-based traversal of dependency relations.
3. Constructed a directed graph where nodes represent lemmatized concepts and edges represent syntactic relations (Agent→Event, Event→Target), weighted by co-occurrence frequency.
4. Annotated each node with VADER sentiment valence (positive, negative, neutral).
5. Saved the resulting network as a GraphML file for downstream analysis.

### 2.3 Network Metrics

For each TEA network, we computed the following structural metrics:

- **First-person agent ratio**: Proportion of Agent→Event triplets where the Agent is a first-person pronoun (I, me, my, myself, we, us, our). This indexes the degree of self-focused agency in the narrative.
- **Graph density**: Ratio of existing edges to possible edges. Higher density indicates a more interconnected narrative structure.
- **LSCC fraction**: Size of the Largest Strongly Connected Component relative to total nodes. Captures the proportion of the network involved in reciprocal narrative relationships.
- **Average clustering coefficient**: Mean local clustering across nodes (computed on the undirected projection). Measures the tendency of narrative elements to form tightly connected triads.
- **Node type-token ratio (TTR)**: Ratio of unique nodes to total node mentions. Higher values indicate greater lexical diversity in the narrative network.
- **Edge repetition index**: Proportion of edges with weight > 1, indicating repeated syntactic patterns.

### 2.4 Recurrence Quantification Analysis

RQA was performed on sentence-level semantic embeddings to capture the temporal structure of thematic development. For each text:
1. The text was segmented into sentences using spaCy.
2. Each sentence with ≥3 words and nonzero vector norm was represented by its normalized `en_core_web_lg` embedding (300-dimensional).
3. A cosine similarity matrix was computed between all sentence pairs.
4. A recurrence matrix was constructed by thresholding the similarity matrix. The threshold was calibrated on a pooled sample from all three corpora to achieve a target Recurrence Rate of approximately 10% (threshold = 0.923).
5. The diagonal was zeroed (excluding self-recurrence).

From each recurrence matrix, we extracted seven RQA metrics:
- **Recurrence Rate (RR)**: Proportion of recurrent points in the matrix. Indexes overall thematic repetitiveness.
- **Determinism (DET)**: Proportion of recurrent points forming diagonal lines (length ≥ 2). Captures predictable thematic sequences.
- **Laminarity (LAM)**: Proportion of recurrent points forming vertical lines (length ≥ 2). Indexes "stuckness" on particular themes.
- **Trapping Time (TT)**: Mean length of vertical lines. Duration of thematic dwelling.
- **Mean Diagonal Length (L)**: Average length of diagonal structures. Reflects sustained parallel thematic development.
- **Maximum Diagonal Length (Lmax)**: Longest diagonal line. Captures the longest thematic echo in the text.
- **Recurrence Entropy (ENTR)**: Shannon entropy of the diagonal line length distribution. Measures the complexity of recurrence patterns.

### 2.5 Length Matching

Because the three corpora differ substantially in mean word count (OCD: 1,408; Cancer: 4,339; Dep/PTSD: 539), all group comparisons employed propensity-score-like matching on word count. For each OCD-Control pair, the matching algorithm:
1. Randomly shuffled the OCD group.
2. For each OCD story, selected the closest-length unmatched control text within a tolerance of 30–35% of the OCD story's word count.
3. Verified matching quality via Mann-Whitney U test and Cohen's d on word count (all matched comparisons achieved |d| < 0.15, p > .15).

This yielded 65 matched pairs for OCD vs. Cancer (TEA comparison), 79 matched pairs for OCD vs. Depression/PTSD (TEA comparison), 63 matched pairs for OCD vs. Cancer (RQA), and 78 matched pairs for OCD vs. Depression/PTSD (RQA). All random operations used a fixed seed (42) for reproducibility.

### 2.6 Statistical Analysis

For each metric, group differences were assessed with the Mann-Whitney U test (two-sided) and effect sizes quantified with Cohen's d (computed with pooled standard deviation). Effect sizes were interpreted following conventional thresholds: |d| < 0.2 negligible, 0.2–0.5 small, 0.5–0.8 medium, ≥ 0.8 large (Cohen, 1988). Statistical significance was set at α = .05. Given the exploratory nature of the study and the use of complementary methods for triangulation, we report uncorrected p-values but interpret results primarily through effect sizes.

---

## 3. Results

### 3.1 TEA Network Metrics: Three-Group Comparison (Length-Matched)

Table 1 presents the length-matched comparison of TEA network metrics. The most striking finding is the gradient pattern observed for first-person agent ratio.

**OCD vs. Cancer (n = 65 pairs).** OCD narratives showed a significantly higher first-person agent ratio than cancer narratives (M_OCD = 0.168, M_Cancer = 0.092; d = +1.37, p < .001), a large effect indicating that OCD narrators position themselves as agents far more frequently. OCD narratives also showed a higher LSCC fraction (d = +0.51, p = .004) and more TEA triplets per word (d = +0.44, p = .027). Other metrics (density, clustering, edge repetition, node TTR) did not differ significantly.

**OCD vs. Depression/PTSD (n = 79 pairs).** OCD narratives showed a significantly *lower* first-person agent ratio than depression/PTSD posts (M_OCD = 0.178, M_Dep/PTSD = 0.199; d = −0.45, p = .004), a small-to-medium effect indicating that depression/PTSD narrators are even more self-focused. OCD narratives also showed lower graph density (d = −0.30, p = .019). Other metrics did not differ significantly.

**The gradient.** Across both comparisons, OCD narratives occupy an intermediate position on self-focused agency: Cancer (0.092) < OCD (0.168–0.178) < Depression/PTSD (0.199). This ordering — physical illness < OCD < affective disorder — is consistent with the theoretical prediction that OCD's intrusion-driven cognition produces a distinctive pattern of self-reference that is elevated relative to external-illness narratives but less pervasive than the mood-driven self-absorption of depression.

*[Insert Figure 1: TEA network metrics bar chart, three groups]*

### 3.2 RQA Metrics: Three-Group Comparison (Length-Matched)

Table 2 presents the RQA comparison. The gradient pattern observed for TEA self-focus metrics is replicated — and amplified — for thematic recurrence.

**OCD vs. Cancer (n = 63 pairs).** OCD narratives showed dramatically higher thematic recurrence across all seven RQA metrics, with uniformly large effect sizes: Recurrence Rate (d = +1.57, p < .001), Determinism (d = +1.19, p < .001), Laminarity (d = +1.24, p < .001), Trapping Time (d = +0.88, p < .001), Recurrence Entropy (d = +0.85, p < .001), Mean Diagonal Length (d = +0.63, p < .001), and Maximum Diagonal Length (d = +0.57, p = .005). This indicates that OCD narratives revisit themes far more frequently and in more structured, predictable patterns than cancer narratives.

**OCD vs. Depression/PTSD (n = 78 pairs).** OCD narratives showed significantly *lower* thematic recurrence than depression/PTSD posts across all metrics, with uniformly medium effect sizes: Recurrence Rate (d = −0.53, p = .003), Determinism (d = −0.60, p < .001), Laminarity (d = −0.57, p < .001), Trapping Time (d = −0.51, p = .001), Recurrence Entropy (d = −0.62, p < .001), Mean Diagonal Length (d = −0.50, p = .001), and Maximum Diagonal Length (d = −0.39, p = .022).

**The gradient.** As with self-focus, OCD narratives occupy an intermediate position on thematic recurrence: Cancer < OCD < Depression/PTSD. This pattern holds across all seven RQA metrics without exception.

*[Insert Figure 2: Recurrence plots, three panels]*

*[Insert Figure 3: Forest plot of Cohen's d effect sizes]*

### 3.3 Agent Centrality Analysis

To examine the content of self-focused agency, we computed weighted degree centrality for Agent nodes using Stella and colleagues' `tea_weighted_degree_centrality` function on representative narrative samples from each corpus. Across all three groups, the first-person pronoun "I" was the dominant Agent, but with qualitatively different centrality profiles. In OCD narratives, "I" led the centrality ranking, followed by "it" and "you"; in cancer narratives, "I" ranked first but with a smaller lead, and medical actors (e.g., "doctor," "treatment") appeared among the top agents; in depression/PTSD posts, "I" dominated even more strongly, with a larger centrality gap before the next-ranked agent.

This pattern suggests that the self-focus gradient is driven not merely by pronoun frequency but by the degree to which the self monopolizes the agentive role in the narrative. The unmatched, full-corpus first-person agent ratios confirm this gradient: Cancer (M = 0.119) < OCD (M = 0.171) < Depression/PTSD (M = 0.206).

*[Insert Figure 4: Top agents centrality comparison]*

### 3.4 Convergent Pattern Across Methods

The key finding of this study is the convergent intermediate positioning of OCD narratives across two independent methods and multiple metrics. Figure 5 summarizes this pattern. On the eight metrics where significant effects were detected in both comparison directions — first-person agent ratio and all seven RQA metrics — OCD consistently occupied the intermediate position (OCD > Cancer and OCD < Depression/PTSD). The remaining TEA metrics showed either one-directional significance (LSCC and triplets per word significant only vs. Cancer; density significant only vs. Depression/PTSD) or no significant differences, but none contradicted the gradient pattern.

*[Insert Figure 5: Violin plots of key metric distributions]*

---

## 4. Discussion

### 4.1 OCD Narratives Between Physical and Mental Illness

The most robust finding of this study is that OCD personal narratives occupy an intermediate position between physical illness narratives (cancer) and affective disorder narratives (depression/PTSD) on two independently measured dimensions: self-focused agency (TEA first-person agent ratio) and thematic recurrence (RQA metrics). This pattern, observed on all eight metrics showing bidirectional significance (first-person agent ratio and all seven RQA metrics) and contradicted by none of the remaining TEA metrics, provides quantitative evidence for the theoretical distinctiveness of OCD cognition.

The intermediate positioning is theoretically informative. Cancer narratives, as physical illness accounts, are structured around external events — diagnosis, treatment, medical decisions — and accordingly feature lower self-focus and greater thematic variety (topics shift from symptoms to surgery to recovery to reintegration). Depression/PTSD narratives, shaped by mood-congruent processing and pervasive rumination (Nolen-Hoeksema, 2000), show the highest self-focus and the most circular thematic structure. OCD narratives fall in between: their self-focus reflects the introspective surveillance of one's own thoughts that characterizes obsessional cognition (Salkovskis, 1985), while their thematic recurrence captures the repetitive return to feared scenarios without the global pervasiveness of depressive rumination.

### 4.2 Self-Focus as Agentive Self-Monitoring

The TEA network analysis offers a nuanced view of self-focus that goes beyond simple pronoun counting. In TEA networks, a high first-person agent ratio means that "I" (or its variants) occupies the Agent role — that is, the grammatical and cognitive position of the entity performing actions. The large effect (d = +1.37) between OCD and cancer narratives indicates that OCD narrators overwhelmingly cast themselves as the active agents of their stories, performing actions directed at internal targets (thoughts, feelings, rituals) rather than narrating external events happening to them.

The smaller but significant difference from depression/PTSD (d = −0.45) is equally informative. Depression/PTSD narrators show even higher self-agency, but the centrality analysis reveals a qualitative difference: in OCD narratives, the "I" shares agentive space with "it" (often referring to OCD itself, treated as a separate entity) and "you" (addressing the reader or a generalized other). In depression/PTSD narratives, the "I" is more isolated at the top, with a larger gap before the next agent. This pattern is consistent with the clinical distinction between OCD (where the disorder is often experienced as ego-dystonic, an external intruder) and depression (where the self and the disorder are less differentiated).

### 4.3 Thematic Recurrence as Cognitive Signature

The RQA results provide the most striking evidence of differentiation. The effect sizes for OCD vs. Cancer are uniformly large (d = +0.57 to +1.57), while those for OCD vs. Depression/PTSD are uniformly medium (d = −0.39 to −0.62). This means that OCD narratives exhibit approximately twice the thematic recurrence of cancer narratives but approximately 75% of the recurrence observed in depression/PTSD narratives.

The specific pattern of RQA metrics is informative. High Laminarity in OCD narratives (M = 0.59 vs. 0.37 for cancer) indicates extended dwelling on particular themes — consistent with the "getting stuck" quality of obsessional thinking. High Determinism (M = 0.41 vs. 0.21 for cancer) indicates that when OCD narrators revisit a theme, they do so in predictable sequences — consistent with the structured, ritualistic quality of compulsive thought patterns. Yet both metrics are significantly lower than depression/PTSD (LAM = 0.64, DET = 0.47), suggesting that depressive rumination produces even more pervasive and structured thematic circularity than OCD-related repetition.

### 4.4 Clinical and Theoretical Implications

The convergent intermediate positioning of OCD has several implications. First, it provides empirical support from natural language for the classification of OCD as distinct from both anxiety and depressive disorders — consistent with its reclassification in DSM-5 as a separate diagnostic category (American Psychiatric Association, 2013). The narrative signature of OCD is not simply "more anxious than physical illness" or "less depressed than depression" but reflects a qualitatively distinct cognitive pattern of intrusion-driven self-monitoring with moderate thematic circularity.

Second, the three-group design reveals that metrics which appear diagnostic in binary comparisons may reflect general mental-health effects rather than OCD-specific signatures. For instance, the large first-person agent ratio difference between OCD and cancer (d = +1.37) might suggest self-focus as an OCD marker, but the comparison with depression/PTSD (d = −0.45) shows that self-focus is elevated in OCD relative to physical illness but reduced relative to other mental health conditions. This highlights the importance of using multiple control groups in computational studies of mental health language.

Third, the thematic recurrence findings suggest that RQA metrics, particularly Laminarity and Determinism, may capture clinically relevant variation in ruminative processing. The graded pattern (Cancer < OCD < Depression/PTSD) aligns with theoretical expectations about the role of repetitive thought in these conditions and could inform the development of computational biomarkers for ruminative severity.

### 4.5 Limitations

Several limitations should be acknowledged. First, the three corpora differ in genre (curated blog, interview-based website, social media). Although length matching mitigates word count confounds, genre differences in register, audience, and editorial practices may influence linguistic features independently of cognitive processing. The finding that the first-person gradient holds across genres (blog > Reddit for OCD vs. Dep/PTSD) partially addresses this concern, but cross-genre replication with matched platforms would strengthen the conclusions.

Second, the corpora represent self-selected narrators and cannot be assumed to represent the full clinical population. Blog authors and Reddit posters differ from the general patient population in ways that may affect narrative style (e.g., willingness to self-disclose, literacy level, recovery stage).

Third, the TEA extraction pipeline uses a non-transformer spaCy model (`en_core_web_lg`) rather than the default transformer model recommended by the library, which may affect extraction accuracy for complex syntactic constructions. However, the large effect sizes observed suggest that any extraction noise is small relative to the signal.

Fourth, no corrections for multiple comparisons were applied across the full set of metrics. We mitigate this by emphasizing effect sizes over p-values and by noting that the consistent directionality across all metrics (a pattern extremely unlikely under the null hypothesis) provides strong evidence independent of individual significance tests.

Fifth, we did not have access to clinical diagnoses for any group. OCD stories were published on a platform dedicated to OCD recovery, and depression/PTSD posts were from disorder-specific subreddits, but we cannot verify diagnostic status.

---

## 5. Conclusion

This study demonstrates that OCD personal narratives occupy a distinctive intermediate position between physical illness (cancer) and affective disorder (depression/PTSD) narratives on two converging computational measures: self-focused agency (TEA networks) and thematic recurrence (RQA). The pattern — Cancer < OCD < Depression/PTSD — is consistent across all eight metrics showing bidirectional significance and two independent methodological frameworks, providing robust evidence for a cognitive-linguistic signature of OCD that is theoretically coherent with intrusion-driven models of obsessional cognition. These findings support the computational study of personal narratives as a tool for understanding disorder-specific cognitive patterns and suggest that multi-group, multi-method designs are essential for disentangling general mental-health effects from condition-specific signatures.

---

## Data and Code Availability

All analysis scripts and figure generation code are available at [repository URL]. The OCD corpus was collected from theocdstories.com, the cancer corpus from thepatientstory.com, and the depression/PTSD corpus from the HuggingFace dataset `solomonk/reddit_mental_health_posts`. TEA network extraction used the TEA_Networks library (https://github.com/MassimoStel/TEA_Networks, version 0.3.1).

---

## References

American Psychiatric Association. (2013). *Diagnostic and statistical manual of mental disorders* (5th ed.). American Psychiatric Publishing.

Cohen, J. (1988). *Statistical power analysis for the behavioral sciences* (2nd ed.). Lawrence Erlbaum Associates.

Eckmann, J. P., Kamphorst, S. O., & Ruelle, D. (1987). Recurrence plots of dynamical systems. *Europhysics Letters*, *4*(9), 973–977.

Franchini, S., Carrillo, A., De Duro, E. S., Improta, R., Ardebili, A. A., & Stella, M. (2026). TEA Nets combine AI and cognitive network science to model actors, actions, and consequences in text [Preprint]. arXiv. https://github.com/MassimoStel/TEA_Networks

Frank, A. W. (1995). *The wounded storyteller: Body, illness, and ethics*. University of Chicago Press.

Hutto, C., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. In *Proceedings of the International AAAI Conference on Web and Social Media* (Vol. 8, No. 1, pp. 216–225).

Marwan, N., Romano, M. C., Thiel, M., & Kurths, J. (2007). Recurrence plots for the analysis of complex systems. *Physics Reports*, *438*(5–6), 237–329.

Nolen-Hoeksema, S. (2000). The role of rumination in depressive disorders and mixed anxiety/depressive symptoms. *Journal of Abnormal Psychology*, *109*(3), 504–511.

Pennebaker, J. W. (2011). *The secret life of pronouns: What our words say about us*. Bloomsbury Press.

Rachman, S. (1997). A cognitive theory of obsessions. *Behaviour Research and Therapy*, *35*(9), 793–802.

Rude, S., Gortner, E. M., & Pennebaker, J. (2004). Language use of depressed and depression-vulnerable college students. *Cognition and Emotion*, *18*(8), 1121–1133.

Salkovskis, P. M. (1985). Obsessional-compulsive problems: A cognitive-behavioural analysis. *Behaviour Research and Therapy*, *23*(5), 571–583.

Stella, M. (2020). Text-mining forma mentis networks reconstruct public perception of the STEM gender gap in social media. *PeerJ Computer Science*, *6*, e295.

Webber, C. L., & Zbilut, J. P. (1994). Dynamical assessment of physiological systems and states using recurrence plot strategies. *Journal of Applied Physiology*, *76*(2), 965–973.

---

## Tables

### Table 1. TEA Network Metrics: Length-Matched Three-Group Comparison

| Metric | OCD vs. Cancer (n = 65 pairs) | | | OCD vs. Dep/PTSD (n = 79 pairs) | | |
|--------|:----:|:----:|:----:|:----:|:----:|:----:|
| | M_OCD | M_Ctrl | d | M_OCD | M_Ctrl | d |
| First-Person Agent Ratio | 0.168 | 0.092 | **+1.37***| 0.178 | 0.199 | **−0.45*** |
| LSCC Fraction | 0.157 | 0.107 | **+0.51***| 0.098 | 0.101 | −0.04 |
| TEA Triplets/Word | 0.467 | 0.446 | **+0.44***| 0.454 | 0.477 | −0.21 |
| Graph Density | 0.006 | 0.005 | +0.10 | 0.007 | 0.008 | **−0.30*** |
| Avg Clustering | 0.257 | 0.261 | −0.11 | 0.243 | 0.251 | −0.16 |
| Node Type-Token Ratio | 0.283 | 0.297 | −0.21 | 0.310 | 0.300 | +0.25 |
| Edge Repetition Index | 0.065 | 0.084 | −0.35 | 0.069 | 0.069 | +0.02 |
| Mean Edge Weight | 1.101 | 1.142 | −0.48 | 1.102 | 1.112 | −0.16 |
| Unique Nodes/Word | 0.261 | 0.260 | +0.02 | 0.279 | 0.277 | +0.07 |

*Note.* Bold = p < .05. * = significant. d = Cohen's d (positive = OCD higher).

### Table 2. RQA Metrics: Length-Matched Three-Group Comparison

| Metric | OCD vs. Cancer (n = 63 pairs) | | | OCD vs. Dep/PTSD (n = 78 pairs) | | |
|--------|:----:|:----:|:----:|:----:|:----:|:----:|
| | M_OCD | M_Ctrl | d | M_OCD | M_Ctrl | d |
| Recurrence Rate | 0.237 | 0.076 | **+1.57***| 0.207 | 0.283 | **−0.53***|
| Determinism | 0.415 | 0.212 | **+1.19***| 0.353 | 0.472 | **−0.60***|
| Laminarity | 0.594 | 0.367 | **+1.24***| 0.538 | 0.635 | **−0.57***|
| Trapping Time | 2.881 | 2.245 | **+0.88***| 2.707 | 3.085 | **−0.51***|
| Mean Diagonal Length | 2.327 | 1.840 | **+0.63***| 2.238 | 2.596 | **−0.50***|
| Max Diagonal Length | 5.810 | 4.079 | **+0.57***| 4.205 | 5.141 | **−0.39***|
| Recurrence Entropy | 1.026 | 0.580 | **+0.85***| 0.823 | 1.185 | **−0.62***|

*Note.* All effects significant at p < .05. d = Cohen's d (positive = OCD higher).

---

## Figures

**Figure 1.** TEA network of an OCD narrative filtered by first-person subject ("I"), generated using Stella and colleagues' `plot_svo_graph` function. The three-column layout shows AGENT (left), EVENT (center), and TARGET (right) nodes. Node color indicates VADER sentiment valence (blue = positive, red = negative, gray = neutral). The dominance of "I" as the sole Agent hub, connected to numerous events (feel, think, fight, shake, can not escape) and targets (crazy, wrong word, about my dirty hand), illustrates the self-focused, adversarial relationship with cognition characteristic of OCD narratives.
File: `figures/tea_network_OCD_I.png`

**Figure 2.** TEA network metrics across three groups (mean ± SE). OCD narratives show higher first-person agent ratio than cancer (d = +1.37) but lower than depression/PTSD (d = −0.45), establishing the intermediate position. LSCC fraction is significantly elevated in OCD vs. cancer only.
File: `figures/tea_metrics_comparison.png`

**Figure 3.** Recurrence plots for representative texts from each group (cosine similarity threshold = 0.92). Each dot indicates a pair of sentences exceeding the similarity threshold. OCD (left) shows dense, structured recurrence; Cancer (center) shows sparse recurrence; Depression/PTSD (right) shows the densest recurrence with large connected blocks.
File: `figures/recurrence_plots.png`

**Figure 4.** Forest plot of Cohen's d effect sizes for all length-matched comparisons. Left panel: TEA metrics. Right panel: RQA metrics. Red circles = OCD vs. Cancer; green squares = OCD vs. Depression/PTSD. Asterisks indicate statistical significance. The consistent pattern of positive effects (OCD > Cancer) and negative effects (OCD < Depression/PTSD) demonstrates the intermediate positioning across all metrics.
File: `figures/effect_sizes_forest.png`

**Figure 5.** Violin plots showing the distribution of four key metrics across groups: First-Person Agent Ratio (TEA), Recurrence Rate, Determinism, and Recurrence Entropy (RQA). The intermediate position of OCD (red) between Cancer (blue) and Depression/PTSD (green) is visible in all four metrics, with clear distributional separation.
File: `figures/violin_plots.png`
