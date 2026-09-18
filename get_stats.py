import pandas as pd
import numpy as np

ocd = pd.read_csv("results/tea_metrics.csv")
cancer = pd.read_csv("control_patient_stories/results/tea_metrics.csv")
mh = pd.read_csv("control_mental_health/results/tea_metrics.csv")

print("=== Corpus sizes ===")
for name, df in [("OCD", ocd), ("Cancer", cancer), ("Dep/PTSD", mh)]:
    wc = df["n_words"]
    print(f"  {name}: n={len(df)}, mean_wc={wc.mean():.0f}, SD={wc.std():.0f}, median={wc.median():.0f}, range=[{wc.min()}-{wc.max()}]")

ocd_rqa = pd.read_csv("results/rqa_ocd.csv")
cancer_rqa = pd.read_csv("results/rqa_cancer.csv")
mh_rqa = pd.read_csv("results/rqa_mh.csv")
print(f"\n=== RQA sizes: OCD={len(ocd_rqa)}, Cancer={len(cancer_rqa)}, MH={len(mh_rqa)} ===")

print("\n=== TEA descriptives (full corpus, pre-matching) ===")
for name, df in [("OCD", ocd), ("Cancer", cancer), ("Dep/PTSD", mh)]:
    for col in ["first_person_ratio", "lscc_frac", "density", "avg_clustering", "node_ttr", "repetition_idx"]:
        vals = df[col].dropna()
        print(f"  {name} {col}: M={vals.mean():.4f} SD={vals.std():.4f}")
    print()

print("=== RQA descriptives (full corpus, pre-matching) ===")
for name, df in [("OCD", ocd_rqa), ("Cancer", cancer_rqa), ("Dep/PTSD", mh_rqa)]:
    for col in ["RR", "DET", "LAM", "TT", "ENTR", "L"]:
        vals = df[col].dropna()
        print(f"  {name} {col}: M={vals.mean():.4f} SD={vals.std():.4f}")
    print()
