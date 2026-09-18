"""Insert figures into paper_draft.docx at the marked positions."""
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os, re

BASE = r"C:\Users\jsche\Desktop\NLP\ocd_stories"
FIGURES_DIR = os.path.join(BASE, "figures")

FIGURE_MAP = {
    "*[Insert Figure 1: TEA network metrics bar chart, three groups]*": {
        "file": "tea_metrics_comparison.png",
        "caption": "Figure 1. TEA network metrics across three groups (mean \u00b1 SE). "
                   "OCD narratives show higher first-person agent ratio than cancer (d = +1.37) "
                   "but lower than depression/PTSD (d = \u22120.45), establishing the intermediate position.",
        "width": 6.5,
    },
    "*[Insert Figure 2: Recurrence plots, three panels]*": {
        "file": "recurrence_plots.png",
        "caption": "Figure 2. Recurrence plots for representative texts from each group "
                   "(cosine similarity threshold = 0.92). Each dot indicates a pair of sentences "
                   "exceeding the similarity threshold. OCD (left) shows dense, structured recurrence; "
                   "Cancer (center) shows sparse recurrence; Depression/PTSD (right) shows the densest "
                   "recurrence with large connected blocks.",
        "width": 6.5,
    },
    "*[Insert Figure 3: Forest plot of Cohen\u2019s d effect sizes]*": {
        "file": "effect_sizes_forest.png",
        "caption": "Figure 3. Forest plot of Cohen\u2019s d effect sizes for all length-matched comparisons. "
                   "Left panel: TEA metrics. Right panel: RQA metrics. Red circles = OCD vs. Cancer; "
                   "green squares = OCD vs. Depression/PTSD. Asterisks indicate statistical significance.",
        "width": 6.5,
    },
    "*[Insert Figure 4: Top agents centrality comparison]*": {
        "file": "top_agents_centrality.png",
        "caption": "Figure 4. Top 10 Agent nodes by weighted degree centrality for each corpus. "
                   "The first-person pronoun \u201cI\u201d dominates all three corpora but with different relative "
                   "centrality, reflecting the self-focus gradient.",
        "width": 6.5,
    },
    "*[Insert Figure 5: Violin plots of key metric distributions]*": {
        "file": "violin_plots.png",
        "caption": "Figure 5. Violin plots showing the distribution of four key metrics across groups: "
                   "First-Person Agent Ratio (TEA), Recurrence Rate, Determinism, and Recurrence Entropy (RQA). "
                   "The intermediate position of OCD (red) between Cancer (blue) and Depression/PTSD (green) "
                   "is visible in all four metrics.",
        "width": 6.5,
    },
}

TEA_NETWORK_FIGURE = {
    "file": "tea_network_OCD_I.png",
    "caption": "Figure 6. TEA network of OCD narratives filtered by first-person subject (\u201cI\u201d), "
               "generated using Stella and colleagues\u2019 plot_svo_graph function. The three-column layout "
               "shows AGENT (left), EVENT (center), and TARGET (right) nodes. Node color indicates "
               "VADER sentiment valence (blue = positive, red = negative, gray = neutral).",
    "width": 6.5,
}


def add_figure(doc, paragraph_index, fig_info):
    """Replace the paragraph at paragraph_index with image + caption."""
    p = doc.paragraphs[paragraph_index]
    p.clear()
    run = p.add_run()
    img_path = os.path.join(FIGURES_DIR, fig_info["file"])
    if not os.path.exists(img_path):
        p.text = f"[IMAGE NOT FOUND: {fig_info['file']}]"
        return
    run.add_picture(img_path, width=Inches(fig_info["width"]))
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    caption_para = doc.add_paragraph()
    caption_run = caption_para.add_run(fig_info["caption"])
    caption_run.font.size = Pt(9)
    caption_run.font.italic = True
    caption_run.font.color.rgb = RGBColor(80, 80, 80)
    caption_para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    caption_para.paragraph_format.space_after = Pt(12)

    body = doc.element.body
    p_elem = p._element
    cap_elem = caption_para._element
    body.remove(cap_elem)
    p_elem.addnext(cap_elem)


def main():
    src = os.path.join(BASE, "paper_draft_fixed.docx")
    doc = Document(src)

    indices_to_replace = []
    for i, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        for marker, fig_info in FIGURE_MAP.items():
            clean_marker = marker.replace("\u2019", "'").replace("'", "'")
            clean_text = text.replace("\u2019", "'").replace("'", "'")
            if clean_marker.strip("*") in clean_text or clean_marker in text:
                indices_to_replace.append((i, fig_info))
                break

    for idx, fig_info in reversed(indices_to_replace):
        print(f"  Inserting {fig_info['file']} at paragraph {idx}")
        add_figure(doc, idx, fig_info)

    figures_section_idx = None
    for i, para in enumerate(doc.paragraphs):
        if para.text.strip() == "Figures":
            figures_section_idx = i
            break

    if figures_section_idx is not None:
        tea_fig_idx = None
        for i in range(figures_section_idx, len(doc.paragraphs)):
            if "tea_network_OCD_I" in doc.paragraphs[i].text or "TEA network of an OCD narrative" in doc.paragraphs[i].text:
                tea_fig_idx = i
                break
        if tea_fig_idx:
            print(f"  Inserting TEA network figure at paragraph {tea_fig_idx}")
            p = doc.paragraphs[tea_fig_idx]
            p.clear()
            run = p.add_run()
            img_path = os.path.join(FIGURES_DIR, TEA_NETWORK_FIGURE["file"])
            if os.path.exists(img_path):
                run.add_picture(img_path, width=Inches(TEA_NETWORK_FIGURE["width"]))
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    out = os.path.join(BASE, "paper_with_figures.docx")
    doc.save(out)
    print(f"\nSaved: {out}")
    print(f"Figures inserted: {len(indices_to_replace)}")


if __name__ == "__main__":
    main()
