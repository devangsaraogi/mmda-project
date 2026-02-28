# Devang's Midterm Plan — Visual Modality (Image Evidence)

> **Role**: Evaluate image evidence for claim verification using CLIP
> **Model**: CLIP (ViT-L/14) — cosine similarity between claim and image evidence
> **Question**: "How useful is image evidence alone for verifying textual claims?"

---

## Part A: What Needs to Be Done

### A1. Shared Work (Devang + Mahim)

These are needed by BOTH reports and should be done together or divided early.

- [ ] **Dataset Preparation (WebQA-Adv)**
  - Download WebQA dataset (~34K QA pairs, ~390K images, ~540K text snippets)
  - Build the adversarial augmentation pipeline:
    - True claims (40%): reformat questions as declarative statements
    - Text contradictions (25%): negate or alter claims, keep correct images
    - Visual contradictions (20%): swap images between related topics
    - Temporal mismatches (10%): pair content from different time periods
    - Ambiguous (5%): insufficient evidence cases
  - Create train/val/test splits
  - Compute dataset statistics (class distribution, image resolution stats, text lengths, etc.)

- [ ] **Sections 2–4 of the report** (build on first submission — shared content, adapted per person)
  - Section 2 (Introduction): mostly reusable, add contributions paragraph specific to your modality
  - Section 3 (Related Works): expand to 15+ references, add CLIP/ViT/visual verification literature
  - Section 4 (Dataset): write precise technical description with stats

### A2. Devang's Individual Work

#### Code & Experiments

- [ ] **CLIP Baseline Implementation**
  - Load pre-trained CLIP ViT-L/14 (frozen, no fine-tuning)
  - For each (claim, candidate_images) pair:
    - Encode claim text via CLIP text encoder
    - Encode each candidate image via CLIP image encoder
    - Compute cosine similarity scores
    - Use max/mean similarity as verification score
  - Map similarity scores to 3-way verdict (True / False / Unverifiable)
    - Approach 1: Threshold-based (tune thresholds on val set)
    - Approach 2: Train a small classifier (MLP) on top of similarity features

- [ ] **ViT Baseline (optional, if time permits)**
  - Fine-tune ViT-B/16 on images only (no claim text)
  - This is the "pure image" baseline — expected to perform poorly
  - Useful as a comparison point: "ViT alone can't do anything, CLIP at least captures claim-image relationship"

- [ ] **Ablation / Analysis**
  - Break down performance by misinformation type:
    - How does CLIP perform on visual contradictions (swapped images)?
    - How does CLIP perform on text contradictions (claim is wrong, image is fine)?
    - How does CLIP perform on temporal mismatches?
  - Qualitative examples:
    - Success cases: where CLIP correctly flags a mismatched image
    - Failure cases: where CLIP gives high similarity despite the claim being false
  - Confusion matrix (3-way: True/False/Unverifiable)
  - Similarity score distributions for true vs false claims

- [ ] **Generate Figures**
  - CLIP similarity score histogram (true claims vs false claims)
  - Confusion matrix
  - Per-class performance bar chart (Macro F1, accuracy per class)
  - Qualitative examples: 2-3 success cases, 2-3 failure cases with images
  - Optional: t-SNE/UMAP of CLIP embeddings colored by label

#### Report Writing (Individual Submission)

- [ ] **Section 1 — Abstract**
  - Summarize: problem → image modality analysis with CLIP → key results → takeaway (image evidence alone is insufficient)

- [ ] **Section 5.1 — Unimodal Models & Methods (1–1.5 pages)**
  - Explain CLIP architecture in detail:
    - Contrastive pre-training on 400M image-text pairs
    - Dual encoder: ViT-L/14 for images, Transformer for text
    - How cosine similarity in shared embedding space works
    - Why CLIP is appropriate: it's the strongest off-the-shelf claim-image matching model
  - Explain how you adapted CLIP for verification:
    - Similarity score computation
    - Threshold tuning or classifier on top
    - Pre-training data (LAION/OpenAI) and its relevance to web-sourced claims
  - If ViT baseline was run, describe it briefly as a comparison

- [ ] **Section 5.2 — Experiments & Evaluation (0.5 page)**
  - Train/val/test split sizes and strategy
  - Hardware: GPU type, memory
  - Software: PyTorch, transformers/open_clip library version
  - Hyperparameters: CLIP model variant, similarity thresholds, classifier LR/epochs if applicable
  - Metrics: Macro F1, Accuracy, per-class F1, Recall@K for evidence retrieval
  - No data augmentation on image side (frozen CLIP)

- [ ] **Section 5.3 — Results (1 page)**
  - Summary table: CLIP performance across metrics
  - If ViT baseline exists: comparison table
  - Confusion matrix
  - Breakdown by misinformation type
  - Qualitative examples (successes + failures with actual images)
  - Learning curves if classifier was trained on top

- [ ] **Section 5.4 — Discussion**
  - CLIP catches visual contradictions to some extent (swapped images have lower similarity)
  - CLIP fails on text contradictions (image is correct, claim is wrong — CLIP still sees high similarity)
  - CLIP fails on temporal mismatches (old vs new photo looks the same)
  - Key insight: similarity is not the same as factual verification
  - Discuss: what can be combined with Mahim's text results? Where do they complement?

- [ ] **Section 6 — Updated Research Vision (0.5 page)**
  - Image evidence alone is insufficient — limited to catching image swaps
  - Text evidence (Mahim's results) likely catches different error types
  - Complementarity motivates cross-modal fusion (CMVN)
  - Updated plan for multimodal phase: fuse CLIP image features with RoBERTa text features via cross-attention
  - Concrete next steps: implement Stage 2 contradiction detector

- [ ] **References**: Expand to 15+ (add CLIP variants, visual verification papers, OOC detection papers from research doc)

- [ ] **Appendix**: Annotated bibliography (graded), extended results, hyperparameter search

#### Presentation (20 min, individual)

- [ ] **Slides Structure** (~20 slides for 20 min):
  1. Title slide (1 min)
  2. Motivation: why multimodal misinformation matters (2 min)
  3. Dataset & Data Story: show actual WebQA-Adv examples with images (3 min) — **high grading weight**
  4. Related Works: cluster into (1) CLIP/VLMs for verification, (2) OOC detection, (3) gap (2 min)
  5. CLIP Model & Setup: architecture figure, how similarity scoring works (3 min)
  6. Results & Analysis: lead with best figure, confusion matrix, breakdown by type, qualitative examples (5 min) — **highest grading weight**
  7. Research Vision & Next Steps: what I learned, why multimodal is needed (3 min)
  8. Q&A slide: summary table visible (1 min)

- [ ] **Prepare for Q&A defense questions**:
  - "Why CLIP and not BLIP-2 or LLaVA?"
  - "Isn't CLIP already cross-modal? Why call this unimodal?"
  - "What if you fine-tuned CLIP instead of freezing it?"
  - "How do your results compare to Mahim's text results?"

---

## Part B: Timeline

| When | Task | Deliverable |
|------|------|-------------|
| **Week 1** | Download WebQA, build adversarial augmentation pipeline (with Mahim) | WebQA-Adv dataset with splits |
| **Week 1** | Implement CLIP similarity scoring pipeline | Working CLIP inference code |
| **Week 2** | Run CLIP on full dataset, tune thresholds, compute all metrics | Results tables + figures |
| **Week 2** | Run ablations: per-type breakdown, qualitative analysis | Analysis notebooks |
| **Week 3** | Write Sections 5.1–5.4 (experiments, results, discussion) | Draft of core sections |
| **Week 3** | Write Sections 1, 6, expand Section 3 to 15+ refs | Complete report draft |
| **Week 3** | Build presentation slides | Slide deck draft |
| **Week 4** | Revise report, finalize figures, write annotated bibliography | Final report |
| **Week 4** | Practice presentation (time it!), prepare Q&A answers | Final presentation |

---

## Part C: Key Technical Details

### CLIP Similarity Scoring Pipeline

```python
# Pseudocode
import clip

model, preprocess = clip.load("ViT-L/14")

for claim, images, label in dataset:
    text_features = model.encode_text(clip.tokenize(claim))
    image_features = model.encode_image(preprocess(images))

    # Cosine similarity
    similarity = (text_features @ image_features.T).squeeze()

    # Max similarity across candidate images
    max_sim = similarity.max()

    # Threshold-based verdict
    if max_sim > threshold_high:
        verdict = "TRUE"
    elif max_sim < threshold_low:
        verdict = "FALSE"
    else:
        verdict = "UNVERIFIABLE"
```

### Expected Results Pattern

| Misinformation Type | Expected CLIP Performance | Why |
|---|---|---|
| True claims | Moderate-high similarity | Image matches claim topic |
| Text contradiction | **High similarity (false negative)** | Image is correct, CLIP doesn't check text facts |
| Visual contradiction | **Lower similarity (detectable)** | Wrong image → lower match with claim |
| Temporal mismatch | High similarity (miss) | Old/new photos of same thing look similar |
| Ambiguous | Random | Insufficient signal |

### Metrics to Report

- Macro F1 (primary — handles class imbalance)
- Per-class F1 (True / False / Unverifiable)
- Overall Accuracy
- Confusion Matrix
- Similarity score distributions per class
- Recall@K if doing evidence retrieval

---

## Part D: Literature Grounding

### Why CLIP for Visual Verification — What the Literature Says

**1. CLIP cosine similarity is the standard baseline for OOC detection.**
Papadopoulos et al. (WACV 2025) in ["Similarity over Factuality"](https://openaccess.thecvf.com/content/WACV2025/papers/Papadopoulos_Similarity_over_Factuality_Are_we_Making_Progress_on_Multimodal_Out-of-Context_WACV_2025_paper.pdf) showed that many OOC detectors, including CLIP-based ones, exploit image-text *similarity* rather than actually reasoning about *factuality*. They use CLIP-based similarities (which they call MUSE — Multimodal Similarities) as a primary feature. This is directly relevant — your baseline tests exactly this: can similarity alone work?

**2. CLIP similarity is used in multiple misinformation detectors.**
- **FND-CLIP** ([arXiv 2205.14304](https://arxiv.org/abs/2205.14304)): Uses CLIP encoders alongside ResNet + BERT, concatenating features weighted by cross-modal similarity. Key insight: raw CLIP similarity is a useful *feature* but not sufficient alone.
- **SARD** ([ScienceDirect 2024](https://www.sciencedirect.com/science/article/pii/S1319157824002490)): Combines CLIP contrastive learning with heterogeneous information networks.
- **VAE-CLIP** ([Electronics 2024](https://www.mdpi.com/2079-9292/13/15/2958)): Uses CLIP similarity as one feature among many, feeding it into a VAE for fake news detection.
- **Chen et al. (CVPR Workshop 2023)**: ["Harnessing the Power of Text-Image Contrastive Models"](https://openaccess.thecvf.com/content/CVPR2023W/WMF/html/Chen_Harnessing_the_Power_of_Text-Image_Contrastive_Models_for_Automatic_Detection_CVPRW_2023_paper.html) — directly uses CLIP cosine similarity for automatic misinformation detection.

**3. CLIP struggles with fine-grained reasoning.**
- Huang et al. (ICASSP 2022): ["Text-Image De-contextualization Detection"](https://ieeexplore.ieee.org/abstract/document/9746193/) compared CLIP and VinVL for detecting different image-text inconsistency types. CLIP captures global similarity but misses fine-grained entity mismatches.
- CLIP achieves 75.4% top-1 on ImageNet but struggles with counting, fine-grained classification, and compositional reasoning — exactly the kind of reasoning needed for fact-checking.

**4. More sophisticated visual approaches exist (context for "why not these").**
- **SNIFFER** (CVPR 2024): Uses instruction-tuned InstructBLIP with two-stage checking. Outperforms CLIP by a large margin. You can cite this to explain what a stronger approach looks like, justifying your focus on understanding the baseline first.
- **D-TIIL** (ICLR 2024): Uses diffusion models to *localize* inconsistencies at word and image-region level. Shows the state-of-the-art has moved beyond similarity to localization.
- **Retrieval Augmented Verification** ([arXiv 2404.10702](https://arxiv.org/abs/2404.10702)): Zero-shot multimodal disinformation detection using graph-based entity representations + CLIP visual features + external evidence retrieval. Demonstrates that CLIP alone isn't enough — you need evidence retrieval on top.

**5. NewsCLIPpings proves CLIP can generate challenging OOC pairs.**
Luo et al. (EMNLP 2021) used CLIP to retrieve visually similar but contextually wrong images to create the NewsCLIPpings dataset (988K pairs). This validates that CLIP captures surface similarity but not semantic correctness — exactly the limitation your experiments will demonstrate.

**6. VERITE benchmark exposes unimodal bias.**
[Papadopoulos et al. (2023)](https://github.com/stevejpapad/image-text-verification): VERITE was designed to prevent models from "cheating" with unimodal features. When you design WebQA-Adv, ensure visual contradictions can't be caught by text alone and vice versa.

### Synthetic Misinformation Generation
Papadopoulos et al. (2023) in ["Synthetic Misinformers"](https://dl.acm.org/doi/abs/10.1145/3592572.3592842) (47 citations) used CLIP to generate synthetic OOC pairs by finding semantically similar but incorrect images — the same principle behind your WebQA-Adv visual contradiction generation.

### CLIP ViT-L/14 Specifics
- **Architecture**: 24-layer Vision Transformer, 14x14 patch size, 768-dim embeddings
- **Pre-training**: Contrastive loss on 400M image-text pairs (WebImageText dataset)
- **HuggingFace**: [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)
- **open_clip alternative**: [mlfoundations/open_clip](https://github.com/mlfoundations/open_clip) — open-source with LAION-2B trained variants

---

## Part E: References to Add (with Links)

Papers to cite (in addition to existing 10 in bib):

| # | Paper | Venue | Why Cite | Link |
|---|---|---|---|---|
| 1 | Qi et al. — SNIFFER | CVPR 2024 | State-of-art OOC detection with explanations | [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Qi_SNIFFER_Multimodal_Large_Language_Model_for_Explainable_Out-of-Context_Misinformation_Detection_CVPR_2024_paper.html) |
| 2 | Luo et al. — NewsCLIPpings | EMNLP 2021 | CLIP-based OOC dataset generation | [Paper](https://aclanthology.org/2021.emnlp-main.545/) |
| 3 | Papadopoulos et al. — VERITE | MIR 2023 | Unimodal bias in OOC benchmarks | [GitHub](https://github.com/stevejpapad/image-text-verification) |
| 4 | Papadopoulos et al. — Similarity over Factuality | WACV 2025 | CLIP measures similarity not factuality | [Paper](https://openaccess.thecvf.com/content/WACV2025/papers/Papadopoulos_Similarity_over_Factuality_Are_we_Making_Progress_on_Multimodal_Out-of-Context_WACV_2025_paper.pdf) |
| 5 | Huang et al. — D-TIIL | ICLR 2024 | Fine-grained inconsistency localization | [Paper](https://arxiv.org/abs/2404.18033) |
| 6 | Aneja et al. — COSMOS | AAAI 2023 | Self-supervised OOC detection | [Project](https://shivangi-aneja.github.io/projects/cosmos/) |
| 7 | Papadopoulos et al. — Synthetic Misinformers | MAD 2023 | CLIP for synthetic OOC generation | [Paper](https://dl.acm.org/doi/abs/10.1145/3592572.3592842) |
| 8 | Chen et al. — Harnessing Text-Image Contrastive Models | CVPRW 2023 | CLIP cosine sim for misinfo detection | [Paper](https://openaccess.thecvf.com/content/CVPR2023W/WMF/html/Chen_Harnessing_the_Power_of_Text-Image_Contrastive_Models_for_Automatic_Detection_CVPRW_2023_paper.html) |
| 9 | Dey et al. — Retrieval Augmented Verification | arXiv 2024 | Zero-shot CLIP + evidence retrieval | [Paper](https://arxiv.org/abs/2404.10702) |
| 10 | Fan et al. — LEMMA | CIKM 2024 | LVLM + external knowledge for misinfo | [Paper](https://arxiv.org/html/2402.11943v2) |
| 11 | Huang et al. — Text-Image De-contextualization | ICASSP 2022 | CLIP vs VinVL for OOC detection | [Paper](https://ieeexplore.ieee.org/abstract/document/9746193/) |
| 12 | Jiang & Wang — LVLMs as Classifiers | arXiv 2024 | In-context multimodal fake news detection | [Paper](https://arxiv.org/abs/2407.12879) |

**Total with existing 10 = 22 references** (well above the 15 minimum).
