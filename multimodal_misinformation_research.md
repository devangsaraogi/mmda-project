# Multimodal Misinformation Detection: Comprehensive Research Landscape

> **Purpose**: Reference document for the MMD course project — *Cross-Modal Verification Network for Multimodal Fact-Checking*
> **Authors**: Devang Saraogi (dsaraog), Mahim Dashora (mdashor)
> **Last Updated**: 2026-02-26

---

## Table of Contents

1. [What is Multimodal Misinformation?](#1-what-is-multimodal-misinformation)
2. [Types of Multimodal Misinformation](#2-types-of-multimodal-misinformation)
3. [Major Research Directions](#3-major-research-directions)
4. [Key Datasets](#4-key-datasets)
5. [Approaches & Architectures](#5-approaches--architectures)
6. [Recent Advances (2024-2025)](#6-recent-advances-2024-2025)
7. [Key Surveys to Read](#7-key-surveys-to-read)
8. [Open Challenges](#8-open-challenges)
9. [How This Relates to Our Project (CMVN)](#9-how-this-relates-to-our-project-cmvn)

---

## 1. What is Multimodal Misinformation?

Misinformation where the **deception lies in the combination of modalities** (image + text), not in either modality alone. A real photo paired with a false caption, or a true claim paired with an unrelated image — each piece is authentic individually, but the pairing is misleading.

**Why unimodal detection fails**: A text classifier sees a grammatically correct, plausible sentence. An image classifier sees an unmanipulated photograph. Neither can detect the lie because the lie is in the *relationship* between them.

---

## 2. Types of Multimodal Misinformation

| Type | Description | Example | Detection Difficulty |
|------|-------------|---------|---------------------|
| **Out-of-Context (OOC)** | Real image reused with false/misleading caption | Photo of a 2015 flood captioned as "2024 hurricane damage" | Hard — both modalities are authentic |
| **Cheapfakes** | Simple manipulations: cropping, splicing, re-captioning (no AI) | Cropped photo removing context, speed-altered video | Medium — no pixel-level artifacts |
| **Deepfakes** | AI-generated/manipulated images or video | Face-swapped video of a politician | Medium — detectable via forensics |
| **Text Contradiction** | Correct image, but text claim is factually wrong | Photo of steel bridge + caption "this wooden bridge" | Easier — text evidence often contradicts |
| **Temporal Mismatch** | Authentic content from wrong time period | Old photo presented as current event | Hard — requires temporal reasoning |
| **Entity Swap** | Correct description, wrong entity | Photo of City A labeled as City B | Hard — requires world knowledge |

---

## 3. Major Research Directions

### 3.1 Out-of-Context (OOC) Detection

The most studied sub-problem. The image and text are both real, but their pairing is misleading.

**Key papers:**
- **COSMOS** (Aneja et al., 2023) — Self-supervised contrastive learning to detect OOC image misuse. Learns from image-caption co-occurrences without explicit labels.
  [Project Page](https://shivangi-aneja.github.io/projects/cosmos/) | [ResearchGate](https://www.researchgate.net/publication/371917861)

- **NewsCLIPpings** (Luo et al., EMNLP 2021) — 988K image-caption pairs auto-generated from VisualNews. Uses CLIP, scene matching, and person matching to create challenging OOC samples.
  [ACL Anthology](https://aclanthology.org/2021.emnlp-main.545/)

- **SNIFFER** (Qi et al., CVPR 2024) — Instruction-tuned MLLM for *explainable* OOC detection. Two-stage: (1) internal checking (image-text consistency), (2) external checking (retrieved evidence). Outperforms GPT-4V by 11%.
  [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Qi_SNIFFER_Multimodal_Large_Language_Model_for_Explainable_Out-of-Context_Misinformation_Detection_CVPR_2024_paper.html) | [GitHub](https://github.com/MischaQI/Sniffer) | [arXiv](https://arxiv.org/abs/2403.03170)

- **"Similarity over Factuality"** (Papadopoulos et al., WACV 2025) — Critical analysis showing many OOC detectors exploit unimodal shortcuts (similarity) rather than actually reasoning about factuality.
  [Paper PDF](https://openaccess.thecvf.com/content/WACV2025/papers/Papadopoulos_Similarity_over_Factuality_Are_we_Making_Progress_on_Multimodal_Out-of-Context_WACV_2025_paper.pdf)

### 3.2 Cross-Modal Fact Checking / Claim Verification

Given a claim + multimodal evidence, predict a verdict (True / False / Unverifiable) with evidence attribution.

**Key papers:**
- **Abdelnabi et al. (CVPR 2022)** — Open-domain OOC detection using external evidence retrieval (reverse image search + text queries). Shows evidence retrieval matters but uses simple feature concatenation.
  [CVPR Paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Abdelnabi_Open-Domain_Content-Based_Multi-Modal_Fact-Checking_of_Out-of-Context_Images_via_Online_Resources_CVPR_2022_paper.pdf)

- **LEMMA** (Fan et al., CIKM 2024) — LVLM-Enhanced Multimodal Misinformation Detection with External Knowledge Augmentation. Parallel text and image search for evidence, reasoning-aware multi-query generation. Improves accuracy by 9-13% over baselines.
  [arXiv](https://arxiv.org/html/2402.11943v2)

- **Tahmasebi et al. (CIKM 2024)** — Tests LVLMs (GPT-4V, LLaVA, InstructBLIP) for zero-shot multimodal fact verification. Proposes re-ranking for evidence retrieval. Competitive but black-box.
  [ACM DL](https://doi.org/10.1145/3627673.3679826)

- **FVQA 2.0** (2023) — Introduces adversarial samples into fact-based VQA to test robustness.
  [ACL Anthology](https://aclanthology.org/2023.findings-eacl.11.pdf)

- **CMIE** (2025) — Combining MLLM Insights with External Evidence for Explainable OOC Misinformation Detection.
  [arXiv](https://arxiv.org/html/2505.23449)

### 3.3 Image-Text Inconsistency Localization

Beyond binary detection — *where exactly* is the inconsistency? Which words and which image regions conflict?

**Key paper:**
- **D-TIIL** (Huang et al., ICLR 2024) — Uses text-to-image diffusion models as "omniscient" agents. Outputs: binary mask highlighting inconsistent image regions + per-word consistency scores. Introduced the TIIL dataset (14K pairs).
  [arXiv](https://arxiv.org/abs/2404.18033) | [GitHub](https://github.com/Mingzhen-Huang/D-TIIL) | [OpenReview](https://openreview.net/forum?id=Ny150AblPu) | [Project Page](https://mingzhenhuang.com/projects/InconsisDet.html)

### 3.4 Cheapfake / Shallowfake Detection

Simple manipulations (re-contextualization, cropping, speed changes) that don't use AI but are far more prevalent than deepfakes.

- **MediaEval Grand Challenge on Detecting Cheapfakes** — Community benchmark for multimedia verification.
  [Challenge Page](https://detecting-cheapfakes.github.io/)

### 3.5 Entity Verification

Verifying whether named entities in text match what appears in the image.

- **Cross-Modal Entity Consistency** (2025) — Uses VLMs to verify entities across modalities.
  [Springer](https://link.springer.com/chapter/10.1007/978-3-031-88717-8_25)

---

## 4. Key Datasets

| Dataset | Year | Scale | What It Contains | Use Case | Link |
|---------|------|-------|-----------------|----------|------|
| **WebQA** | 2022 | 34K QA pairs, 390K images, 540K text | Multi-hop multimodal QA from web | Base for adversarial augmentation (our project) | [arXiv](https://arxiv.org/abs/2109.00590) / [CVPR](https://openaccess.thecvf.com/content/CVPR2022/papers/Chang_WebQA_Multihop_and_Multimodal_QA_CVPR_2022_paper.pdf) |
| **NewsCLIPpings** | 2021 | 988K pairs | Auto-generated OOC news image-caption pairs | OOC detection training | [ACL Anthology](https://aclanthology.org/2021.emnlp-main.545/) |
| **COSMOS** | 2021 | ~3K annotated | Real-world OOC examples from news, self-supervised | OOC detection with limited labels | [Project Page](https://shivangi-aneja.github.io/projects/cosmos/) |
| **VERITE** | 2023 | 1,000 pairs (balanced) | Real-world misinfo from Snopes/Reuters, modality-balanced | Bias-free evaluation | [GitHub](https://github.com/stevejpapad/image-text-verification) / [ResearchGate](https://www.researchgate.net/publication/377238125) |
| **Fakeddit** | 2020 | ~1M samples | Reddit posts with fine-grained labels (2/3/6-way) | Large-scale training | — |
| **FakeNewsNet** (GossipCop + PolitiFact) | 2019 | ~23K articles | Celebrity gossip + political claims with verdicts | Social media misinfo | — |
| **TIIL** | 2024 | 14K pairs | Word-level + region-level inconsistency annotations | Fine-grained localization | [arXiv](https://arxiv.org/abs/2404.18033) |
| **MMFakeBench** | 2024 | Large-scale | Mixed forgery types (cheapfakes, deepfakes, OOC) | Comprehensive LVLM evaluation | [arXiv](https://arxiv.org/html/2406.08772) |
| **Fauxtography** | 2019 | 19K tweets | Verified image-text mismatches from Twitter | Real-world OOC | [ACL Anthology](https://aclanthology.org/D19-1216/) |

### Critical Issue: Unimodal Bias

The [VERITE benchmark](https://github.com/stevejpapad/image-text-verification) revealed that in many datasets (COSMOS, VMU-Twitter), a **text-only or image-only** model can outperform multimodal models. This means the dataset doesn't truly require cross-modal reasoning — models are exploiting unimodal shortcuts. This is a key consideration when designing our WebQA-Adv dataset.

---

## 5. Approaches & Architectures

### 5.1 Feature Extraction + Fusion + Classification (Classical Pipeline)

The dominant paradigm with three stages:
1. **Unimodal encoders**: BERT/RoBERTa for text, ResNet/ViT for images
2. **Fusion**: Concatenation (early), cross-attention (deep), or decision-level (late)
3. **Classification head**: Binary or multi-class

**Fusion strategies:**
- **Early fusion**: Concatenate features before classification
- **Late fusion**: Separate classifiers per modality, combine decisions
- **Deep fusion**: Cross-attention, co-attention, or Transformer-based interaction

Typical accuracy: 85-90% on Gossipcop/Fakeddit benchmarks.

### 5.2 CLIP-Based Methods

CLIP (Contrastive Language-Image Pretraining) has become foundational for this space.

| Method | How It Uses CLIP | Key Idea | Link |
|--------|-----------------|----------|------|
| **FND-CLIP** | CLIP encoders + ResNet + BERT; concatenates features weighted by cross-modal similarity | Multi-encoder fusion with similarity weighting | [arXiv](https://arxiv.org/abs/2205.14304) |
| **SARD** | CLIP contrastive learning + heterogeneous information network | Combines semantic alignment with social context | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1319157824002490) |
| **SAMPLE** | CLIP features + normalized cross-modal similarity to adjust representation intensity | Similarity-aware prompt learning | [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0020025523010319) |
| **CLIP cosine similarity** (baseline) | Compute claim-image similarity, threshold for classification | Simplest approach — tests whether embedding proximity suffices | (standard baseline) |

**Limitation**: CLIP measures *similarity* but cannot reason about *contradictions*. A claim-image pair can be topically similar (high CLIP score) yet factually contradictory.

### 5.3 Cross-Modal Attention / Transformer Architectures

More sophisticated fusion that models *interactions* between modalities:

- **Cross-Attention Fusion**: Query one modality using the other as key/value. Two parallel attention layers (text→image, image→text).
- **MCOT** (Frontiers, 2024): Cross-modal attention + contrastive learning + optimal transport for distribution alignment.
  [Paper](https://www.frontiersin.org/journals/computer-science/articles/10.3389/fcomp.2024.1473457/full)
- **Multi-level Consistency**: Token-level (cross-attention), phrase-level (GNN), global-level (CLIP) — checks consistency at multiple granularities.
- **Hierarchical Cross-Modal Interaction Network**: Dual-enhanced co-attention + hierarchical mutual contrastive learning.
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231225009804)

### 5.4 Graph Neural Networks (GNNs)

Model social propagation patterns, user networks, and entity relationships as graphs.

**Why graphs?** Social networks are naturally graph-structured. Misinformation spreads through specific patterns (fast, wide, via certain user types) that GNNs can capture.

**Three main categories:**
1. **Knowledge-driven**: Link claims to knowledge graphs for verification
2. **Propagation-based**: Model how news spreads through retweet/share networks
3. **Heterogeneous social context**: Model users, posts, and interactions as heterogeneous graphs

**Key models:**
- **FANG** (Factual News Graph) — Models social interactions (following, posting, reposting) as graph edges
- **GCAN** (Graph-aware Co-Attention Network) — GRU + GNN for joint text/user representation learning
- **HAGNN** (Hierarchically Aggregated GNN) — 95.7% accuracy on Weibo rumor detection
- **GraMuFeN** — Graph-based multi-modal fusion combining text and image features

**Surveys:**
- [Fake News Detection Through Graph-based Neural Networks: A Survey](https://arxiv.org/abs/2307.12639) (2023)
- [Disinformation Detection Using GNNs: A Survey](https://link.springer.com/article/10.1007/s10462-024-10702-9) (2024)

### 5.5 Multimodal Large Language Models (MLLMs)

The newest wave — using instruction-tuned VLMs for both detection AND explanation.

| Model | Venue | Approach | Key Result | Link |
|-------|-------|----------|------------|------|
| **SNIFFER** | CVPR 2024 | Two-stage instruction tuning on InstructBLIP; internal + external checking | Outperforms GPT-4V by 11%; provides explanations | [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Qi_SNIFFER_Multimodal_Large_Language_Model_for_Explainable_Out-of-Context_Misinformation_Detection_CVPR_2024_paper.html) |
| **LEMMA** | CIKM 2024 | LVLM + parallel text/image evidence retrieval | +9-13% accuracy over baselines | [arXiv](https://arxiv.org/html/2402.11943v2) |
| **GPT-4V** (zero-shot) | Various | Direct prompting for verification | >80% accuracy but black-box | — |
| **CMIE** | 2025 | Combines MLLM insights with external evidence | Explainable OOC detection | [arXiv](https://arxiv.org/html/2505.23449) |
| **MIRAGE** | 2025 | Agentic framework — LLM invokes tools (reverse image search, web search, KB lookup) | Active evidence gathering | [arXiv](https://arxiv.org/pdf/2510.17590) |

### 5.6 Diffusion Model-Based

- **D-TIIL** (ICLR 2024): Uses text-to-image diffusion as an "omniscient" agent to localize inconsistencies at word and pixel level. Novel use of generative models for discriminative tasks.
  [arXiv](https://arxiv.org/abs/2404.18033)

### 5.7 Contrastive Learning Approaches

- **ERIC-FND**: Retrieves Wikipedia knowledge entities as external info, uses attention-based enhancement with contrastive learning.
  [arXiv](https://arxiv.org/abs/2503.03107)
- **Knowledge-Aware Multimodal Pre-training**: Incorporates knowledge graphs into pre-training.
  [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1566253524004937)
- **Event-driven Multi-View Learning** (ACL 2024): Models events as views for multimodal fake news detection.
  [ACL Anthology](https://aclanthology.org/2024.acl-long.316.pdf)

### 5.8 Causal / Counterfactual Methods

- **Counterfactual Multimodal Fact-Checking**: Uses causal intervention to isolate evidence that is *causally* related to verification, removing spurious correlations.
  [Springer](https://link.springer.com/chapter/10.1007/978-981-97-8620-6_40)

---

## 6. Recent Advances (2024-2025)

### Trend 1: MLLM-Powered Detection + Explainability
Shift from binary classification → detection with natural language explanations. SNIFFER, LEMMA, CMIE all produce human-readable rationales alongside predictions.

### Trend 2: Evidence Retrieval Integration
Modern systems actively retrieve external evidence rather than relying only on the input pair. This mirrors how human fact-checkers work.
- **Beyond Retrieval** (2025): Improving evidence *quality* not just quantity for LLM-based fact-checking. [arXiv](https://arxiv.org/html/2505.03135v2)
- **HiEAG** (2025): Evidence-Augmented Generation for OOC detection. [arXiv](https://arxiv.org/html/2511.14027)

### Trend 3: Addressing Unimodal Bias
VERITE and "Similarity over Factuality" exposed that many models exploit unimodal shortcuts. New benchmarks enforce modality balancing so models must truly reason cross-modally.

### Trend 4: Agentic / Tool-Augmented Systems
LLMs that can invoke tools (reverse image search, web search, knowledge bases) to actively investigate claims rather than passively classify them.
- **MIRAGE**: [arXiv](https://arxiv.org/pdf/2510.17590)
- **Multimodal Fact-Checking: An Agent-based Approach** (2025): [arXiv](https://arxiv.org/html/2512.22933)
- **Multimedia Verification Through Multi-Agent Deep Research MLLMs** (2025): [arXiv](https://arxiv.org/html/2507.04410)

### Trend 5: Fine-Grained Localization
Moving beyond "fake or real" to "which specific words and image regions are inconsistent" (D-TIIL, TIIL dataset).

### Trend 6: Knowledge Distillation from MLLMs
Distilling knowledge from large models (GPT-4V, etc.) into smaller, deployable models.

### Trend 7: Multilingual / Cross-Cultural Detection
Expanding beyond English-only datasets and models.
- [Explainable Multilingual and Multimodal Fake-News Detection](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1690616/full)

---

## 7. Key Surveys to Read

| Survey | Year | Focus | Link |
|--------|------|-------|------|
| **Multi-modal Misinformation Detection: Approaches, Challenges and Opportunities** | 2024 | Comprehensive — all approaches | [ACM Computing Surveys](https://dl.acm.org/doi/full/10.1145/3697349) / [arXiv](https://arxiv.org/pdf/2203.13883) |
| **Multi-modal Fake News Detection: A Comprehensive Survey on Deep Learning Technology** | 2025 | Deep learning focused | [Springer](https://link.springer.com/article/10.1007/s44443-025-00317-7) |
| **Disinformation Detection Using Graph Neural Networks: A Survey** | 2024 | GNN-specific | [Springer](https://link.springer.com/article/10.1007/s10462-024-10702-9) |
| **Fake News Detection Through Graph-based Neural Networks: A Survey** | 2023 | GNN methods taxonomy | [arXiv](https://arxiv.org/abs/2307.12639) |
| **LLM-for-Misinformation-Research** | Ongoing | Paper list of MLLM-based approaches | [GitHub](https://github.com/ICTMCG/LLM-for-misinformation-research) |
| **Automated Fact-Checking Resources** | Ongoing | Comprehensive resource list | [GitHub](https://github.com/Cartus/Automated-Fact-Checking-Resources) |
| **A Systematic Review of Multimodal Fake News Detection on Social Media Using Deep Learning** | 2025 | Systematic review | [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2590123025008291) |

---

## 8. Open Challenges

1. **Unimodal Bias**: Many benchmarks can be "solved" by text-only or image-only models — true cross-modal reasoning is rarely tested.
2. **Generalization**: Models trained on one domain (e.g., politics) fail on another (e.g., health).
3. **Real-Time Detection**: Most methods are too slow for social media scale.
4. **Explainability vs. Accuracy Trade-off**: Explainable models often sacrifice detection performance.
5. **Adversarial Robustness**: Misinformation creators can adapt to evade detectors.
6. **Temporal Dynamics**: Misinformation evolves; static models degrade.
7. **Fine-Grained Localization**: Beyond binary labels — which *specific* elements are inconsistent?
8. **Dataset Quality**: Many datasets have label noise, size limitations, or domain restrictions.

---

## 9. How This Relates to Our Project (CMVN)

### Where CMVN Fits in the Landscape

| Aspect | Existing Work | Our Approach (CMVN) |
|--------|--------------|---------------------|
| **Detection** | Similarity-based (CLIP) or black-box (MLLMs) | Explicit contradiction detector with interpretable consistency scores (s_CI, s_CT, s_IT) |
| **Evidence** | Some retrieve, most don't | Hybrid retrieval (CLIP + BM25) with cross-encoder re-ranking |
| **Explainability** | Binary classification (no explanation) OR MLLM (ungrounded explanations) | Flan-T5 explanations grounded in cited evidence + faithfulness metric |
| **Dataset** | Existing benchmarks have unimodal bias issues | WebQA-Adv designed with controlled adversarial manipulations |
| **Training** | Pipeline (separate stages) | Joint multi-task loss (verdict + evidence + explanation) |

### For Midterm: Unimodal Baselines

The midterm focuses on what **each modality alone** can catch:
- **Devang (Text)**: RoBERTa on claim + text evidence → catches text contradictions (25% of dataset) but misses visual contradictions (20%), temporal mismatches (10%)
- **Mahim (Image)**: ViT/CLIP on claim + image evidence → catches visual contradictions but misses text-only deception

The *gap* between unimodal results motivates the multimodal CMVN for the final project.

### Papers Most Relevant to Our Work

1. **SNIFFER** (CVPR 2024) — closest to our explainability goal
2. **LEMMA** (CIKM 2024) — closest to our evidence retrieval approach
3. **D-TIIL** (ICLR 2024) — inspiration for localization of inconsistencies
4. **VERITE** (2023) — methodology for avoiding unimodal bias in our dataset
5. **Abdelnabi et al.** (CVPR 2022) — external evidence retrieval for OOC detection
6. **"Similarity over Factuality"** (WACV 2025) — cautionary tale for evaluation

---

## References (BibTeX-ready)

Papers already in our `references.bib`:
- Chang et al. (2022) — WebQA
- Zlatkova et al. (2019) — Fauxtography
- Abdelnabi et al. (2022) — Open-domain OOC
- Tahmasebi et al. (2024) — LVLM verification
- Radford et al. (2021) — CLIP
- Li et al. (2023) — BLIP-2
- Liu et al. (2019) — RoBERTa
- Dosovitskiy et al. (2021) — ViT
- Chung et al. (2022) — Flan-T5
- Zhang et al. (2020) — BERTScore

**Papers to add for midterm (need 15+ total):**
- Qi et al. (2024) — SNIFFER
- Luo et al. (2021) — NewsCLIPpings
- Aneja et al. (2023) — COSMOS
- Papadopoulos et al. (2023) — VERITE
- Papadopoulos et al. (2025) — Similarity over Factuality
- Huang et al. (2024) — D-TIIL
- Fan et al. (2024) — LEMMA
- Nakamura et al. (2020) — Fakeddit
