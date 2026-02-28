# Mahim's Midterm Plan — Text Modality (Text Evidence)

> **Role**: Evaluate text evidence for claim verification using RoBERTa
> **Model**: RoBERTa-base — fine-tuned for Natural Language Inference (claim vs text evidence)
> **Question**: "How useful is text evidence alone for verifying textual claims?"

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
  - Compute dataset statistics (class distribution, text lengths, vocab size, etc.)

- [ ] **Sections 2–4 of the report** (build on first submission — shared content, adapted per person)
  - Section 2 (Introduction): mostly reusable, add contributions paragraph specific to your modality
  - Section 3 (Related Works): expand to 15+ references, add NLI/textual entailment/text verification literature
  - Section 4 (Dataset): write precise technical description with stats

### A2. Mahim's Individual Work

#### Code & Experiments

- [ ] **RoBERTa NLI Baseline Implementation**
  - Load pre-trained RoBERTa-base (or start from `roberta-large-mnli` which is already NLI-tuned)
  - Input format: `[CLS] claim [SEP] text_evidence [SEP]`
  - Fine-tune on WebQA-Adv for 3-way classification (True / False / Unverifiable)
  - This is standard NLI: does the text evidence entail, contradict, or neither support the claim?

- [ ] **Training Details**
  - Optimizer: AdamW
  - Learning rate: 2e-5 with linear warmup
  - Batch size: 16 or 32
  - Epochs: 3-5 (typical for fine-tuning)
  - Max sequence length: 512 tokens (claim + evidence)
  - Loss: Cross-entropy with class weights (handle imbalance)

- [ ] **Optional: Additional Text Baselines**
  - BERT-base (compare to RoBERTa to justify model choice)
  - TF-IDF + Logistic Regression (simple baseline to show deep learning advantage)
  - DeBERTa-v3 (stronger NLI model, if time permits)

- [ ] **Ablation / Analysis**
  - Break down performance by misinformation type:
    - How does RoBERTa perform on text contradictions (claim negated/altered)?
    - How does RoBERTa perform on visual contradictions (text is fine, image is wrong)?
    - How does RoBERTa perform on temporal mismatches?
  - Qualitative examples:
    - Success cases: where RoBERTa detects claim contradicts text evidence
    - Failure cases: where text evidence is consistent but image is the problem
  - Confusion matrix (3-way: True/False/Unverifiable)
  - Attention visualization: what tokens does RoBERTa attend to when making decisions?

- [ ] **Generate Figures**
  - Confusion matrix
  - Per-class performance bar chart (Macro F1, accuracy per class)
  - Training/validation loss curves
  - Attention heatmaps on example claim-evidence pairs
  - Qualitative examples: 2-3 success cases, 2-3 failure cases
  - Optional: confidence score distributions for correct vs incorrect predictions

#### Report Writing (Individual Submission)

- [ ] **Section 1 — Abstract**
  - Summarize: problem → text modality analysis with RoBERTa → key results → takeaway (text evidence catches some but misses image-based deception)

- [ ] **Section 5.1 — Unimodal Models & Methods (1–1.5 pages)**
  - Explain RoBERTa architecture in detail:
    - BERT foundation: masked language modeling, next sentence prediction
    - RoBERTa improvements: dynamic masking, larger batches, no NSP, more data
    - Pre-training on 160GB of text data
    - Fine-tuning for NLI: how [CLS] token representation is used for classification
  - Frame as Natural Language Inference:
    - Premise = text evidence, Hypothesis = claim
    - Entailment → True, Contradiction → False, Neutral → Unverifiable
  - Why RoBERTa is appropriate:
    - Strong NLI performance (MNLI, SNLI benchmarks)
    - Well-suited for semantic understanding of claims vs evidence
    - Pre-training data relevant to web-sourced factual text
  - If additional baselines were run, describe them briefly

- [ ] **Section 5.2 — Experiments & Evaluation (0.5 page)**
  - Train/val/test split sizes and strategy
  - Hardware: GPU type, memory
  - Software: PyTorch, HuggingFace transformers version
  - Hyperparameters: LR, batch size, epochs, warmup, max seq length, class weights
  - Metrics: Macro F1, Accuracy, per-class F1
  - Regularization: dropout, weight decay
  - Data: text evidence + claim only (no images used)

- [ ] **Section 5.3 — Results (1 page)**
  - Summary table: RoBERTa performance across metrics
  - If additional baselines exist: comparison table (BERT vs RoBERTa vs TF-IDF)
  - Confusion matrix
  - Breakdown by misinformation type
  - Training curves (loss, accuracy over epochs)
  - Attention heatmap examples showing what RoBERTa focuses on
  - Qualitative success/failure examples

- [ ] **Section 5.4 — Discussion**
  - RoBERTa catches text contradictions well (claim says X, evidence says Y)
  - RoBERTa fails on visual contradictions (text evidence is consistent, problem is the image)
  - RoBERTa fails on temporal mismatches (text doesn't capture time inconsistency)
  - NLI framing is natural for text-only verification but inherently limited
  - Key insight: text evidence alone misses an entire category of deception
  - Discuss: how does this complement Devang's CLIP results? Where do they overlap/differ?

- [ ] **Section 6 — Updated Research Vision (0.5 page)**
  - Text evidence catches text-based deception but is blind to image-based deception
  - Image evidence (Devang's results) likely catches the opposite
  - Complementarity is clear → motivates cross-modal fusion (CMVN)
  - Updated plan: combine RoBERTa text features with CLIP image features via cross-attention
  - Concrete next steps: implement Stage 2 contradiction detector, joint training

- [ ] **References**: Expand to 15+ (add NLI papers, RoBERTa variants, textual entailment, fact-checking)

- [ ] **Appendix**: Annotated bibliography (graded), extended results, hyperparameter search

#### Presentation (20 min, individual)

- [ ] **Slides Structure** (~20 slides for 20 min):
  1. Title slide (1 min)
  2. Motivation: why multimodal misinformation matters (2 min)
  3. Dataset & Data Story: show example claims + text evidence pairs, label distribution (3 min) — **high grading weight**
  4. Related Works: cluster into (1) NLI/textual entailment, (2) text-based fact-checking, (3) gap (2 min)
  5. RoBERTa Model & Setup: architecture figure, NLI framing, fine-tuning details (3 min)
  6. Results & Analysis: lead with best figure, confusion matrix, breakdown by type, attention heatmaps, qualitative examples (5 min) — **highest grading weight**
  7. Research Vision & Next Steps: what I learned, why multimodal is needed (3 min)
  8. Q&A slide: summary table visible (1 min)

- [ ] **Prepare for Q&A defense questions**:
  - "Why RoBERTa and not DeBERTa or a larger model?"
  - "How does your NLI framing differ from standard NLI benchmarks?"
  - "What if you increased the context window to include more evidence?"
  - "How do your results compare to Devang's CLIP results?"
  - "Could text evidence alone ever catch visual contradictions?"

---

## Part B: Timeline

| When | Task | Deliverable |
|------|------|-------------|
| **Week 1** | Download WebQA, build adversarial augmentation pipeline (with Devang) | WebQA-Adv dataset with splits |
| **Week 1** | Implement RoBERTa fine-tuning pipeline (data loading, tokenization, training loop) | Working training code |
| **Week 2** | Train RoBERTa on WebQA-Adv, tune hyperparameters | Trained model + metrics |
| **Week 2** | Run ablations: per-type breakdown, attention analysis, qualitative examples | Analysis notebooks |
| **Week 3** | Write Sections 5.1–5.4 (experiments, results, discussion) | Draft of core sections |
| **Week 3** | Write Sections 1, 6, expand Section 3 to 15+ refs | Complete report draft |
| **Week 3** | Build presentation slides | Slide deck draft |
| **Week 4** | Revise report, finalize figures, write annotated bibliography | Final report |
| **Week 4** | Practice presentation (time it!), prepare Q&A answers | Final presentation |

---

## Part C: Key Technical Details

### RoBERTa NLI Pipeline

```python
# Pseudocode
from transformers import RobertaForSequenceClassification, RobertaTokenizer

model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=3  # True, False, Unverifiable
)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

for claim, text_evidence, label in dataset:
    # NLI format: premise (evidence) + hypothesis (claim)
    inputs = tokenizer(
        claim, text_evidence,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    outputs = model(**inputs)
    logits = outputs.logits  # [batch, 3]
    prediction = logits.argmax(dim=-1)  # 0=True, 1=False, 2=Unverifiable
```

### Alternative Starting Point: Pre-trained NLI Model

```python
# Start from a model already fine-tuned on MNLI → less training needed
from transformers import AutoModelForSequenceClassification

# Option A: RoBERTa already trained on MNLI
model = AutoModelForSequenceClassification.from_pretrained(
    "FacebookAI/roberta-large-mnli"  # Already maps to entailment/contradiction/neutral
)

# Option B: DeBERTa trained on MNLI+FEVER+ANLI (strongest NLI model)
model = AutoModelForSequenceClassification.from_pretrained(
    "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
)
```

### Expected Results Pattern

| Misinformation Type | Expected RoBERTa Performance | Why |
|---|---|---|
| True claims | Good (high confidence) | Evidence supports claim — straightforward entailment |
| Text contradiction | **Good (detectable)** | Claim says X, evidence says Y — textual contradiction is clear |
| Visual contradiction | **Poor (miss)** | Text evidence is fine, problem is in the image — RoBERTa can't see images |
| Temporal mismatch | Poor (miss) | Text doesn't encode temporal inconsistency well |
| Ambiguous | Moderate | May predict Unverifiable if evidence is weak |

### Metrics to Report

- Macro F1 (primary — handles class imbalance)
- Per-class F1 (True / False / Unverifiable)
- Overall Accuracy
- Confusion Matrix
- Training/Validation Loss Curves
- Attention Weights Visualization

---

## Part D: Literature Grounding

### Why RoBERTa for Textual Fact Verification — What the Literature Says

**1. Fine-tuned Transformers outperform LLMs for fact-checking.**
Setty (SIGIR 2024) in ["Surprising Efficacy of Fine-Tuned Transformers for Fact-Checking over Larger Language Models"](https://doi.org/10.1145/3626772.3661361) showed that fine-tuned RoBERTa/XLM-RoBERTa **surpasses GPT-4 and GPT-3.5** on claim verification tasks (23 citations). Key finding: smaller, task-specific models beat general-purpose LLMs for NLI-based fact-checking. This directly justifies choosing RoBERTa over prompting a large LLM.

**2. Fact verification as NLI is a well-established framing.**
The FEVER dataset ([Thorne et al., NAACL 2018](https://aclanthology.org/N18-1074/)) established the standard: given a claim and evidence, classify as Supported/Refuted/NotEnoughInfo. This maps exactly to your True/False/Unverifiable labels. FEVER contains 185,445 claims from Wikipedia — similar domain to WebQA.

**3. RoBERTa is a strong NLI baseline — but DeBERTa is stronger.**
- **RoBERTa-large on MNLI**: ~90.2% accuracy
- **DeBERTa-v3-large on MNLI**: ~91.1% accuracy (+0.9%)
- DeBERTa uses disentangled attention (separate content and position vectors) and enhanced mask decoder
- Pre-trained [DeBERTa-v3-large-mnli-fever-anli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli) is trained on MNLI + FEVER + ANLI — perfect for fact verification
- Consider running DeBERTa as an **additional baseline** to compare against RoBERTa

**4. NLI models have been specifically tested for claim verification.**
- **Košprdić et al. (2024)**: ["Scientific Claim Verification with Fine-Tuned NLI Models"](https://www.scitepress.org/Papers/2024/129000/129000.pdf) — fine-tuned RoBERTa for scientific claim verification, showing NLI framing works well for domain-specific fact-checking.
- **HealthFC** ([Vladika et al., LREC 2024](https://aclanthology.org/2024.lrec-main.709/), 34 citations): Verified health claims with evidence-based fact-checking using XLM-RoBERTa. Shows RoBERTa generalizes across domains.
- **FENICE** ([Scirè et al., ACL Findings 2024](https://aclanthology.org/2024.findings-acl.841/), 27 citations): Uses RoBERTa for factuality evaluation of summarization based on NLI and claim extraction.

**5. Pre-trained NLI checkpoints exist specifically for fact-checking.**
- [`FacebookAI/roberta-large-mnli`](https://huggingface.co/FacebookAI/roberta-large-mnli): RoBERTa fine-tuned on Multi-NLI (392K premise-hypothesis pairs). Good starting point.
- [`cross-encoder/nli-roberta-base`](https://huggingface.co/cross-encoder/nli-roberta-base): Cross-encoder variant for pairwise scoring.
- [`ynie/roberta-large-snli_mnli_fever_anli`](https://dataloop.ai/library/model/ynie_roberta-large-snli_mnli_fever_anli_r1_r2_r3-nli/): Trained on SNLI + MNLI + FEVER + ANLI — most robust NLI checkpoint.
- **Strategy**: Start from one of these checkpoints (already knows NLI), then fine-tune on WebQA-Adv. This is transfer learning: NLI pre-training → domain-specific fact-checking.

**6. Fontana et al. (2025) confirm RoBERTa remains competitive.**
["Evaluating Open-Source LLMs for Automated Fact-Checking"](https://arxiv.org/abs/2503.05565) found that **fine-tuned RoBERTa shows great performance** for fact-checking, often matching or outperforming much larger open-source LLMs. Reinforces the choice of RoBERTa as a strong baseline.

**7. Hybrid approaches show NLI's role in pipelines.**
- **Rosenbaum et al. (2025)**: ["Hybrid Fact-Checking that Integrates KGs, LLMs, and Retrieval Agents"](https://aclanthology.org/2025.winlp-main.19/) uses DeBERTa-v3-MNLI as the NLI component in a hybrid pipeline. Cast fact verification as NLI, then combine with knowledge graph evidence and retrieval.
- **TIFIN at CheckThat! 2025** ([CEUR 2025](https://ceur-ws.org/Vol-4038/paper_74.pdf)): Multi-lingual NLI-based fact-checking using the MNLI framework. Notes DeBERTa's architectural advantages over RoBERTa for this specific task.

**8. Key NLI benchmarks for context.**
| Benchmark | Size | Labels | Relevance |
|---|---|---|---|
| **MNLI** (Williams et al., 2018) | 433K pairs | Entailment/Contradiction/Neutral | General NLI training |
| **SNLI** (Bowman et al., 2015) | 570K pairs | Entailment/Contradiction/Neutral | Foundational NLI dataset |
| **FEVER** (Thorne et al., 2018) | 185K claims | Supported/Refuted/NEI | Directly fact-checking |
| **ANLI** (Nie et al., 2020) | 163K pairs | Entailment/Contradiction/Neutral | Adversarial NLI — harder examples |

These provide context for why RoBERTa works: it's been validated on hundreds of thousands of NLI pairs.

### WebQA-Specific Literature
- **WebQA** (Chang et al., CVPR 2022, [216 citations](http://openaccess.thecvf.com/content/CVPR2022/html/Chang_WebQA_Multihop_and_Multimodal_QA_CVPR_2022_paper.html)): The base dataset. Requires multi-hop reasoning over text AND images.
- **RAMQA** (Bai et al., NAACL Findings 2025, [6 citations](https://aclanthology.org/2025.findings-naacl.60/)): Retrieval-augmented multimodal QA framework tested on WebQA. Shows evidence retrieval quality matters.
- **Progressive Evidence Refinement** (Yang et al., 2023, [7 citations](https://arxiv.org/abs/2310.09696)): Tested on WebQA, improving evidence retrieval for multimodal QA.

---

## Part E: References to Add (with Links)

Papers to cite (in addition to existing 10 in bib):

| # | Paper | Venue | Why Cite | Link |
|---|---|---|---|---|
| 1 | Setty — Fine-Tuned Transformers for Fact-Checking | SIGIR 2024 | RoBERTa beats GPT-4 for fact-checking | [Paper](https://doi.org/10.1145/3626772.3661361) |
| 2 | Thorne et al. — FEVER | NAACL 2018 | Standard fact verification benchmark & NLI framing | [Paper](https://aclanthology.org/N18-1074/) |
| 3 | Williams et al. — MNLI | NAACL 2018 | NLI benchmark RoBERTa was trained on | [HuggingFace](https://huggingface.co/FacebookAI/roberta-large-mnli) |
| 4 | Bowman et al. — SNLI | EMNLP 2015 | Foundational NLI dataset | — |
| 5 | He et al. — DeBERTa | ICLR 2021 | Stronger NLI model for comparison | [GitHub](https://github.com/microsoft/DeBERTa) |
| 6 | Devlin et al. — BERT | NAACL 2019 | Foundation model RoBERTa builds on | — |
| 7 | Vladika et al. — HealthFC | LREC 2024 | RoBERTa for domain-specific fact-checking | [Paper](https://aclanthology.org/2024.lrec-main.709/) |
| 8 | Scirè et al. — FENICE | ACL Findings 2024 | RoBERTa for NLI-based factuality evaluation | [Paper](https://aclanthology.org/2024.findings-acl.841/) |
| 9 | Fontana et al. — LLMs for Fact-Checking | arXiv 2025 | RoBERTa competitive with larger LLMs | [Paper](https://arxiv.org/abs/2503.05565) |
| 10 | Qi et al. — SNIFFER | CVPR 2024 | State-of-art multimodal misinfo detection | [Paper](https://openaccess.thecvf.com/content/CVPR2024/html/Qi_SNIFFER_Multimodal_Large_Language_Model_for_Explainable_Out-of-Context_Misinformation_Detection_CVPR_2024_paper.html) |
| 11 | Papadopoulos et al. — VERITE | MIR 2023 | Unimodal bias in benchmarks | [GitHub](https://github.com/stevejpapad/image-text-verification) |
| 12 | Papadopoulos et al. — Similarity over Factuality | WACV 2025 | Critical evaluation of OOC methods | [Paper](https://openaccess.thecvf.com/content/WACV2025/papers/Papadopoulos_Similarity_over_Factuality_Are_we_Making_Progress_on_Multimodal_Out-of-Context_WACV_2025_paper.pdf) |

**Total with existing 10 = 22 references** (well above the 15 minimum).
