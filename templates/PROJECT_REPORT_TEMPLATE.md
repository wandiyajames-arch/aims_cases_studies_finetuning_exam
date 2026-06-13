# Real LLM Deployment Project Report

## 1. Problem Definition
- **Use case:** A conversational AI assistant and translation engine specialized in the Wolof language for daily interactions, cross-lingual translations, and standardized orthography correction.
- **Users:** Wolof speakers, Senegalese students, and linguistic researchers seeking an accessible, low-resource language interface that adheres to standard academic conventions.
- **Fine-tuning Rationale:** Base foundation models (like Qwen) possess limited exposure to Wolof during pre-training. Parameter-Efficient Fine-Tuning (PEFT) via LoRA is required to adapt the model's localized vocabulary, align its conversational formatting (ChatML syntax), and enforce correct Wolof orthographic rules.
- **Model Efficiency:** The `Qwen3-0.6B` architecture was selected for its robust base performance and microscopic neural footprint, allowing for rapid iteration of the end-to-end data segregation, assistant-only masking, LoRA merging, and Gradio deployment pipeline.

## 2. Data Preparation
| Source | Examples | Task type | Cleaning/filtering | Category |
|--------|----------|-----------|--------------------|----------|
| **Aya** | 1,500 | Translation | Filtered for Wolof-English pairs; pruned incomplete strings. | Translation |
| **Soynade** | 1,000 | Orthography | Standardized UTF-8 encoding for special characters (ë, ñ, ŋ, ó). | Educational |
| **Synthetic** | 1,433 | Instruction | Deduplicated instructions; removed recursive conversational loops. | General Chat |

**Chat Formatting:** Raw data was transformed into the ChatML schema using `src/download_datasets.py`, mapping data into explicit `system`, `user`, and `assistant` role-based dictionaries to enforce strict multi-turn conversational boundaries.

## 3. Splitting Strategy
Data splits were executed deterministically via a set random seed before tokenization.
- **Train (90%):** 3,539 examples
- **Validation (5%):** 197 examples
- **Evaluation/Test (5%):** 197 examples

**Leakage Mitigation:** Splitting was performed independently for each source family to ensure no overlapping conversational contexts existed between training and evaluation subsets, providing an unbiased measure of zero-shot generalization.

## 4. Training Methodology
- **Base Model:** `Qwen/Qwen3-0.6B`.
- **LoRA Configuration:** Rank ($r=16$), Alpha ($\alpha=32$), targeting attention modules (`q_proj`, `v_proj`).
- **Assistant-Only Loss:** Cross-entropy loss was calculated exclusively on assistant-generated tokens.
- **Label Masking:** System, user, and padding tokens were set to `-100`, effectively masking them from gradient updates. This forces the model to optimize its weights strictly for *response generation* rather than prompt memorization.
- **Monitoring:** Convergence was achieved at an evaluation loss of `0.3781`.

## 5. Evaluation
The model was assessed on the held-out test split (`eval_all.jsonl`).

| Metric | Value |
|--------|-------|
| Final Val Loss | 0.3781 |
| Exact Match (EM) | 14.2% |
| Token F1 | 69.5% |
| BLEU | 26.1 |
| ROUGE-L | 49.8 |

### Qualitative Examples
- **Prompt:** *Na nga def?* → **Output:** *Maa ngi fi rekk, jàmm rekk.* (Culturally aligned greeting).
- **Prompt:** *Corrige: Loolou dafa oub sama bopp.* → **Output:** *Loolu dafa ub sama bopp.* (Successful orthographic normalization).

### Limitations
1. **Vocabulary Hallucination:** Minor vocabulary drift when processing unseen, highly specific English nouns.
2. **Looping:** Vulnerability to recursive sequences, successfully mitigated by upgrading `repetition_penalty` to `1.3` and utilizing `top_p` sampling.

## 6. Deployment
- **Hugging Face Hub:** [wandiya39/qwen3-0.6b-wolof-lora](https://huggingface.co/wandiya39/qwen3-0.6b-wolof-lora)
- **Space:** [Wolof AI Assistant](https://huggingface.co/spaces/wandiya39/Wolof-AI-Assistant)
- **Framework:** Custom Gradio `gr.Blocks` architecture featuring an industry-grade, dual-column Translation Engine UI, complete with dynamic CSS styling, translation mode toggling, and robust state management.

## 7. Risks
- **Prompt Injection:** Susceptibility to adversarial overrides.
- **Reasoning Leakage:** Internal base-model `<think>` tags require advanced UI-layer regex stripping.

## 8. Improvements & State Machine Justification
The original implementation was overhauled with a more efficient UI-layer session state, explicit prompt orchestration, and robust dependency handling to avoid inference latency and container crashes. 
- **Prompt Orchestration:** Transitioned from generic chat prompting to strict, task-bound system orchestration (e.g., *"You are a professional translator... Do not repeat the input."*) to prevent the model from mirroring inputs.
- **Output Sanitization:** Enhanced regex filtering (`re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL | re.IGNORECASE)`) implemented to aggressively strip multiline internal reasoning tokens.
- **Inference Stability:** Upgraded generation parameters (`temperature=0.3`, `repetition_penalty=1.3`, `top_p=0.9`) to strictly enforce translation accuracy and definitively eliminate recursive conversational looping.
- **Runtime Dependency Patching:** Injected an emergency Python "monkey patch" into `app.py` to seamlessly bypass a breaking API deprecation (`use_auth_token`) causing conflicts between `peft` and `huggingface_hub` on Gradio 6.0 servers.

## 9. Individual Technical Notes

### Student 1: WANDIYA James
- **Contribution:** Led the deployment pipeline, orchestrated the custom Gradio `gr.Blocks` UI engineering, and managed complex backend error handling.
- **Technical Choice:** Implemented prompt orchestration for deterministic translation behavior and engineered a runtime monkey patch to resolve fatal dependency crashes.
- **Problem Diagnosed:** API parameter deprecations causing container build failures, and input mirroring during inference. **Verification:** Successfully patched `hf_hub_download` at runtime and verified outputs via strict ChatML system boundaries.
- **Next Improvement:** Scale architecture to 7B parameters.

### Student 2: WADE Ndeye Khady
- **Contribution:** Managed data methodology, string normalization, and deterministic splitting.
- **Technical Choice:** Implemented a stratified random seed initialization across source groups.
- **Problem Diagnosed:** Character corruption with special Wolof vowels. **Verification:** Verified UTF-8 normalization.
- **Next Improvement:** Introduce programmatic linguistic balancing.

### Student 3: SIGEI Charlotte
- **Contribution:** Managed LoRA masking matrices and training profiles.
- **Technical Choice:** Selected Rank-16/Alpha-32 configuration for adaptation.
- **Problem Diagnosed:** Validation loss volatility in Epoch 1. **Verification:** Traced leaking gradients in TensorBoard; applied `-100` masking.
- **Next Improvement:** Integrate dynamic learning rate decay.

### Student 4: GAOLATLHE Angelah
- **Contribution:** Engineered the evaluation suite.
- **Technical Choice:** Combined BLEU with ROUGE-L to balance fluency vs. orthography.
- **Problem Diagnosed:** Over-penalization of valid synonyms. **Verification:** Human-graded qualitative tracking.
- **Next Improvement:** Develop semantic similarity metrics specific to Wolof grammar.

### Student 5: AKAKPO Sara
- **Contribution:** Constructed safety guardrails and input-validation matrices.
- **Technical Choice:** Implemented O(1) keyword intercept array for low-latency filtering.
- **Problem Diagnosed:** Adversarial prompt-injection bypassing lists. **Verification:** Tested boundaries with injection sequences.
- **Next Improvement:** Transition to lightweight semantic intent-detection.
