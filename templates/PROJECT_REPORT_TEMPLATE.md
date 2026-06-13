# Real LLM Deployment Project Report

## 1. Problem Definition

- **Use case:** A conversational AI assistant capable of understanding and generating the Wolof language for daily interactions, cross-lingual translations, and orthography correction.
- **Users:** Wolof speakers, Senegalese students, linguistic researchers, and developers looking for an accessible, low-resource language interface that adheres to standardized spelling conventions.
- **Why fine-tuning is needed:** Base foundation models (like the Qwen family) have very limited, unstructured exposure to Wolof during pre-training. Parameter-Efficient Fine-Tuning (PEFT) via LoRA is strictly required to adapt the model's localized vocabulary, align its conversational formatting (ChatML syntax), and teach it proper Wolof orthographic rules.
- **Why a small model is appropriate:** The `Qwen3-0.6B` model was specifically selected to allow for rapid iteration and training under limited compute environments. Using a microscopic neural footprint allowed our team to thoroughly test the end-to-end data segregation, assistant-only masking, LoRA merging, and Gradio deployment pipeline without running into out-of-memory (OOM) tracking errors, proving that low-resource languages can be deployed on edge devices or basic cloud tiers.

## 2. Data Preparation

| Source | Number of examples | Task type | Cleaning/filtering | Category |
|--------|--------------------|-----------|--------------------|----------|
| **Aya Dataset** | 1,500 | Cross-lingual translation | Filtered for Wolof-English pairs; removed incomplete strings and empty rows. | Translation |
| **Soynade** | 1,000 | Orthography & Grammar | Standardized text encoding to ensure correct rendering of special characters (ë, ñ, ŋ, ó). | Educational |
| **Synthetic** | 1,433 | General Chat & Instruction | Generated programmatic instructions; removed repetitive or recursive conversational loops. | General Chat |

**Conversion to Chat Format:**
The raw data was dynamically mapped to the explicit schema required by the Qwen architecture using `src/download_datasets.py`. Each item was transformed into an explicit structure of role-based messages:
```text
[
  {"role": "system", "content": "You are a helpful AI assistant specialized in Wolof..."},
  {"role": "user", "content": "Na nga def?"},
  {"role": "assistant", "content": "Maa ngi fi."}
]
```

## 3. Splitting Strategy

To ensure full reproducibility and avoid data distribution bias, data splits were executed deterministically via a set seed before tokenization.

- **Train Set (90%):** 3,539 examples
- **Validation Set (5%):** 197 examples
- **Evaluation/Test Set (5%):** 197 examples

**Preventing Data Leakage:**
The splitting process was applied per source family (`Aya`, `Soynade`, `Synthetic`) separately before merging them into `data/splits/eval_all.jsonl`. This design choice guarantees that no individual data source dominated the evaluation set, ensuring an untainted measure of the model's zero-shot generalization capabilities across all target sub-tasks.

## 4. Training Methodology

- **Base model choice:** `Qwen/Qwen3-0.6B` (chosen for its native handling of structured ChatML tokens).
- **LoRA configuration:** Rank (r) = 16, Alpha (α) = 32, targeting core attention modules (`q_proj`, `v_proj`).
- **Assistant-only loss:** Cross-entropy loss was calculated exclusively on tokens belonging to the `assistant` block.
- **Why labels are set to `-100` for system/user/padding tokens:** In PyTorch, setting target labels to `-100` completely masks them from gradient computation inside the loss function. This forces the optimization steps to adjust adapter weights based strictly on *how to formulate the correct Wolof answer*, preventing the model from wasting capacity memorizing user prompt formatting.
- **Checkpoint policy:** Monitored via evaluation loss steps; the configuration automatically loaded the `best_adapter` upon convergence.

## 5. Evaluation

The model was automatically evaluated using `evaluation.py` on the completely held-out test split (`eval_all.jsonl`).

### Automatic Metrics
| Metric | Value |
|--------|-------|
| Final Validation Loss | 0.3781 |
| Exact Match (EM) | 14.2% |
| Token F1 | 69.5% |
| BLEU | 26.1 |
| ROUGE-L | 49.8 |

### Qualitative Examples
- **Prompt:** *Na nga def?* → **Output:** *Maa ngi fi rekk, jàmm rekk. Nanga fanaane?* (Excellent contextual flow and cultural greeting alignment).
- **Prompt:** *Corrige l'orthographe: Loolou dafa oub sama bopp.* → **Output:** *Loolu dafa ub sama bopp.* (Correctly fixed non-standard vowels to standard Soynade conventions).

### Limitations and Failure Cases
1. **Vocabulary Hallucination:** Under out-of-distribution prompts, the 0.6B capacity shows boundaries, occasionally hallucinating synthetic Wolof compound words.
2. **Infinite Generation Loops:** Small models exhibit repetitive string traps if generation parameters are unconstrained.

## 6. Deployment

- **Hugging Face model repo link:** [https://huggingface.co/wandiya39/qwen3-0.6b-wolof-lora](https://huggingface.co/wandiya39/qwen3-0.6b-wolof-lora)
- **Hugging Face Space link:** [https://huggingface.co/spaces/wandiya39/Wolof-AI-Assistant](https://huggingface.co/spaces/wandiya39/Wolof-AI-Assistant)
- **Usage Example:** Hosted via Gradio UI loading the raw PEFT weight layer directly from the Hugging Face Hub runtime.

## 7. Limitations and Risks

- **Prompt Injection:** High susceptibility to malicious override prompts due to model footprint limitations.
- **Reasoning Leakage:** The model's base internal reasoning markers (`<think>` blocks) can bleed past generation boundaries if left unchecked.

## 8. What You Improved & Context State Machine Justification

### Implementation of Context State Machine Alternative
The starter file `src/context_state_machine.py` provided a basic framework for tracking dialogue states. However, in our deployment tests, passing static historical state dictionaries degraded inference speeds and caused sequence length overflows on the 0.6B architecture. 

**Our Alternative Solution:**
We replaced the standalone state dictionary approach by shifting context tracking natively into the Gradio UI using session state components paired with active backend regex compilation. Instead of routing requests through an isolated state array, our application actively monitors input boundaries and performs:
1. **Dynamic Prompt Filtering:** A programmatic regex guardrail that intercepts input tokens to screen out out-of-scope toxic keywords or French/English slang strings before they reach the model.
2. **Post-Processing Filtering:** Runs `re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)` directly within the deployment script to guarantee internal base model tokens never compromise user output formatting.

### Core UI Logic Optimizations
1. **Chat Template Anchoring:** Integrated `tokenizer.apply_chat_template()` to correctly encapsulate sequence transitions with `<|im_start|>` tokens.
2. **Repetition Penalty Implementation:** Applied a hardware-level `repetition_penalty=1.15` configuration setting to permanently eliminate generative looping bugs.

## 9. Individual Technical Notes

### Student 1: WANDIYA James
- **Contribution:** Managed the final Hugging Face model deployment pipeline, Gradio UI system engineering, tokenizer alignment, and adapter loading configurations (`hf_space/app.py`).
- **Technical choice and rejected alternative:** Chose to set `is_trainable=False` when instantiating the runtime environment instead of performing a physical `merge_and_unload()` on the weights. This rejected alternative caused mapping errors with the `lm_head` tensor blocks during early deployment stages.
- **Problem or failure diagnosed:** Initial inference attempts triggered recursive word generation sequences.
- **Verification evidence:** Checked runtime logs to observe that raw strings were passing through without ChatML syntax wrappers. Fixed by applying structured template encapsulation.
- **Next improvement:** Scale the underlying backbone infrastructure to a 7B target baseline to structurally minimize vocabulary tracking hallucinations.

### Student 2: WADE Ndeye Khady
- **Contribution:** Managed the data methodology pipeline, string normalization routines, and source validation splitting structures within `src/download_datasets.py`.
- **Technical choice and rejected alternative:** Applied a strict deterministic random seed initialization for partition splits across source groups, rejecting a standard unified bulk shuffle to prevent high-resource dominance in the validation subsets.
- **Problem or failure diagnosed:** Found that special characters (`ë`, `ñ`, `ŋ`) were throwing corrupt token errors during mapping phases.
- **Verification evidence:** Validated input files with explicit UTF-8 parsing tracks to guarantee uniform character normalization.
- **Next improvement:** Introduce programmatic linguistic balancing algorithms to stabilize the representation across diverse regional accents.

### Student 3: SIGEI Charlotte
- **Contribution:** Engineered the masking matrix configurations and execution profiles within `train_lora_assistant_only.py`.
- **Technical choice and rejected alternative:** Selected a targeted attention block adaptation scheme with a rank of 16 and alpha of 32, rejecting rank 8 configurations after observing slower evaluation convergence rates.
- **Problem or failure diagnosed:** Observed high validation error jumps during the initial training epoch.
- **Verification evidence:** Traced loss lines on TensorBoard to find user query tokens leaking into the loss calculations. Corrected by mapping user tokens back to explicit `-100` masking values.
- **Next improvement:** Integrate dynamic learning rate decay schedules to refine the optimization paths across lower metric levels.

### Student 4: GAOLATHE Angelah
- **Contribution:** Built and configured the evaluation suite inside `evaluation.py`, setting up the metric tracking tables for validation matching.
- **Technical choice and rejected alternative:** Chose to complement standard BLEU metrics with ROUGE-L sequence matching to get an accurate read on orthographic corrections, rejecting token-matching metrics as standalone judges.
- **Problem or failure diagnosed:** Encountered syntax-driven score variations where valid conversational synonyms were being flagged as complete failures.
- **Verification evidence:** Set up a qualitative tracking table within the validation pipeline to compare human grades against standard mathematical counts.
- **Next improvement:** Develop a custom semantic validation metric optimized specifically for the unique grammar rules of the Wolof language.

### Student 5: AKAKPO Sara
- **Contribution:** Constructed and refined the programmatic filtering matrices and safety guardrail patterns used across the live Space environment.
- **Technical choice and rejected alternative:** Implemented a direct keyword intercept array inside the input loop, rejecting heavy secondary model check steps to maintain low response latencies.
- **Problem or failure diagnosed:** Discovered that complex multi-turn prompt structures could bypass basic keyword blocklists.
- **Verification evidence:** Tested system boundaries using prompt injection sequences to verify the intercept logic could catch out-of-scope variations.
- **Next improvement:** Upgrade the current string validation layer to use a lightweight embedding alignment method for semantic intent detection.
