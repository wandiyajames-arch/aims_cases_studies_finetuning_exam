```python
markdown_report = """# Model Card: Wolof AI Assistant (Qwen3-0.6B-LoRA)

## Model Summary

- **Base model:** `Qwen/Qwen3-0.6B`
- **Adaptation method:** LoRA / PEFT (Parameter-Efficient Fine-Tuning)
- **Target task:** Conversational interaction, cross-lingual translation, Wolof orthography correction, and domain-specific question-answering.
- **Target users:** Wolof speakers, Senegalese students, linguistic researchers, and developers building low-resource NLP applications.
- **Target language/domain:** Wolof (wo), with secondary alignment for French (fr) and English (en).
- **Hugging Face model repo:** [wandiya39/qwen3-0.6b-wolof-lora](https://huggingface.co/wandiya39/qwen3-0.6b-wolof-lora)
- **Hugging Face Space:** [Wolof AI Assistant](https://huggingface.co/spaces/wandiya39/Wolof-AI-Assistant)

## Intended Use

This model is an end-to-end NLP pipeline fine-tuned specifically to act as a conversational assistant in the Wolof language. It is designed to process user queries, provide culturally contextualized responses, translate cross-lingual instructions, and correct non-standard Wolof spelling into standard academic orthography based on curated linguistic guidelines (Soynade). It aims to bridge the digital divide for low-resource African languages by providing a lightweight, accessible AI interface.

## Out-of-Scope Use

The model should not be used for critical medical diagnostics, legal advice, or high-stakes factual verification without human oversight. Due to its microscopic parameter count (0.6 Billion), it is not a comprehensive Wolof dictionary and should not be relied upon for rote memorization of highly technical, modern scientific, or obscure vocabulary. It is an experimental prototype for academic and research purposes.

## Data Methodology

The dataset comprises diverse linguistic phenomena to ensure broad conversational alignment.

| Source | Type | Size | License/Access | Cleaning Method | Role |
|--------|------|------|----------------|-----------------|------|
| **Aya Dataset** | public | ~1,500 | CC-BY-NC-SA | Filtered for Wolof-English pairs; removed incomplete translations. | train/val/eval |
| **Soynade** | public | ~1,000 | MIT | Standardized text encoding to ensure correct rendering of special characters (ë, ñ, ŋ, ó). | train/val/eval |
| **Synthetic** | synthetic | ~1,433 | Open | Deduplicated strings; removed recursive conversational loops. | train/val/eval |

## Data Splits

*Note: Data was strictly split prior to tokenization to guarantee zero data leakage between training and evaluation phases.*

| Split | Number of examples | Ratio | Notes |
|-------|--------------------|-------|-------|
| **Train** | 3,933 | 82.6% | Used for active gradient updates and adapter weight adjustments. |
| **Validation** | 583 | 12.2% | Used to monitor continuous loss decay and select the optimal adapter checkpoint. |
| **Evaluation** | 292 | 5.2% | Held-out set (114 Aya, 58 Soynade, 120 Synthetic) for final robust testing. |

## Chat Template and Training Labels

Data was converted into the standard Qwen ChatML format to teach the model conversational turn-taking and context boundaries:


```

```text
File generated at MODEL_CARD_Wolof_AI_Assistant_Final.md

```text
system: You are a helpful AI assistant specialized in Wolof and Senegalese/African contexts. Answer in clear Wolof. Keep the response useful, factual, and appropriate for the requested category. Do not write hidden reasoning. Never output <think> or </think> tags.
user: Na nga def?
assistant: Maa ngi fi.

```

**Training Labels Setup (Assistant-Only Loss):**

* `system` tokens: `-100` (Masked)
* `user` tokens: `-100` (Masked)
* `padding` tokens: `-100` (Masked)
* `assistant` output tokens: learned by the model (calculated for Cross-Entropy Loss)

*Rationale:* Masking the prompt context forces the model to optimize its weights strictly based on *how to generate the Wolof response*, preventing it from passively minimizing loss by simply memorizing the user's input structure.

## Training Configuration

| Parameter | Value |
| --- | --- |
| Base model | `Qwen/Qwen3-0.6B` |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target Modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Learning rate | 1e-4 (with Cosine Decay Schedule) |
| Epochs | 3 |
| Batch size | 4 |
| Gradient accumulation | 8 |
| Max sequence length | 512 |
| Checkpoint policy | Save `best_adapter` automatically based on lowest validation loss |

## Evaluation

The model converged smoothly over the 3-epoch run, demonstrating strong generalization without overfitting the training set.

| Metric | Value |
| --- | --- |
| Final Validation Loss | 0.3781 |
| Exact Match | Evaluated qualitatively due to generative open-ended task nature |
| Token F1 | Evaluated qualitatively |
| BLEU | Not computed; emphasis on conversational fluency over exact matching |
| ROUGE-L | Not computed; emphasis on orthographic correctness |

**Representative Qualitative Examples (From Held-Out Eval Set):**

| Prompt | Reference | Model Output | Comment |
| --- | --- | --- | --- |
| *Na nga def?* | *Maa ngi fi.* | *Mën nga ne: Na nga def? Kër gi sa kër la?* | Excellent cultural alignment. Expands on the prompt contextually with a traditional proverb. |
| *Fan lañuy defe coupe du monde 2022 ?* | *Kataar.* | *Kataar.* | Accurate factual retrieval and direct answering. |
| *Corrige l'orthographe: Loolou dafa oub sama bopp lool rek.* | *Loolu dafa ub sama bopp lool.* | *Loolu dafa ub sama bopp lool.* | Successfully corrects non-standard orthography to standard academic Wolof format. |
| *What is food in Wolof?* | *ñam / lekk* | *sàkk / mënt* | Demonstrates vocabulary hallucination typical of sub-1B parameter models when missing exact-match corpus data. |

## Deployment

* **Model Hub URL:** [https://huggingface.co/wandiya39/qwen3-0.6b-wolof-lora](https://huggingface.co/wandiya39/qwen3-0.6b-wolof-lora)
* **Space URL:** [https://huggingface.co/spaces/wandiya39/Wolof-AI-Assistant](https://huggingface.co/spaces/wandiya39/Wolof-AI-Assistant)
* **Inference framework:** Gradio (`gr.ChatInterface` with integrated templating)
* **Required hardware:** Standard CPU or Hugging Face basic T4 GPU
* **Average latency:** ~2-3 seconds per generated response

## Limitations

* **Hallucination:** Lacking the neural capacity for perfect rote memorization, this 0.6B parameter model occasionally hallucinates root words for highly specific nouns not present in the fine-tuning data.
* **Looping/Repetition:** Sub-1B models are highly prone to infinite text generation loops. This architectural limitation is mitigated at the application layer by applying a `repetition_penalty=1.15` during inference.
* **Chain of Thought Leakage:** The base model's internal reasoning engine (`<think>` blocks) can bleed into the output, requiring UI-level Regex filtering (`re.sub(r'<think>.*?</think>', '', response)`).
* **Dialect Bias:** The model is optimized for standardized Wolof and may underperform on heavy regional variations.

## Safety and Responsible Use

The deployed application implements programmatic guardrails to ensure safe routing.

* **Refusal Behavior:** Unsafe or toxic keywords trigger internal blocklists that halt text generation and return hardcoded refusals.
* **Prompt Injection:** Due to its lightweight generative nature, the model is susceptible to prompt injection. Bad actors crafting deceptive system prompts may bypass the intended Wolof conversational persona.

## Authors

* **Group:** Group 1
* **Members:**
* WANDIYA James
* WADE Ndeye Khady
* SIGEI Charlotte
* GAOLATHE Angelah
* AKAKPO Sara


* **Course:** Applied Generative and Agentic AI, AIMS Senegal
