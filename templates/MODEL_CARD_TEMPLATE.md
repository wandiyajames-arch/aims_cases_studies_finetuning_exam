# Model Card: Wolof AI Assistant (Qwen3-0.6B-LoRA)

## Model Summary

- **Base model:** `Qwen/Qwen3-0.6B`
- **Adaptation method:** LoRA / PEFT (Parameter-Efficient Fine-Tuning)
- **Target task:** Conversational interaction, cross-lingual translation, Wolof orthography correction, and domain-specific question-answering.
- **Target users:** Wolof speakers, Senegalese students, linguistic researchers, and developers building low-resource NLP applications.
- **Target language/domain:** Wolof (wo), with secondary alignment for French (fr) and English (en).
- **Hugging Face model repo:** [wandiya39/qwen3-0.6b-wolof-lora](https://huggingface.co/wandiya39/qwen3-0.6b-wolof-lora)
- **Hugging Face Space:** [Wolof AI Assistant](https://huggingface.co/spaces/wandiya39/Wolof-AI-Assistant)

## Intended Use

This model is an end-to-end NLP pipeline fine-tuned specifically to act as a conversational assistant in the Wolof language. It is designed to process user queries, provide culturally contextualized responses, translate cross-lingual instructions, and correct non-standard Wolof spelling into standard academic orthography based on curated linguistic guidelines (Soynade).

## Out-of-Scope Use

The model should not be used for critical medical diagnostics, legal advice, or high-stakes factual verification without human oversight. Due to its microscopic parameter count (0.6 Billion), it is not a comprehensive Wolof dictionary and should not be relied upon for rote memorization of highly technical or modern scientific vocabulary.

## Data Methodology

| Source | Type | Size | License/Access | Cleaning Method | Role |
|--------|------|------|----------------|-----------------|------|
| **Aya Dataset** | public | 1,500 | CC-BY-NC-SA | Filtered for Wolof-English pairs; removed incomplete translations. | train/val/eval |
| **Soynade** | public | 1,000 | MIT | Standardized text encoding to ensure correct rendering of special characters (ë, ñ, ŋ, ó). | train/val/eval |
| **Synthetic** | synthetic | 1,433 | Open | Deduplicated strings; removed recursive conversational loops. | train/val/eval |

## Data Splits

*Note: Data splits were executed deterministically (90/5/5 ratio) per source family before model tokenization to ensure clean distribution tracking.*

| Split | Number of examples | Ratio | Notes |
|-------|--------------------|-------|-------|
| **Train** | 3,539 | 90.0% | Used for active gradient updates and adapter weight adjustments. |
| **Validation** | 197 | 5.0% | Used to monitor loss decay trends and prevent training overruns. |
| **Evaluation** | 197 | 5.0% | Completely isolated test set used for final performance reporting. |

## Chat Template and Training Labels

Data was converted into the standard Qwen ChatML format to teach the model conversational boundaries:

```text
system: You are a helpful AI assistant specialized in Wolof and Senegalese/African contexts. Answer in clear Wolof. Keep the response useful, factual, and appropriate for the requested category. Do not write hidden reasoning. Never output <think> or </think> tags.
user: Na nga def?
assistant: Maa ngi fi.
```

**Training Labels Setup (Assistant-Only Loss):**
- `system` tokens: `-100` (Masked)
- `user` tokens: `-100` (Masked)
- `padding` tokens: `-100` (Masked)
- `assistant` output tokens: learned by the model (calculated for Cross-Entropy Loss)

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `Qwen/Qwen3-0.6B` |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| Target Modules | `q_proj`, `v_proj` |
| Learning rate | 1e-4 (with Cosine Decay Schedule) |
| Epochs | 3 |
| Batch size | 4 |
| Gradient accumulation | 8 |
| Max sequence length | 512 |
| Checkpoint policy | Save `best_adapter` automatically based on lowest validation loss |

## Evaluation

The model was evaluated against the 197 held-out evaluation examples using the `evaluation.py` pipeline script.

| Metric | Value |
|--------|-------|
| Final Validation Loss | 0.3781 |
| Exact Match (EM) | 14.2% |
| Token F1 | 69.5% |
| BLEU | 26.1 |
| ROUGE-L | 49.8 |

**Representative Qualitative Examples:**

| Prompt | Reference | Model Output | Comment |
|--------|-----------|--------------|---------|
| *Na nga def?* | *Maa ngi fi.* | *Mën nga ne: Na nga def? Kër gi sa kër la?* | Excellent cultural alignment. Expands on the prompt contextually. |
| *Fan lañuy defe coupe du monde 2022 ?* | *Kataar.* | *Kataar.* | Accurate factual retrieval and direct Exact Match. |
| *Corrige l'orthographe: Loolou dafa oub sama bopp lool rek.* | *Loolu dafa ub sama bopp lool.* | *Loolu dafa ub sama bopp lool.* | Successfully corrects non-standard orthography to standard formatting. |
| *What is food in Wolof?* | *ñam / lekk* | *sàkk / mënt* | Demonstrates vocabulary hallucination typical of sub-1B models under limited data. |

## Deployment

- **Model Hub URL:** [https://huggingface.co/wandiya39/qwen3-0.6b-wolof-lora](https://huggingface.co/wandiya39/qwen3-0.6b-wolof-lora)
- **Space URL:** [https://huggingface.co/spaces/wandiya39/Wolof-AI-Assistant](https://huggingface.co/spaces/wandiya39/Wolof-AI-Assistant)
- **Inference framework:** Gradio (`gr.ChatInterface`)
- **Required hardware:** Standard CPU or Hugging Face basic T4 GPU
- **Average latency:** ~2-3 seconds per response

## Limitations

- **Hallucination:** Lacking the structural capacity for deep dictionary storage, this 0.6B parameter model occasionally hallucinates specialized terminology roots.
- **Looping/Repetition:** Highly prone to infinite generation looping if inference settings are left completely unpenalized. This is mitigated by enforcing a `repetition_penalty=1.15` configuration block.

## Safety and Responsible Use

The deployed application implements structural guardrails to ensure safe routing. 
- **Refusal Behavior:** Out-of-scope toxic keyword patterns immediately halt processing execution loops to output hardcoded refusal statements.
- **Prompt Injection:** Due to its lightweight scale, the model is vulnerable to creative adversarial overrides designed to bypass persona guidelines.

## Authors

- **Group:** Group 1
- **Members:**
  - WANDIYA James
  - WADE Ndeye Khady
  - SIGEI Charlotte
  - GAOLATHE Angelah
  - AKAKPO Sara
- **Course:** Applied Generative and Agentic AI, AIMS Senegal
