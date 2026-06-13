# Model Card: Wolof AI Assistant (Qwen3-0.6B-LoRA)

## Model Summary
- **Base model:** `Qwen/Qwen3-0.6B`
- **Adaptation method:** LoRA / PEFT
- **Target task:** Conversational interaction, Wolof orthography correction.
- **Hugging Face model repo:** [wandiya39/qwen3-0.6b-wolof-lora](https://huggingface.co/wandiya39/qwen3-0.6b-wolof-lora)
- **Hugging Face Space:** [Wolof AI Assistant](https://huggingface.co/spaces/wandiya39/Wolof-AI-Assistant)

## Intended Use
An end-to-end NLP pipeline fine-tuned to act as a conversational assistant for Wolof speakers, providing contextually relevant answers, translation, and academic orthography correction.

## Data Methodology
| Source | Type | Size |
|--------|------|------|
| **Aya** | Public | 1,500 |
| **Soynade**| Public | 1,000 |
| **Synthetic** | Synthetic| 1,433 |

## Evaluation
| Metric | Value |
|--------|-------|
| Final Validation Loss | 0.3781 |
| Exact Match | 14.2% |
| Token F1 | 69.5% |
| BLEU | 26.1 |
| ROUGE-L | 49.8 |

## Deployment
- **Framework:** Gradio `ChatInterface`.
- **Latency:** ~2-3 seconds per response.
- **Hardware:** Standard CPU / T4 GPU.

## Authors
- **Group 1:** WANDIYA James, WADE Ndeye Khady, SIGEI Charlotte, GAOLATHE Angelah, AKAKPO Sara.
- **Course:** Applied Generative and Agentic AI, AIMS Senegal.
