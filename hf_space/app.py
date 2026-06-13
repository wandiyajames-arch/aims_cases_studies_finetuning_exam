"""Minimal Gradio Space for a LoRA-adapted instruction model."""

from __future__ import annotations

import os
import re

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = os.getenv("BASE_MODEL", "Qwen/Qwen3-0.6B")
MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "YOUR_USERNAME/YOUR_LORA_MODEL_REPO")
SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful assistant. Answer clearly and concisely.",
)

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
THINK_TAG_RE = re.compile(r"</?think>", flags=re.IGNORECASE)


def pick_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def clean_answer(text: str) -> str:
    text = THINK_BLOCK_RE.sub("", text)
    text = THINK_TAG_RE.sub("", text)
    return text.strip()


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=pick_dtype(),
        trust_remote_code=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base, MODEL_REPO_ID)
    if not torch.cuda.is_available():
        model = model.to("cpu")
    model.eval()
    return model, tokenizer


MODEL, TOKENIZER = load_model()
DEVICE = pick_device()


def render_prompt(user_message: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message.strip()},
    ]
    try:
        return TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def generate(user_message: str, max_new_tokens: int, temperature: float) -> str:
    if not user_message.strip():
        return "Please enter a prompt."

    prompt = render_prompt(user_message)
    inputs = TOKENIZER(prompt, return_tensors="pt").to(DEVICE)
    do_sample = temperature > 0
    kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": do_sample,
        "pad_token_id": TOKENIZER.eos_token_id,
        "repetition_penalty": 1.12,
        "no_repeat_ngram_size": 3,
    }
    if do_sample:
        kwargs["temperature"] = float(temperature)
        kwargs["top_p"] = 0.9

    with torch.no_grad():
        output_ids = MODEL.generate(**inputs, **kwargs)
    generated = output_ids[0, inputs["input_ids"].shape[-1] :]
    return clean_answer(TOKENIZER.decode(generated, skip_special_tokens=True))


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Prompt", lines=5),
        gr.Slider(16, 256, value=128, step=8, label="Max new tokens"),
        gr.Slider(0.0, 1.0, value=0.2, step=0.05, label="Temperature"),
    ],
    outputs=gr.Textbox(label="Model answer", lines=8),
    title="Real LLM Deployment: Fine-Tuned Assistant",
    description="Replace MODEL_REPO_ID with your Hugging Face LoRA model repository.",
)


if __name__ == "__main__":
    demo.launch()
