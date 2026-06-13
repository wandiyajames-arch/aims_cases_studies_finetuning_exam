"""Minimal Gradio Space for a LoRA-adapted instruction model."""

from __future__ import annotations

import os
import re

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- UPDATED: HARDCODED CONFIGURATION ---
# Using your specific Hugging Face repository IDs
BASE_MODEL = "Qwen/Qwen3-0.6B"
MODEL_REPO_ID = "wandiya39/qwen3-0.6b-wolof-lora" 
SYSTEM_PROMPT = "You are a helpful AI assistant. Answer in clear Wolof. DO NOT use <think> tags. DO NOT reason internally."
# ------------------------------------------

THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
THINK_TAG_RE = re.compile(r"</?think>", flags=re.IGNORECASE)

def pick_dtype():
    return torch.float32 # Safest for CPU-only Spaces

def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

def clean_answer(text: str) -> str:
    text = THINK_BLOCK_RE.sub("", text)
    text = THINK_TAG_RE.sub("", text)
    return text.strip()

def load_model():
    print(f"Loading tokenizer and model from {MODEL_REPO_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=pick_dtype(),
        trust_remote_code=True,
    )
    
    # Load your LoRA adapter
    model = PeftModel.from_pretrained(base, MODEL_REPO_ID)
    model.eval()
    return model.to(pick_device()), tokenizer

# Load globally
MODEL, TOKENIZER = load_model()
DEVICE = pick_device()

def render_prompt(user_message: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message.strip()},
    ]
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
    
    kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "do_sample": temperature > 0,
        "pad_token_id": TOKENIZER.eos_token_id,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 3,
    }
    if kwargs["do_sample"]:
        kwargs["temperature"] = float(temperature)
        kwargs["top_p"] = 0.9

    with torch.no_grad():
        output_ids = MODEL.generate(**inputs, **kwargs)
    
    generated = output_ids[0, inputs["input_ids"].shape[-1] :]
    return clean_answer(TOKENIZER.decode(generated, skip_special_tokens=True))

# Gradio Interface
demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="Wolof Assistant Prompt", lines=3),
        gr.Slider(16, 256, value=128, step=8, label="Max new tokens"),
        gr.Slider(0.0, 1.0, value=0.3, step=0.05, label="Temperature"),
    ],
    outputs=gr.Textbox(label="Wolof Assistant Response", lines=6),
    title="Wolof AI Assistant (LoRA-Adapted)",
    description="Fine-tuned Qwen-0.6B assistant for Wolof language and Senegalese context."
)

if __name__ == "__main__":
    demo.launch()
