import re
import torch
import gradio as gr
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURATION ---
BASE_MODEL = "Qwen/Qwen3-0.6B"
MODEL_REPO_ID = "wandiya39/qwen3-0.6b-wolof-lora"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load Model
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float32, trust_remote_code=True)
model = PeftModel.from_pretrained(base, MODEL_REPO_ID).to(DEVICE).eval()

def translate_engine(text, mode):
    if not text.strip(): return ""
    
    # SYSTEM ORCHESTRATION: Force the model into 'Translator' role
    # We explicitly define the task boundaries here.
    lang_task = "Wolof" if mode == "English to Wolof" else "English"
    prompt = f"<|im_start|>system\nYou are a professional translator. Convert input text to {lang_task}. Do not repeat the input. Return only the translated text.<|im_end|>\n<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    # Inference parameters: Low temperature = Fact-based, High repetition penalty = No mirroring
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=150, 
            repetition_penalty=1.5, 
            temperature=0.1, 
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode and Scrub
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE).strip()

# --- PROFESSIONAL UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🚀 Wolof-English Translation Engine")
    
    with gr.Row():
        with gr.Column():
            input_box = gr.Textbox(label="Source Text", lines=8, placeholder="Enter text here...")
            direction = gr.Radio(["English to Wolof", "Wolof to English"], label="Translation Direction", value="English to Wolof")
            submit = gr.Button("Execute Translation", variant="primary")
        
        with gr.Column():
            output_box = gr.Textbox(label="AI-Generated Output", lines=8, interactive=False)

    submit.click(fn=translate_engine, inputs=[input_box, direction], outputs=output_box)

if __name__ == "__main__":
    demo.launch()
