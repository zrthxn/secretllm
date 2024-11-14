import os
import streamlit as st

from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline


# The directory where you have the models saved. Each model is saved to its own directory
# inside this parent directory. This script will automatically discover the models you have.
DIRS = Path(".").absolute() / ".checkpoints"

# The hardware device you want to use. Use "cuda" if you have an NVIDIA GPU. Use "mps" if you're on an M series Mac.
DEVICE = "mps"

"""
# Playground
"""

# Automatic discovery of models
available_models = {}
for name in os.listdir(DIRS):
    if (DIRS / name).is_dir() and "model.safetensors" in os.listdir(DIRS / name):
        available_models[name] = DIRS / name

prompt_col, model_col = st.columns((9, 3))
with model_col:
    model_name = st.selectbox("Model", options=available_models.keys(), index=0)

    model = AutoModelForCausalLM.from_pretrained(available_models[model_name])
    tokenizer = AutoTokenizer.from_pretrained(available_models[model_name] / "tokenizer")
    tokenizer.pad_token_id = tokenizer.eos_token_id

with prompt_col:
    prompt = st.text_input("Prompt", placeholder="Write a prompt in the model's language", key="prompt")
    generate = pipeline("text-generation", model=model, tokenizer=tokenizer, device=DEVICE)
    output = generate(prompt, max_new_tokens = 250)[0]["generated_text"]

st.text_area("Generated", placeholder="Generated text comes here", height=300, value=output)
