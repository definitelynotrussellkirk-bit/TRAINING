#!/usr/bin/env python3
"""Simple merge - no backups, just merge adapter into base and save with name"""
import torch
from peft import PeftModel, AutoPeftModelForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
from datetime import datetime

# Output name
output_name = f"DIO_{datetime.now().strftime('%Y%m%d')}"
output_path = f"/path/to/training/{output_name}"

print(f"🔄 Merging adapter into base model...")
print(f"📁 Will save as: {output_path}")

# Load model with adapter
print("Loading model with adapter...")
model = AutoPeftModelForCausalLM.from_pretrained(
    "/path/to/training/current_model",
    device_map="auto",
    torch_dtype=torch.float16
)

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("/path/to/training/current_model")

# Merge adapter into base
print("Merging adapter weights into base model...")
merged_model = model.merge_and_unload()

# Save merged model
print(f"Saving merged model to {output_path}...")
merged_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print(f"✅ DONE! Merged model saved as: {output_name}")
print(f"📁 Location: {output_path}")
