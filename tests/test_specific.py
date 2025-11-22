#!/usr/bin/env python3
"""Test the specific question."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_path = "/path/to/training/current_model"
base_model_path = "/path/to/training/model_qwen25"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(model, model_path)
model.eval()

prompt = """Return grouped collections of entries that are low frequency words.

Items I:
  1. buy
  2. others
  3. cat
  4. more
  5. internet
  6. florida
  7. plant
  8. pieces
  9. drive
  10. eyes
  11. horse
  12. worry

Output indented JSON for the bucket map."""

messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

print("Generating answer...\n")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        temperature=None,
        top_p=None
    )

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print("🤖 Model Answer:")
print(response)
