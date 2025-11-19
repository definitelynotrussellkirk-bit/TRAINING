#!/usr/bin/env python3
"""
Minimal training script to test if the hang is in our code or Transformers
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json

# Load tiny test data
data = []
with open("inbox/tiny.jsonl") as f:
    for line in f:
        data.append(json.loads(line))

# Format for training
formatted_data = []
for item in data:
    messages = item["messages"]
    text = f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n{messages[1]['content']}<|im_end|>"
    formatted_data.append({"text": text})

dataset = Dataset.from_list(formatted_data)

# Load model (4-bit)
print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/training/DIO_20251114",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained("/path/to/training/DIO_20251114")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Setup LoRA
print("Setting up LoRA...")
model = prepare_model_for_kbit_training(model)
lora_config = LoraConfig(
    r=128,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.02,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Tokenize
print("Tokenizing...")
def tokenize_fn(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=2048,
        padding=False
    )

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"], num_proc=None)

# Training args - MINIMAL
training_args = TrainingArguments(
    output_dir="test_minimal_output",
    max_steps=20,  # Just 20 steps to test
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=0.0002,
    logging_steps=5,
    save_steps=100,
    fp16=True,
    report_to="none"
)

# Trainer - NO CALLBACKS
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    callbacks=[]  # NO CALLBACKS
)

# Train
print("Starting training...")
trainer.train()
print("Training completed!")
