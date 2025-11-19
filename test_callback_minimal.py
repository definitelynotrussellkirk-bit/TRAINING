#!/usr/bin/env python3
"""
Test with progressively complex callbacks to find the hang
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import time

# Test 1: Empty callback
class EmptyCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        print(f"Step {state.global_step}", flush=True)
        return control

# Test 2: Callback with state access
class StateAccessCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        current_loss = state.log_history[-1].get('loss', 0.0) if state.log_history else 0.0
        print(f"Step {state.global_step}, loss: {current_loss}", flush=True)
        return control

# Test 3: Callback with time check
class TimeCheckCallback(TrainerCallback):
    def __init__(self):
        self.last_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        current_time = time.time()
        elapsed = current_time - self.last_time
        self.last_time = current_time
        print(f"Step {state.global_step}, elapsed: {elapsed:.2f}s", flush=True)
        return control

# Test 4: Callback with periodic check (every 10 steps)
class PeriodicCallback(TrainerCallback):
    def __init__(self):
        self.last_check = 0

    def on_step_end(self, args, state, control, **kwargs):
        if (state.global_step - self.last_check) >= 10:
            self.last_check = state.global_step
            print(f"Step {state.global_step} - PERIODIC CHECK", flush=True)
        return control

# Test 5: Callback with file I/O
class FileIOCallback(TrainerCallback):
    def __init__(self):
        self.status_file = "test_status.json"

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 10 == 0:
            with open(self.status_file, 'w') as f:
                json.dump({"step": state.global_step}, f)
            print(f"Step {state.global_step} - wrote file", flush=True)
        return control

# Choose which callback to test
import sys
callback_type = sys.argv[1] if len(sys.argv) > 1 else "empty"

callback_map = {
    "empty": EmptyCallback(),
    "state": StateAccessCallback(),
    "time": TimeCheckCallback(),
    "periodic": PeriodicCallback(),
    "fileio": FileIOCallback()
}

callback = callback_map.get(callback_type, EmptyCallback())

print(f"\n{'='*60}")
print(f"Testing with {callback_type} callback")
print(f"{'='*60}\n")

# Load tiny test data
data = []
with open("inbox/tiny.jsonl") as f:
    for line in f:
        data.append(json.loads(line))

formatted_data = []
for item in data:
    messages = item["messages"]
    text = f"<|im_start|>user\n{messages[0]['content']}<|im_end|>\n<|im_start|>assistant\n{messages[1]['content']}<|im_end|>"
    formatted_data.append({"text": text})

dataset = Dataset.from_list(formatted_data)

# Load model
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
    return tokenizer(examples["text"], truncation=True, max_length=2048, padding=False)

tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"], num_proc=None)

# Training args
training_args = TrainingArguments(
    output_dir="test_callback_output",
    max_steps=20,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=0.0002,
    logging_steps=5,
    save_steps=100,
    fp16=True,
    report_to="none"
)

# Trainer with our test callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    callbacks=[callback]
)

# Train
print("Starting training...")
start = time.time()
trainer.train()
elapsed = time.time() - start
print(f"\nTraining completed in {elapsed:.1f}s")
print(f"Average per step: {elapsed/20:.1f}s")
