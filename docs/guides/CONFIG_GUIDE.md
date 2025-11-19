# Training Configuration Guide

## Current Config (`config.json`)

Your training configuration has been set up for the Qwen3-VL-8B-Thinking model on RTX 3090.

### Configuration Parameters

#### Model Settings
```json
"model_name": "qwen3_vl_8b_thinking"
"model_path": "/path/to/training/model"
```
- **model_name**: Identifier for your model
- **model_path**: Location of the base model files

#### Batch & Memory Settings
```json
"batch_size": 1
"gradient_accumulation": 16
```
- **batch_size**: Number of samples per GPU forward pass (1 = fits 8B model in 24GB VRAM)
- **gradient_accumulation**: Accumulate gradients over N steps before updating
- **Effective batch size** = batch_size × gradient_accumulation = **16**

💡 **Why these values?**
- batch_size=1 ensures the 8B model fits comfortably in 24GB VRAM
- gradient_accumulation=16 gives you effective batch of 16 for stable training
- Larger effective batch = more stable gradients but slower training

#### Learning Rate Settings
```json
"learning_rate": 2e-4
"warmup_steps": 100
```
- **learning_rate**: How fast the model learns (2e-4 = 0.0002, standard for LoRA)
- **warmup_steps**: Gradually increase LR over first 100 steps (prevents early instability)

💡 **Tuning tips:**
- Too high (>5e-4): Model may not converge, loss bounces around
- Too low (<1e-5): Training is very slow, may underfit
- 2e-4 is a solid default for LoRA fine-tuning

#### LoRA Settings
```json
"lora_r": 64
"lora_alpha": 32
```
- **lora_r**: Rank of LoRA matrices (higher = more capacity, more VRAM)
- **lora_alpha**: Scaling factor for LoRA weights (typically r/2 or r)

💡 **What this means:**
- r=64 gives strong adaptation capacity
- alpha=32 scales the LoRA contribution appropriately
- Higher r = better fine-tuning but more parameters to train

#### Evaluation & Saving
```json
"eval_steps": 625
"num_eval_samples": 5
"save_steps": 1250
```
- **eval_steps**: Run validation every 625 training steps
- **num_eval_samples**: Test on 5 examples during validation
- **save_steps**: Save checkpoint every 1250 steps

💡 **What to adjust:**
- Increase eval_steps if training is slow and you don't need frequent checks
- Increase num_eval_samples (to 10-20) for better quality assessment
- Decrease save_steps if you want more frequent checkpoints

#### Daemon Settings
```json
"poll_interval": 30
"snapshot_time": "03:00"
```
- **poll_interval**: Check inbox for new data every 30 seconds
- **snapshot_time**: Create daily snapshot at 3:00 AM

#### Sequence Length
```json
"max_length": 2048
```
- **max_length**: Maximum sequence length in tokens
- Longer sequences = more memory usage
- LEO data typically fits in 1024-1536 tokens, so 2048 is safe

---

## Configuration Profiles

### Conservative (Slower but Safer)
```json
{
  "learning_rate": 1e-4,
  "warmup_steps": 200,
  "lora_r": 32,
  "lora_alpha": 16,
  "gradient_accumulation": 32
}
```
Use when: First time training, unsure about data quality, want stability

### Balanced (Current Default)
```json
{
  "learning_rate": 2e-4,
  "warmup_steps": 100,
  "lora_r": 64,
  "lora_alpha": 32,
  "gradient_accumulation": 16
}
```
Use when: Standard training, proven data, good for most cases

### Aggressive (Faster Learning)
```json
{
  "learning_rate": 5e-4,
  "warmup_steps": 50,
  "lora_r": 128,
  "lora_alpha": 64,
  "gradient_accumulation": 8
}
```
Use when: High-quality data, need faster convergence, willing to monitor closely

⚠️ **Warning**: Aggressive settings may cause instability. Monitor loss carefully!

---

## RTX 3090 Memory Guidelines

Your RTX 3090 has **24GB VRAM**. Here's what fits:

### Safe Zone (Current Settings)
- Model: Qwen3-VL-8B (~16GB)
- Batch size: 1
- LoRA r=64
- **VRAM usage**: ~18-20GB
- **Status**: ✅ Safe, room for growth

### Maximum Capacity
- Can push to batch_size=2 with r=64
- Or batch_size=1 with r=128
- **VRAM usage**: ~22-23GB
- **Status**: ⚠️ Tight, may OOM with long sequences

### If OOM (Out of Memory)
1. Reduce lora_r: 64 → 32 → 16
2. Ensure batch_size=1
3. Reduce max_length: 2048 → 1536 → 1024
4. Check no other processes using GPU

---

## Training Speed Estimates

With current config (batch=1, grad_accum=16):

**100K samples:**
- ~6,250 gradient updates
- ~20-25 hours on RTX 3090
- ~4,000-5,000 samples/hour

**1M samples (full training):**
- ~62,500 gradient updates
- ~200-250 hours (~10 days)
- Continuous feeding through inbox

---

## Quick Adjustments

### To train FASTER:
```json
"gradient_accumulation": 8,      // Half the effective batch
"eval_steps": 1000,               // Check less often
"save_steps": 2500                // Save less often
```

### To train MORE THOROUGHLY:
```json
"gradient_accumulation": 32,     // Double the effective batch
"warmup_steps": 200,              // Longer warmup
"learning_rate": 1e-4             // Lower LR
```

### To use LESS MEMORY:
```json
"lora_r": 32,                     // Half the LoRA rank
"max_length": 1536                // Shorter sequences
```

### To get BETTER QUALITY:
```json
"lora_r": 128,                    // More capacity
"num_eval_samples": 20,           // Better evaluation
"eval_steps": 500                 // More frequent checks
```

---

## Next Steps

1. **Test with small batch**: Copy one of your 100K batches to inbox/
2. **Monitor first epoch**: Watch loss decrease, check eval samples
3. **Adjust if needed**: Based on loss curve and quality
4. **Scale up**: Once stable, feed continuous data stream

Your config is ready to go! 🚀
