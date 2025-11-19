# Validation System - Implementation Plan

**Created:** 2025-11-16
**Status:** 60% Complete - Ready for final implementation

---

## 🎯 GOAL

Add fixed validation set to measure true generalization (loss on unseen data) vs training loss (optimistic).

---

## ✅ COMPLETED

1. **Generated validation set** - `data/validation/syllo_validation_1000.jsonl` (1000 examples, seed 42)
2. **Added loader** - `train.py` lines 380-448: `load_fixed_validation_set()`
3. **Updated callback** - Added `fixed_val_dataset` parameter and `last_val_loss` tracking

---

## 🔧 REMAINING IMPLEMENTATION

### Step 1: Add Validation Loss Computation to Callback

**File:** `train.py` around line 660 (after existing eval code)

```python
# Add this method to LiveMonitorCallback class (after on_step_end)

def compute_validation_loss(self):
    """Compute loss on fixed validation set."""
    if self.fixed_val_dataset is None:
        return None

    try:
        import torch
        from torch.utils.data import DataLoader

        # Tokenize if not already done
        if not hasattr(self, '_tokenized_val'):
            def tokenize_fn(examples):
                return self.tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=2048,
                    padding="max_length"
                )

            self._tokenized_val = self.fixed_val_dataset.map(
                tokenize_fn,
                batched=True,
                remove_columns=self.fixed_val_dataset.column_names
            )

        # Compute loss
        self.model_ref.eval()
        total_loss = 0.0
        count = 0

        with torch.no_grad():
            for i in range(min(20, len(self._tokenized_val))):  # Sample 20 for speed
                example = self._tokenized_val[i]
                inputs = {
                    "input_ids": torch.tensor([example["input_ids"]]).to(self.model_ref.device),
                    "attention_mask": torch.tensor([example["attention_mask"]]).to(self.model_ref.device),
                    "labels": torch.tensor([example["input_ids"]]).to(self.model_ref.device)
                }
                outputs = self.model_ref(**inputs)
                total_loss += outputs.loss.item()
                count += 1

        self.model_ref.train()
        return total_loss / count if count > 0 else None

    except Exception as e:
        print(f"Warning: validation loss computation failed: {e}")
        return None
```

### Step 2: Call Validation Loss in on_step_end

**File:** `train.py` around line 616 (inside eval_steps check)

```python
# After the evolution tracking code (around line 720), add:

# Compute validation loss
if self.fixed_val_dataset is not None:
    val_loss = self.compute_validation_loss()
    if val_loss is not None:
        self.last_val_loss = val_loss
        gap = val_loss - current_loss
        status = "✅" if gap < 0.3 else "⚠️" if gap < 0.5 else "🚨"
        print(f"📊 Loss Comparison: Train={current_loss:.4f} | Val={val_loss:.4f} | Gap={gap:+.4f} {status}")
```

### Step 3: Update TrainingStatus Dataclass

**File:** `training_status.py` around line 30

```python
# Add to TrainingStatus dataclass:
validation_loss: Optional[float] = None
val_train_gap: Optional[float] = None  # val_loss - train_loss
```

### Step 4: Update StatusWriter to Accept val_loss

**File:** `training_status.py` around line 130 in `update_progress`

```python
def update_progress(
    self,
    step: int,
    total_steps: int,
    epoch: int,
    loss: float,
    lr: float,
    batch_step: Optional[int] = None,
    batch_total_steps: Optional[int] = None,
    batch_number: Optional[int] = None,
    batch_queue_size: Optional[int] = None,
    current_file: Optional[str] = None,
    val_loss: Optional[float] = None  # ADD THIS
):
    # ...existing code...

    # Calculate gap
    val_train_gap = None
    if val_loss is not None:
        val_train_gap = val_loss - loss

    status = TrainingStatus(
        # ...existing fields...
        validation_loss=val_loss,
        val_train_gap=val_train_gap,
        # ...
    )
```

### Step 5: Pass val_loss from Callback to StatusWriter

**File:** `train.py` around line 601 in `on_step_end`

```python
# Update the status_writer.update_progress call:
self.status_writer.update_progress(
    step=state.global_step,
    total_steps=self.total_steps,
    epoch=int(current_epoch),
    loss=current_loss,
    lr=current_lr,
    batch_step=batch_step,
    batch_total_steps=self.batch_total_steps,
    batch_number=self.batch_number,
    batch_queue_size=self.batch_queue_size,
    current_file=self.current_file,
    val_loss=self.last_val_loss  # ADD THIS
)
```

### Step 6: Update Callback Instantiation

**File:** `train.py` around line 779

```python
# Update the callback creation to pass fixed_val_dataset:
callbacks=[LiveMonitorCallback(
    self.live_monitor,
    self.status_writer,
    self.args.eval_steps,
    total_steps,
    self.raw_train_examples,
    self.tokenizer,
    self.model,
    batch_total_steps=steps_this_batch,
    current_global_step=current_global_step,
    evolution_tracker=self.evolution_tracker,
    current_file=current_file,
    batch_number=batch_number,
    batch_queue_size=batch_queue_size,
    controller=self.controller,
    fixed_val_dataset=self.fixed_val_dataset  # ADD THIS
)]
```

---

## 🎨 UI UPDATES (Optional - Can Do Later)

### Update live_monitor_ui.html

**Add metrics display:**
```html
<!-- Add to metrics panel -->
<div class="metric">
    <div class="label">Validation Loss</div>
    <div class="value" id="val-loss">--</div>
</div>
<div class="metric">
    <div class="label">Train/Val Gap</div>
    <div class="value" id="loss-gap" class="gap-indicator">--</div>
</div>
```

**Update JavaScript:**
```javascript
// In updateMetrics function:
if (data.validation_loss) {
    document.getElementById('val-loss').textContent = data.validation_loss.toFixed(4);

    if (data.val_train_gap !== null) {
        const gap = data.val_train_gap;
        const gapEl = document.getElementById('loss-gap');
        gapEl.textContent = (gap >= 0 ? '+' : '') + gap.toFixed(4);

        // Color code based on gap
        if (gap < 0.3) {
            gapEl.className = 'gap-indicator good';  // Green
        } else if (gap < 0.5) {
            gapEl.className = 'gap-indicator warning';  // Yellow
        } else {
            gapEl.className = 'gap-indicator danger';  // Red - overfitting!
        }
    }
}
```

**Add CSS:**
```css
.gap-indicator.good { color: #10b981; }
.gap-indicator.warning { color: #f59e0b; }
.gap-indicator.danger { color: #ef4444; }
```

---

## 🧪 TESTING

**After implementation:**
1. Restart training daemon
2. Check logs for validation loss messages
3. Verify `training_status.json` has `validation_loss` field
4. Monitor gap - should be small (~0.05-0.2) if not overfitting
5. If gap grows large (>0.5), model is overfitting

**Expected output:**
```
📊 Loss Comparison: Train=0.1500 | Val=0.1800 | Gap=+0.0300 ✅
```

---

## 📊 INTERPRETATION GUIDE

| Train/Val Gap | Status | Action |
|---------------|--------|--------|
| < 0.2 | ✅ Excellent | Keep training |
| 0.2 - 0.4 | ✅ Good | Monitor |
| 0.4 - 0.6 | ⚠️ Warning | Consider stopping soon |
| > 0.6 | 🚨 Overfitting | Stop training or add data |

---

## 📁 FILES MODIFIED

- `train.py` - Added validation loader, callback updates
- `data/validation/syllo_validation_1000.jsonl` - Fixed validation set
- `training_status.py` - Will add validation_loss fields
- `live_monitor_ui.html` - Will add validation display (optional)

---

## 🚀 QUICK START (After Implementation)

Just restart training - validation will happen automatically!

```bash
# Training will now show:
# Training Loss: 0.15  (on data it's seeing)
# Validation Loss: 0.18  (on unseen data - TRUE performance)
```

---

**Implementation Time:** ~30 minutes
**Testing Time:** ~10 minutes
**Total:** ~40 minutes to complete

---

END OF PLAN
