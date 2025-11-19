# Inbox Flattening Feature - Implementation Summary

**Date:** 2025-11-11
**Status:** ✅ Implemented and Tested

## Overview
Added automatic subdirectory flattening to the training daemon. The daemon now automatically extracts `.jsonl` files from subdirectories in `inbox/` and moves them to the root level before training.

## Problem Solved
Previously, the training daemon only scanned `inbox/*.jsonl` at the root level. When LEO outputs were copied as complete directories (e.g., `inbox/20251108T044715Z_dde784/training_samples.jsonl`), the daemon couldn't find them.

## Solution Implemented
Added a `flatten_inbox()` method to the `TrainingDaemon` class that:

1. **Scans** for `.jsonl` files in subdirectories recursively
2. **Moves** files to inbox root with unique naming
3. **Handles** naming collisions (appends counter if needed)
4. **Cleans up** empty subdirectories after moving files
5. **Logs** all actions for visibility

## Technical Details

### Code Changes
**File:** `training_daemon.py`

#### New Method Added (lines 194-240):
```python
def flatten_inbox(self):
    """Move any .jsonl files from subdirectories to inbox root"""
    # Find all .jsonl files in subdirectories (not at root)
    subdirs = [d for d in self.inbox_dir.iterdir() if d.is_dir()]

    if not subdirs:
        return  # No subdirectories, nothing to do

    moved_count = 0

    for subdir in subdirs:
        # Find all .jsonl files recursively in this subdir
        jsonl_files = list(subdir.rglob("*.jsonl"))

        for jsonl_file in jsonl_files:
            # Construct new filename at inbox root
            # Use subdirectory name to make it unique
            subdir_name = subdir.name
            original_name = jsonl_file.stem  # filename without extension
            new_name = f"{original_name}_{subdir_name}.jsonl"
            dest_path = self.inbox_dir / new_name

            # Handle collision (if file already exists at destination)
            counter = 1
            while dest_path.exists():
                new_name = f"{original_name}_{subdir_name}_{counter}.jsonl"
                dest_path = self.inbox_dir / new_name
                counter += 1

            # Move file
            try:
                shutil.move(str(jsonl_file), str(dest_path))
                self.logger.info(f"📁 Flattened: {jsonl_file.relative_to(self.inbox_dir)} → {new_name}")
                moved_count += 1
            except Exception as e:
                self.logger.error(f"Failed to move {jsonl_file}: {e}")

        # Clean up empty subdirectory
        try:
            if subdir.exists() and not any(subdir.iterdir()):
                subdir.rmdir()
                self.logger.info(f"🗑️  Removed empty subdir: {subdir.name}")
        except Exception as e:
            self.logger.warning(f"Could not remove subdir {subdir.name}: {e}")

    if moved_count > 0:
        self.logger.info(f"✅ Flattened {moved_count} file(s) from subdirectories")
```

#### Integration Point (line 380-381):
```python
# Flatten inbox (move .jsonl files from subdirs to root)
self.flatten_inbox()

# Check inbox for data files
inbox_files = self.get_inbox_files()
```

The `flatten_inbox()` method is called **before** `get_inbox_files()` in each poll cycle, ensuring subdirectories are flattened before training begins.

## Naming Strategy

### Example Transformations:
```
Before:
  inbox/20251108T044715Z_abc123/training_samples.jsonl
  inbox/20251109T120000Z_def456/training_samples.jsonl

After:
  inbox/training_samples_20251108T044715Z_abc123.jsonl
  inbox/training_samples_20251109T120000Z_def456.jsonl
```

### Collision Handling:
If a file with the generated name already exists, a counter is appended:
```
training_samples_20251108T044715Z_abc123.jsonl
training_samples_20251108T044715Z_abc123_1.jsonl
training_samples_20251108T044715Z_abc123_2.jsonl
```

## Testing Results

### Test Scenario 1: Single Subdirectory
```bash
# Created: inbox/test_subdir_20251108T123456Z/training_samples.jsonl
# Result: inbox/training_samples_test_subdir_20251108T123456Z.jsonl
# Status: ✅ Subdirectory removed
```

### Test Scenario 2: Multiple Subdirectories with Same Filename
```bash
# Created:
#   inbox/20251108T044715Z_abc123/training_samples.jsonl
#   inbox/20251109T120000Z_def456/training_samples.jsonl

# Result:
#   inbox/training_samples_20251108T044715Z_abc123.jsonl
#   inbox/training_samples_20251109T120000Z_def456.jsonl

# Status: ✅ Both subdirectories removed
# Content: ✅ Verified intact
```

## User Benefits

### Before (Manual Process):
```bash
# Copy LEO output
cp -r /path/to/leo/outputs/20251108_xxxxx inbox/

# Manually extract .jsonl file
mv inbox/20251108_xxxxx/training_samples.jsonl inbox/
rm -rf inbox/20251108_xxxxx/
```

### After (Automatic):
```bash
# Just copy entire directory - daemon handles the rest!
cp -r /path/to/leo/outputs/20251108_xxxxx inbox/

# Daemon automatically:
# 1. Finds training_samples.jsonl in subdirectory
# 2. Moves to inbox/training_samples_20251108_xxxxx.jsonl
# 3. Removes empty subdirectory
# 4. Trains on the file
```

## Documentation Updates

Updated `CLAUDE.md` sections:
1. **Step 1: Prepare Training Data** - Added note about auto-flattening
2. **Auto-Flattening Inbox** - New section explaining the feature
3. **LEO Training Data Outputs** - Simplified copy commands
4. **Notes for Claude** - Added feature announcement

## Edge Cases Handled

1. **No subdirectories** - Function returns immediately
2. **Naming collisions** - Appends counter to filename
3. **Empty subdirectories** - Cleaned up automatically
4. **Permission errors** - Logged but don't crash daemon
5. **Multiple .jsonl files in one subdir** - All moved with unique names

## Performance Impact

- **Minimal** - Only scans subdirectories once per poll cycle
- **Fast operation** - File moves are atomic
- **No blocking** - Happens before training, no impact on GPU usage

## Future Enhancements (Optional)

Potential improvements if needed:
1. Configuration option to disable flattening
2. Preserve original subdirectory structure in filename (nested paths)
3. Support for other file extensions (.json, .txt)
4. Archive subdirectories instead of deleting

## Rollback Instructions

If issues arise, revert with:
```bash
git checkout HEAD~1 training_daemon.py
# Or remove the flatten_inbox() call on line 381
```

## Summary

✅ **Implementation complete**
✅ **Tested successfully**
✅ **Documentation updated**
✅ **No breaking changes**
✅ **Backward compatible** (files at root still work as before)

Users can now copy entire LEO output directories to inbox without manual extraction!
