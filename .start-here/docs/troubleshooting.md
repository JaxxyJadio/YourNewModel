# Troubleshooting & FAQ

## Common Issues & Solutions

### 1. CUDA Out of Memory Errors
- **What it means:** Your GPU ran out of memory during training.
- **How to fix:**
  - Lower the batch size in your config.
  - Reduce max sequence length.
  - Use mixed precision (enables more efficient memory use).
  - Try gradient accumulation to simulate a larger batch size with less memory.

### 2. Tokenizer/Model Config Mismatch
- **What it means:** The model and tokenizer disagree on vocab size or max length.
- **How to fix:**
  - Make sure the vocab size and max length in your model config match those used to train the tokenizer.
  - If you retrain the tokenizer, update the model config and restart training.

### 3. Training is Slow or Unstable
- **What it means:** Training is taking too long, or loss is not decreasing.
- **How to fix:**
  - Check your hardware usage (GPU, CPU, disk).
  - Lower model size or batch size.
  - Make sure your data is clean and not too repetitive.
  - Try a smaller learning rate if loss is unstable.

### 4. Checkpoint/Resume Issues
- **What it means:** Training did not resume from the last checkpoint.
- **How to fix:**
  - Ensure the checkpoint files exist in the `checkpoints/` directory.
  - Use the resume option in `ModelFactory.py` or your training script.
  - If resuming fails, try starting from the latest checkpoint manually.

---

## How to Resume Training

1. Find your latest checkpoint in the `checkpoints/` folder (e.g., `checkpoint_step_500.pt`).
2. When running `ModelFactory.py` or your training script, select the option to resume and provide the checkpoint path if prompted.
3. Training will continue from where it left off.

---

## More Help
- Check the logs for detailed error messages (look for `.log` files or console output).
- If you’re stuck, ask for help or consult the other docs in `.start-here/docs/`.
- Don’t be afraid to start over—mistakes are part of the process!
