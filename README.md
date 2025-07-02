# Quick Start for First-Time LLM Builders

Welcome! If you’ve never built a language model before, don’t worry—this guide will walk you through the process step by step. You don’t need any prior LLM experience, just basic Python skills and a working GPU setup.

**What you’ll do:**
1. **Prepare your data** (clean up your text files)
2. **Train a tokenizer** (teaches the model how to read text)
3. **Configure and train your model** (choose size/settings, then let it learn)
4. **Test your model** (see what it can do!)

---

## Step-by-Step Checklist

1. **Get your data ready**
   - Put your text data (one sentence or paragraph per line) in the `data/` folder.
   - If you have messy data, use the cleaning script or ask for help (see `data_preparation.md`).

2. **Train the tokenizer**
   - Open a terminal and run:
     ```
     python src/YourNewModel/ModelFactory.py
     ```
   - Choose the option to train a tokenizer. Follow the prompts (vocab size, etc.).
   - This creates `vocab.json` and `merges.txt`—these are needed for the model.

3. **Configure and train your model**
   - Run `ModelFactory.py` again and choose to configure/train the model.
   - Pick a model size that fits your GPU (see the hardware guide in `ModelFactory.md`).
   - Training will start and save progress in the `checkpoints/` folder.

4. **Test your model**
   - Once training finishes (or after a checkpoint), use the inference script (`talkto_YourNewModel.py`) to chat with your model or run sample prompts.
   - See `evaluation.md` for more ways to test.

---

**Tips:**
- If you get stuck, check `troubleshooting.md` or ask for help.
- You can stop and resume training at any time—your progress is saved.
- Don’t worry about making mistakes! You can always start over or try different settings.

---

# Flow Guide: How the Project Files Work Together

This guide explains how each major file and module in your project fits into the overall workflow, from data preparation to model training and inference.

---

## Visual Overview

```
Raw Data (.jsonl, .txt, etc.)
      |
      v
[Data Cleaning/Preparation]
      |
      v
Cleaned Data (.jsonl)  --->  [Tokenizer Training] <---+ 
      |                        |                      |
      |                        v                      |
      +------------------> vocab.json, merges.txt ----+
      |                        |
      v                        v
[Data Loader] <---------- [Model Training Config]
      |                        |
      v                        v
[Model Training (Transformer)]
      |
      v
Checkpoints, Logs, Trained Model
      |
      v
[Evaluation / Inference]
```

---

## High-Level Workflow (Expanded)

1. **Data Preparation**
   - Clean and preprocess raw datasets using scripts or manual review.
   - Ensure data is in `.jsonl` format, one text per line.
   - Place cleaned files in `data/`.
   - Tip: Use `gsm8k_loader.py` or your own script for cleaning/validation.

2. **Tokenizer Training**
   - Run `ModelFactory.py` and select the tokenizer training option.
   - Trains a BPE tokenizer on your cleaned data.
   - Outputs `vocab.json` and `merges.txt` (used by both training and inference).
   - Tip: Match vocab size and max length to your model config.

3. **Model Configuration & Training**
   - Use `ModelFactory.py` to interactively set model size, layers, hidden dim, etc.
   - Loads tokenizer and data loader modules automatically.
   - Launches training via `train_YourNewModel.py`.
   - Handles checkpointing, logging, and metrics tracking.
   - Tip: You can resume from checkpoints in the `checkpoints/` directory.

4. **Evaluation & Inference**
   - Evaluate model performance using validation/test sets.
   - Use `talkto_YourNewModel.py` or custom scripts for inference.
   - Tip: See `evaluation.md` and `deployment.md` for more details.

---

## File/Module Roles (Detailed)

- **src/YourNewModel/ModelFactory.py**
  - Interactive CLI for configuring, launching tokenizer training, and model training.
  - Saves configs for reproducibility and easy resumption.
  - Entry point for most users.

- **src/YourNewModel/tokenizer_YourNewModel.py**
  - Implements and trains a HuggingFace-compatible BPE tokenizer.
  - Generates `vocab.json` and `merges.txt` for use in model training and inference.
  - Can be run standalone or via `ModelFactory.py`.

- **src/YourNewModel/training/YourNewModel/data/gsm8k_loader.py**
  - Loads and preprocesses datasets for training.
  - Handles batching, tokenization, and data streaming for large files.
  - Can be adapted for new data formats.

- **src/YourNewModel/training/YourNewModel/train_YourNewModel.py**
  - Main training script for the transformer model.
  - Loads config, tokenizer, and data loader.
  - Handles checkpointing, logging, metrics, and mixed precision.
  - Supports resuming from checkpoints.

- **src/YourNewModel/modelling_YourNewModel.py**
  - Defines the transformer model architecture (layers, attention, etc.).
  - Modular and extensible for different model sizes and research.
  - Implements best practices for initialization, masking, and extensibility.

- **src/YourNewModel/utils_YourNewModel.py**
  - Utility functions for training, logging, checkpointing, and metrics.
  - Used by both training and evaluation scripts.

- **src/YourNewModel/talkto_YourNewModel.py**
  - (If present) Provides scripts or functions for running inference or interactive chat with the trained model.
  - Useful for quick testing and deployment.

- **data/*.jsonl**
  - Cleaned and preprocessed datasets for training and evaluation.
  - Place your own datasets here for custom training.

- **checkpoints/**
  - Directory for saving model checkpoints and best models during training.
  - Use for resuming or selecting the best model.

---

## Example Flow (Step-by-Step)

1. **Prepare Data:**
   - Clean your dataset (e.g., with `gsm8k_loader.py` or a custom script).
   - Place cleaned `.jsonl` files in the `data/` directory.
   - Validate data format (one text per line, UTF-8).
2. **Train Tokenizer:**
   - Run `python src/YourNewModel/ModelFactory.py` and select tokenizer training.
   - Choose vocab size and other options as prompted.
   - Outputs `vocab.json` and `merges.txt` in the appropriate directory.
3. **Configure & Train Model:**
   - Use `ModelFactory.py` to set model parameters (size, layers, batch size, etc.).
   - Launch training; monitor logs and GPU usage.
   - Training uses the tokenizer and data loader modules.
   - Checkpoints and logs are saved automatically in `checkpoints/`.
4. **Evaluate/Infer:**
   - Use evaluation scripts or `talkto_YourNewModel.py` for inference.
   - Optionally, export the model for deployment.

---

## Practical Tips

- Always match tokenizer and model config (vocab size, max length).
- For large models, use distributed/multi-GPU training (see `advanced_usage.md`).
- Monitor GPU memory and adjust batch size as needed.
- Save configs and logs for reproducibility.
- Use checkpoints to resume or select the best model.
- See `troubleshooting.md` for common issues and solutions.

---

For more details, see the individual module docstrings and the other documentation pages in `.start-here/docs/`.
