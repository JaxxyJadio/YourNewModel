# ModelFactory.md

## ModelFactory: Standard Model Size Guidelines

This document provides recommended settings for common model sizes when using `ModelFactory.py` to train your own dense LLM. Adjust as needed for your hardware and dataset.

---

## Standard Model Sizes

| Model Size | Layers | Hidden Dim | Attention Heads | Vocab Size | Max Length | Batch Size | Training Steps | Notes |
|------------|--------|------------|-----------------|------------|------------|------------|----------------|-------|
| Small      | 6      | 512        | 8               | 8,000      | 256        | 16         | 10,000         | Fast prototyping |
| Base       | 12     | 768        | 12              | 16,000     | 256/512    | 8-16       | 25,000         | GPT-2 base scale |
| Large      | 24     | 1024       | 16              | 32,000     | 512        | 4-8        | 50,000         | Requires more VRAM |
| XL         | 32     | 1600       | 20              | 50,000     | 1024       | 2-4        | 100,000        | Multi-GPU recommended |

---

## Recommended Parameter Ranges

- **Vocab Size:** 8,000–50,000 (depends on language diversity)
- **Max Length:** 256–1024 (longer = more memory)
- **Batch Size:** As large as fits in GPU RAM (use gradient accumulation if needed)
- **Training Steps:** 10,000+ (more for larger models)
- **Learning Rate:** 1e-4 to 5e-4 (start conservative)
- **Mixed Precision:** Yes (recommended for speed/memory)

---

## Example: Training a Base Model

```
python src/YourNewModel/ModelFactory.py
# Use these when prompted:
# - Model size: Base
# - Vocab size: 16000
# - Max length: 256
# - Batch size: 8
# - Training steps: 25000
# - Learning rate: 5e-4
# - Use mixed precision: y
```

---

## Tips
- Always match tokenizer vocab size and max length to your model config.
- For large models, use distributed/multi-GPU training.
- Save configs for reproducibility.
- Monitor GPU memory and adjust batch size as needed.
- You can resume training from checkpoints in the output directory.

---

## System Hardware Guide: What Model Size Should You Pick?

| GPU (VRAM)         | CPU Cores/Threads | Recommended Model Size | Notes |
|--------------------|-------------------|-----------------------|-------|
| 4–6GB (e.g. GTX 1650, RTX 3050) | 4–8 | Small                | Use batch size 4–8, max length ≤256 |
| 8GB (e.g. RTX 4060, 3070, A4000) | 8–20 | Base                 | Batch size 8–16, max length 256–512 |
| 12–16GB (e.g. RTX 3080, 4070, 3090, A5000) | 12–32 | Large        | Batch size 8+, max length 512 |
| 24GB+ (e.g. RTX 4090, A6000, H100, multi-GPU) | 16+ | XL or bigger | Use multi-GPU for best results |

- **If you have a modern CPU (Intel i7/i9, AMD Ryzen 7/9, Threadripper, Xeon, EPYC):** You are not CPU-limited for any single-GPU training. More cores = faster data loading and preprocessing.
- **If you have less VRAM than recommended:** Lower batch size and/or max length, or use gradient accumulation.
- **If you have multiple GPUs:** Use distributed training for Large/XL models.

### Example Recommendations
- **RTX 4060 + i7-14700F:** Use Base (12L, 768H, 16k vocab, 256–512 max length, batch 8–16)
- **RTX 4090 + Ryzen 9:** Use Large or XL (24–32L, 1024–1600H, 32k–50k vocab, 512–1024 max length)
- **GTX 1650 + i5:** Use Small (6L, 512H, 8k vocab, 256 max length, batch 4–8)

---

For more details, see the comments in `ModelFactory.py` or your project README.
