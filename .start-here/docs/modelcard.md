---
tags:
- language-model
- transformer
- pytorch
- custom-llm
license: apache-2.0
datasets:
- All deduplicated JSONL datasets (736,671+ unique samples)
model-index:
- name: YourNewModel
  results: []
---

# YourNewModel

## Model Description

YourNewModel is a dense transformer-based language model designed for research, prototyping, and downstream NLP tasks. The model is modular, extensible, and supports efficient training on a range of hardware.

- **Architecture:** Dense Transformer (24 layers, 6144 hidden, 24 heads, 12,088 max sequence length)
- **Tokenizer:** BPE, custom implementation, trained on 736,671+ deduplicated samples
- **Framework:** PyTorch
- **Trained on:** All deduplicated JSONL datasets (736,671+ samples), including:
  - English question-answer pairs
  - Large amounts of Python, C#, and HTML code
  - Mixed-format and multi-domain data

> **Note:** The tokenizer is now fully trained and integrated. This model card reflects the latest model and data pipeline.

## Intended Uses & Limitations

**Intended Uses:**
- Text and code generation, completion, and summarization
- Research and experimentation
- Fine-tuning for downstream tasks (including code understanding and generation)

**Limitations:**
- May not generalize well outside the training domain
- Not suitable for production without further evaluation
- Outputs may contain biases present in the training data

## Training Details

- **Data:** All deduplicated JSONL datasets (736,671+ samples, including code and text)
- **Vocab Size:** 175,000
- **Max Sequence Length:** 12,088
- **Model Size:** 24 layers, 6144 hidden, 24 heads
- **Batch Size:** 4 (configurable)
- **Steps:** 10,000+ (configurable)
- **Optimizer:** AdamW
- **Learning Rate:** 3e-4 (default)
- **Mixed Precision:** Yes (torch.cuda.amp)
- **Checkpointing:** Enabled
- **Parameter Count:** ~2.2 billion (see below)

### Parameter Count Calculation

For a standard transformer decoder block:

- Embedding: `vocab_size * n_embd` = 175,000 × 6,144 = 1,075,200,000
- Each layer (approx):
  - Attention: `3 × n_embd × n_embd` (QKV) + `n_embd × n_embd` (output) = 4 × 6,144² = 150,994,944
  - MLP: `n_embd × (4 × n_embd)` (in) + `(4 × n_embd) × n_embd` (out) = 2 × 6,144 × 24,576 = 302,330,880
  - LayerNorms: negligible
  - Total per layer ≈ 453,325,824
- 24 layers: 24 × 453,325,824 ≈ 10,879,819,776
- Final LM head: `n_embd × vocab_size` = 6,144 × 175,000 = 1,075,200,000
- **Total (approx):** 1.1B (embed) + 10.9B (layers) + 1.1B (head) ≈ **13.1 billion**

> Actual parameter count may be lower due to weight sharing, bias terms, and optimizations. For your config, expect **~13B parameters**.

## Evaluation

- **Metrics:** <e.g., Perplexity, accuracy, etc.>
- **Validation Data:** <describe or link>
- **Results:** <add results when available>

## Ethical Considerations

- The model may reflect biases in the training data.
- Not intended for use in critical or high-stakes applications without further validation.
- Users are responsible for evaluating outputs for fairness and safety.

## Citation

If you use this model or codebase, please cite:

```
@misc{yournewmodel2025,
  title={YourNewModel: A Modular Dense Transformer LLM},
  author={Your Name},
  year={2025},
  url={https://github.com/yourname/yournewmodel}
}
```

## Contact

For questions or support, please open an issue or contact <your-email>.
