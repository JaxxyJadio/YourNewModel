"""
Script to generate a new randomly initialized YourNewModel using config and tokenizer files.
- Loads model and tokenizer config (YAML/JSON)
- Instantiates model with random weights
- Saves model weights to disk
"""
import os
import sys
import argparse
import yaml
import json
import torch
from pathlib import Path
from modelling_YourNewModel import YourNewModel, YourNewModelConfig
from tokenizer_YourNewModel import YourNewModelTokenizer, BPETokenizer


def load_tokenizer(tokenizer_cfg):
    vocab_file = tokenizer_cfg.get('vocab_file', 'vocab.json')
    vocab_file = str(Path(vocab_file))
    if not Path(vocab_file).exists():
        raise FileNotFoundError(f"Tokenizer vocab file not found: {vocab_file}")
    with open(vocab_file, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    if vocab_data.get('type') == 'simple':
        tokenizer = YourNewModelTokenizer(vocab_size=vocab_data['vocab_size'])
        tokenizer.load(vocab_file)
    else:
        tokenizer = BPETokenizer(vocab_size=vocab_data.get('vocab_size', 50257))
        tokenizer.load(vocab_file)
    return tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate a new randomly initialized YourNewModel.")
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    parser.add_argument('--output', type=str, default='YourNewModel.pt', help='Path to save model weights')
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    model_cfg = config.get('model', {})
    tokenizer_cfg = config.get('tokenizer', {})

    # Load tokenizer (for vocab size)
    tokenizer = load_tokenizer(tokenizer_cfg)
    vocab_size = model_cfg.get('vocab_size', len(tokenizer.token_to_id))

    # Build model config dataclass
    model_config = YourNewModelConfig(
        vocab_size=vocab_size,
        n_positions=model_cfg.get('max_position_embeddings', 1024),
        n_embd=model_cfg.get('hidden_dim', 768),
        n_layer=model_cfg.get('num_layers', 12),
        n_head=model_cfg.get('num_heads', 12),
        n_inner=None,
        activation_function=model_cfg.get('activation', 'gelu_new'),
        attn_pdrop=model_cfg.get('dropout', 0.1),
        embd_pdrop=model_cfg.get('dropout', 0.1),
        resid_pdrop=model_cfg.get('dropout', 0.1),
        layer_norm_epsilon=model_cfg.get('layer_norm_epsilon', 1e-5),
        initializer_range=0.02,
        use_cache=True,
        scale_attn_weights=True,
        scale_attn_by_inverse_layer_idx=False,
        reorder_and_upcast_attn=False
    )

    # Instantiate model
    model = YourNewModel(model_config)
    # Model weights are randomly initialized by default

    # Save model weights
    torch.save(model.state_dict(), args.output)
    print(f"Randomly initialized model saved to {args.output}")

if __name__ == "__main__":
    main()
