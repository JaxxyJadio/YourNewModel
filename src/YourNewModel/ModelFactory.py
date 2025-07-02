import os
import sys
import yaml
import subprocess
from pathlib import Path

def main():
    print("=== ModelFactory: LLM Configurator (YAML-based) ===\n")
    config_path = Path(__file__).parent / "config_YourNewModel.yaml"
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"Loaded config from {config_path}\n")

    # --- Tokenizer training ---
    print("[1/3] Training tokenizer...")
    tok_cmd = [
        sys.executable, str(Path('src/YourNewModel/tokenizer_YourNewModel.py')),
        "--config", str(config_path)
    ]
    print(f"Running: {' '.join(tok_cmd)}")
    subprocess.run(tok_cmd, check=True)

    # --- Model training ---
    print("[2/3] Starting model training...")
    train_cmd = [
        sys.executable, str(Path('src/YourNewModel/training/YourNewModel/train_YourNewModel.py')),
        "--config", str(config_path)
    ]
    print(f"Running: {' '.join(train_cmd)}")
    subprocess.run(train_cmd, check=True)

    print("\n[3/3] Training complete. Model weights and checkpoints saved.")
    print("\nYou can resume or evaluate your model using the saved config and weights.")

if __name__ == "__main__":
    main()
