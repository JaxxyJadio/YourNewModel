"""
Training script for YourNewModel with improved tokenizer and data handling.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import json
import sys
from typing import Dict, List, Optional
from tqdm import tqdm
import yaml

# Add the package root to the path for relative imports
current_dir = os.path.dirname(os.path.abspath(__file__))
package_root = os.path.abspath(os.path.join(current_dir, '../../..'))
sys.path.insert(0, package_root)

from YourNewModel.modelling_YourNewModel import YourNewModel, YourNewModelConfig, create_model

# Import our new tokenizer and data utilities
from YourNewModel.tokenizer_YourNewModel import BPETokenizer, YourNewModelTokenizer, create_tokenizer
from YourNewModel.training.YourNewModel.data.gsm8k_loader import GSM8KDataset, create_data_loaders, prepare_gsm8k_for_tokenizer_training


class TextDataset(Dataset):
    """Simple text dataset for training."""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer.encode(text)[:self.max_length]
        
        # Pad if necessary
        if len(tokens) < self.max_length:
            tokens.extend([0] * (self.max_length - len(tokens)))
        
        return torch.tensor(tokens, dtype=torch.long)


def create_tokenizer(tokenizer_type: str = "bpe", vocab_size: int = 175000, train_texts: Optional[List[str]] = None) -> object:
    """Create and optionally train a tokenizer."""
    if tokenizer_type == "bpe":
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        if train_texts:
            tokenizer.train(train_texts)
    else:
        tokenizer = YourNewModelTokenizer(vocab_size=vocab_size)
    
    return tokenizer


def load_all_datasets(paths: List[str]) -> List[Dict[str, str]]:
    """
    Load and combine multiple JSONL datasets into a single list.
    Args:
        paths (List[str]): List of file paths to JSONL datasets.
    Returns:
        List[Dict[str, str]]: Combined list of dataset entries.
    """
    data = []
    for path in paths:
        if not os.path.exists(path):
            print(f"Warning: Dataset not found: {path}")
            continue
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line.strip())
                        data.append(entry)
                    except Exception as e:
                        print(f"Error parsing line in {path}: {e}")
    return data


class Trainer:
    """Training class for the Dense LLM."""
    
    def __init__(
        self,
        model: YourNewModel,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        batch_size: int = 4,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 10000,
        eval_interval: int = 500,
        save_interval: int = 1000,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.checkpoint_dir = checkpoint_dir
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Setup data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )
        
        if val_dataset:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        
        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=self._lr_lambda,
        )
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Move model to device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {self.model.get_num_params():,}")
    
    def _lr_lambda(self, step):
        """Learning rate schedule with warmup."""
        if step < self.warmup_steps:
            return step / self.warmup_steps
        return 1.0
    
    def train_step(self, batch):
        """Single training step."""
        self.model.train()
        
        # Move batch to device
        input_ids = batch.to(self.device)
        
        # Create labels (shifted input for next token prediction)
        labels = input_ids[:, 1:].contiguous()
        input_ids = input_ids[:, :-1].contiguous()
        
        # Forward pass
        logits = self.model(input_ids)
        
        # Compute loss
        loss = nn.functional.cross_entropy(
            logits["logits"].view(-1, logits["logits"].size(-1)),
            labels.view(-1),
            ignore_index=0,  # Ignore padding tokens
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Update weights
        self.optimizer.step()
        self.scheduler.step()
        
        return loss.item()
    
    def evaluate(self):
        """Evaluate the model on validation set."""
        if not self.val_dataset:
            return None
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch.to(self.device)
                labels = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1].contiguous()
                
                logits = self.model(input_ids)
                loss = nn.functional.cross_entropy(
                    logits["logits"].view(-1, logits["logits"].size(-1)),
                    labels.view(-1),
                    ignore_index=0,
                )
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'step': self.step,
            'epoch': self.epoch,
            'config': self.model.config,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        print(f"Checkpoint loaded from {path}")
    
    def train(self):
        """Main training loop."""
        print("Starting training...")
        
        train_iter = iter(self.train_loader)
        pbar = tqdm(total=self.max_steps, desc="Training")
        
        while self.step < self.max_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                batch = next(train_iter)
                self.epoch += 1
            
            # Training step
            loss = self.train_step(batch)
            self.step += 1
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'loss': f"{loss:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
                'epoch': self.epoch,
            })
            
            # Evaluation
            if self.step % self.eval_interval == 0:
                val_loss = self.evaluate()
                if val_loss is not None:
                    print(f"\nStep {self.step}: Train Loss = {loss:.4f}, Val Loss = {val_loss:.4f}")
                    
                    # Save best model
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(os.path.join(self.checkpoint_dir, "best_model.pt"))
                else:
                    print(f"\nStep {self.step}: Train Loss = {loss:.4f}")
            
            # Save checkpoint
            if self.step % self.save_interval == 0:
                self.save_checkpoint(os.path.join(self.checkpoint_dir, f"checkpoint_step_{self.step}.pt"))
        
        pbar.close()
        print("Training completed!")
        
        # Save final model
        self.save_checkpoint(os.path.join(self.checkpoint_dir, "final_model.pt"))


def create_sample_dataset(num_samples: int = 1000, seq_length: int = 100) -> List[str]:
    """Create a simple sample dataset for testing."""
    import random
    import string
    
    texts = []
    for _ in range(num_samples):
        # Generate random text
        text = ''.join(random.choices(string.ascii_letters + string.digits + ' ', k=seq_length))
        texts.append(text)
    
    return texts


def prepare_training_texts_rich(data: List[Dict[str, str]]) -> List[str]:
    """
    Prepare a list of training texts supporting input, output, instruction, text, question, answer, and code fields.
    Args:
        data (List[Dict[str, str]]): List of dataset entries.
    Returns:
        List[str]: List of formatted training strings.
    """
    texts = []
    for entry in data:
        # Extract all possible fields
        instruction = entry.get('instruction', '')
        input_ = entry.get('input', '')
        output = entry.get('output', '')
        text = entry.get('text', '')
        question = entry.get('question', '')
        answer = entry.get('answer', '')
        code = entry.get('code', '')
        # Compose a rich prompt
        formatted = ''
        if instruction:
            formatted += f"Instruction: {instruction}\n"
        if input_:
            formatted += f"Input: {input_}\n"
        if question:
            formatted += f"Question: {question}\n"
        if text:
            formatted += f"Text: {text}\n"
        if code:
            formatted += f"Code: {code}\n"
        if answer:
            formatted += f"Answer: {answer}\n"
        if output:
            formatted += f"Output: {output}\n"
        # Fallback: if nothing, skip
        if not formatted.strip():
            continue
        texts.append(formatted.strip())
        # Also add individual fields for variety
        for field in [instruction, input_, question, text, code, answer, output]:
            if field:
                texts.append(field)
    return texts


class FlexibleQADataset(Dataset):
    """
    Dataset for QA/instruction data supporting input, output, instruction, text, question, answer, and code fields with fallback logic.
    """
    def __init__(self, data: List[Dict[str, str]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.fields_priority = [
            'instruction', 'input', 'question', 'text', 'code', 'answer', 'output'
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        # Try to build a prompt using all available fields, fallback to any present
        prompt = ''
        for field in self.fields_priority:
            value = entry.get(field, None)
            if value:
                prompt += f"{field.capitalize()}: {value}\n"
        if not prompt.strip():
            # Fallback: try to use the whole entry as string
            prompt = str(entry)
        tokens = self.tokenizer.encode(prompt)[:self.max_length]
        if len(tokens) < self.max_length:
            tokens.extend([0] * (self.max_length - len(tokens)))
        return torch.tensor(tokens, dtype=torch.long)


if __name__ == "__main__":
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "..", "..", "YourNewModel", "config_YourNewModel.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    tokenizer_cfg = config.get('tokenizer', {})
    model_cfg = config.get('model', {})
    data_cfg = config.get('data', {})

    # Set vocab size from config
    vocab_size = model_cfg.get('vocab_size', 175000)

    # Load all datasets
    dataset_paths = [
        os.path.join(os.path.dirname(__file__), 'data', 'Test-gsm8k.jsonl'),
        os.path.join(os.path.dirname(__file__), 'data', 'Train-gsm8k.jsonl'),
        os.path.join(os.path.dirname(__file__), 'data', 'YourNewModel-Dataset.jsonl'),
    ]
    all_data = load_all_datasets(dataset_paths)
    if not all_data:
        raise FileNotFoundError("No data loaded from any dataset files.")

    # Prepare texts for tokenizer
    from YourNewModel.tokenizer_YourNewModel import prepare_training_texts_from_gsm8k
    texts = prepare_training_texts_rich(all_data)

    # Create tokenizer (optionally train on the dataset)
    tokenizer = YourNewModelTokenizer(vocab_size=vocab_size)
    # Optionally, train tokenizer here if needed
    # tokenizer.train(texts)

    # Create datasets
    from YourNewModel.training.YourNewModel.data.gsm8k_loader import GSM8KDataset
    train_dataset = FlexibleQADataset(all_data, tokenizer, max_length=64)
    val_dataset = None  # Optionally split or use a separate validation set

    # Create model
    from YourNewModel.modelling_YourNewModel import create_model
    model = create_model("small")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=4,
        learning_rate=3e-4,
        max_steps=1000,
        eval_interval=100,
        save_interval=500,
    )

    # Train model
    trainer.train()

    # Test generation
    print("\nTesting generation...")
    model.eval()
    test_input = torch.randint(0, vocab_size, (1, 10))
    generated = model.generate(test_input, max_length=50)
    print(f"Generated tokens: {generated[0].tolist()}")
    print(f"Generated text: {tokenizer.decode(generated[0].tolist())}")
