"""
Quick test script to verify the Dense LLM implementation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from src.YourNewModel.modelling_YourNewModel import YourNewModelConfig, YourNewModel
from src.YourNewModel.tokenizer_YourNewModel import YourNewModelTokenizer
from src.YourNewModel.training.YourNewModel.train_YourNewModel import TextDataset, Trainer


def test_model_creation():
    """Test model creation and parameter counting."""
    print("=== Testing Model Creation ===")
    
    for size in ["small", "medium", "large"]:
        model = YourNewModel(config=YourNewModelConfig())
        num_params = model.get_num_params()
        print(f"{size.capitalize()} model: {num_params:,} parameters")
    
    print("✓ Model creation successful\n")


def test_forward_pass():
    """Test forward pass with different input sizes."""
    print("=== Testing Forward Pass ===")
    
    model = YourNewModel(config=YourNewModelConfig())
    model.eval()
    
    test_cases = [
        (1, 10),   # Single sequence, short
        (2, 50),   # Batch of 2, medium
        (4, 100),  # Batch of 4, long
    ]
    
    for batch_size, seq_len in test_cases:
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        logits = outputs["logits"]
        expected_shape = (batch_size, seq_len, model.config.vocab_size)
        assert logits.shape == expected_shape, f"Expected {expected_shape}, got {logits.shape}"
        
        print(f"✓ Forward pass [{batch_size}, {seq_len}] -> {logits.shape}")
    
    print("✓ Forward pass tests successful\n")


def test_generation():
    """Test text generation."""
    print("=== Testing Text Generation ===")
    
    model = YourNewModel(config=YourNewModelConfig())
    model.eval()
    
    # Test generation with different parameters
    input_ids = torch.randint(0, 1000, (1, 5))
    
    test_cases = [
        {"max_length": 20, "temperature": 1.0},
        {"max_length": 30, "temperature": 0.8},
        {"max_length": 15, "temperature": 1.5},
    ]
    
    for i, params in enumerate(test_cases):
        generated = model.generate(input_ids, **params)
        print(f"✓ Generation test {i+1}: {input_ids.shape} -> {generated.shape}")
        assert generated.shape[1] == params["max_length"], "Generation length mismatch"
    
    print("✓ Generation tests successful\n")


def test_tokenizer():
    """Test the simple tokenizer."""
    print("=== Testing Tokenizer ===")
    
    tokenizer = YourNewModelTokenizer(vocab_size=256)
    
    test_texts = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "🚀 Testing special characters 123",
    ]
    
    for text in test_texts:
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"Original: '{text}'")
        print(f"Tokens: {tokens[:10]}..." if len(tokens) > 10 else f"Tokens: {tokens}")
        print(f"Decoded: '{decoded}'")
        print()
    
    print("✓ Tokenizer tests successful\n")


def test_training_components():
    """Test training-related components."""
    print("=== Testing Training Components ===")
    
    # Create sample data
    texts = ["Hello world" * 10, "Testing training" * 8, "Dense LLM model" * 12]
    tokenizer = YourNewModelTokenizer(vocab_size=256)
    
    # Create dataset
    dataset = TextDataset(texts, tokenizer, max_length=50)
    print(f"✓ Dataset created with {len(dataset)} samples")
    
    # Test dataset item
    sample = dataset[0]
    print(f"✓ Sample shape: {sample.shape}")
    
    # Create small model for training test
    config = YourNewModelConfig(
        vocab_size=256,
        n_positions=64,
        n_embd=128,
        n_layer=2,
        n_head=4,
        n_inner=512,
    )
    model = YourNewModel(config)
    
    print(f"✓ Training model created: {model.get_num_params():,} parameters")
    print("✓ Training components successful\n")


def test_memory_usage():
    """Test memory usage with different model sizes."""
    print("=== Testing Memory Usage ===")
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Testing on GPU: {torch.cuda.get_device_name()}")
        
        for size in ["small"]:  # Only test small to avoid memory issues
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            model = YourNewModel(config=YourNewModelConfig()).to(device)
            input_ids = torch.randint(0, 1000, (2, 100), device=device)
            
            with torch.no_grad():
                logits = model(input_ids)
            
            memory_used = torch.cuda.max_memory_allocated() / 1024**2
            print(f"✓ {size.capitalize()} model GPU memory: {memory_used:.1f} MB")
            
            del model, logits
            torch.cuda.empty_cache()
    else:
        print("CUDA not available, skipping GPU memory tests")
    
    print("✓ Memory usage tests completed\n")


def test_model_compatibility():
    """Test model saving and loading."""
    print("=== Testing Model Save/Load ===")
    
    import tempfile
    import os
    
    model = YourNewModel(config=YourNewModelConfig())
    
    # Save model
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        checkpoint_path = f.name
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': model.config,
    }
    torch.save(checkpoint, checkpoint_path)
    print("✓ Model saved")
    
    # Load model
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    loaded_model = YourNewModel(config=checkpoint['config'])
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    print("✓ Model loaded")
    
    # Test equivalence
    input_ids = torch.randint(0, 1000, (1, 20))
    
    with torch.no_grad():
        original_output = model(input_ids)["logits"]
        loaded_output = loaded_model(input_ids)["logits"]
    
    assert torch.allclose(original_output, loaded_output, atol=1e-6), "Model outputs don't match"
    print("✓ Model outputs match after save/load")
    
    # Clean up
    os.unlink(checkpoint_path)
    print("✓ Save/load tests successful\n")


def main():
    """Run all tests."""
    print("🚀 Starting Tests: Your New Model is coming to life!\n")

    try:
        test_model_creation()
        test_forward_pass()
        test_generation()
        test_tokenizer()
        test_training_components()
        test_memory_usage()
        test_model_compatibility()
        
        print("🎉 All tests passed successfully!")
        print("\nYour Dense LLM implementation is ready to use!")
        print("\nNext steps:")
        print("1. Train the model: python train.py")
        print("2. Generate text: python generate.py --checkpoint checkpoints/final_model.pt --prompt 'Hello world'")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
