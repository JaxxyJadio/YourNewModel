"""
Text generation and inference script for YourNewModel.

This script provides functionalities to load a pre-trained YourNewModel
from a checkpoint, generate text based on a given prompt, and run performance benchmarks.
It supports both command-line generation and an interactive mode for continuous
generation.
"""

import torch
import argparse
import time
from typing import Optional, Dict, Any, List

# Use relative imports for package compatibility
from .modelling_YourNewModel import YourNewModel
from .tokenizer_YourNewModel import YourNewModelTokenizer


def load_model(checkpoint_path: str, device: str = "auto") -> YourNewModel:
    """
    Loads a pre-trained LLM from a specified checkpoint path.

    The model configuration is loaded from the checkpoint, and the model
    state dictionary is then loaded into the model. The model is set to
    evaluation mode (`.eval()`) to disable dropout and batch normalization
    updates during inference.

    Args:
        checkpoint_path (str): The file path to the saved model checkpoint.
                               This checkpoint is expected to be a PyTorch
                               dictionary containing 'config' and 'model_state_dict'.
        device (str, optional): The device to load the model onto.
                                'auto' (default) will automatically select 'cuda' if
                                a GPU is available, otherwise 'cpu'.
                                Can also be explicitly 'cpu' or 'cuda'.

    Returns:
        YourNewModel: The loaded LLM instance, ready for inference.

    Raises:
        FileNotFoundError: If the checkpoint_path does not exist.
        KeyError: If the checkpoint does not contain 'config' or 'model_state_dict'.
        Exception: For other potential loading errors.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint['config']
        
        model = YourNewModel(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval() # Set model to evaluation mode
        
        print(f"Model loaded successfully from {checkpoint_path}")
        print(f"Device: {device}")
        print(f"Parameters: {model.get_num_params():,}")
        
        return model
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        raise
    except KeyError as e:
        print(f"Error: Checkpoint is missing a required key: {e}. Expected 'config' and 'model_state_dict'.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        raise


def generate_text(
    model: YourNewModel,
    tokenizer: YourNewModelTokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    num_samples: int = 1,
) -> List[Dict[str, Any]]:
    """
    Generates text based on a given prompt using the loaded LLM.

    The function encodes the prompt, feeds it to the model, and decodes
    the generated token IDs back into human-readable text. It can generate
    multiple samples based on the `num_samples` parameter.

    Args:
        model (YourNewModel): The loaded LLM instance.
        tokenizer (YourNewModelTokenizer): The tokenizer used to encode prompts and
                                     decode generated tokens.
        prompt (str): The initial text prompt to start generation from.
        max_length (int, optional): The maximum number of tokens to generate,
                                    including the prompt tokens. Defaults to 100.
        temperature (float, optional): Controls the randomness of predictions.
                                       Higher values (e.g., 1.0) make the output
                                       more random, lower values (e.g., 0.7) make it
                                       more deterministic. Defaults to 1.0.
        num_samples (int, optional): The number of independent text samples to generate.
                                     Defaults to 1.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                              contains the original 'prompt', the 'generated_text',
                              and the raw 'generated_tokens' (list of token IDs).
    """
    device = next(model.parameters()).device
    
    # Encode prompt
    prompt_tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    
    results = []
    
    for i in range(num_samples):
        print(f"\nGenerating sample {i+1}/{num_samples}...")
        
        with torch.no_grad(): # Disable gradient calculation for inference
            generated = model.generate(
                input_ids=input_ids.clone(), # Use clone to ensure input_ids is not modified
                max_length=max_length,
                temperature=temperature,
                # Assuming model.generate handles top_k/top_p internally or they are not supported
            )
        
        # Decode generated text
        generated_tokens = generated[0].cpu().tolist()
        generated_text = tokenizer.decode(generated_tokens)
        
        results.append({
            'prompt': prompt,
            'generated_text': generated_text,
            'generated_tokens': generated_tokens,
        })
        
        print(f"Generated text: {generated_text}")
    
    return results


def interactive_generation(model: YourNewModel, tokenizer: YourNewModelTokenizer):
    """
    Provides an interactive command-line interface for text generation.

    Users can type prompts and issue commands to adjust generation parameters
    like temperature, maximum length, and number of samples.
    The loop continues until the user types 'quit'.

    Args:
        model (YourNewModel): The loaded LLM instance.
        tokenizer (YourNewModelTokenizer): The tokenizer for encoding/decoding.
    """
    print("\n=== Interactive Text Generation ===")
    print("Type your prompts below. Type 'quit' to exit.")
    print("Commands:")
    print("  - 'temp <value>': Set temperature (e.g., 'temp 0.8', default: 1.0)")
    print("  - 'len <value>': Set max length (e.g., 'len 200', default: 100)")
    print("  - 'samples <value>': Set number of samples (e.g., 'samples 3', default: 1)")
    print()
    
    temperature = 1.0
    max_length = 100
    num_samples = 1
    
    while True:
        try:
            user_input = input(">>> ").strip()
            
            if user_input.lower() == 'quit':
                print("Exiting interactive mode.")
                break
            
            if user_input.startswith('temp '):
                try:
                    temperature = float(user_input.split(maxsplit=1)[1])
                    print(f"Temperature set to {temperature}")
                except (ValueError, IndexError):
                    print("Invalid temperature value. Please provide a number (e.g., 'temp 0.7').")
                continue
            
            if user_input.startswith('len '):
                try:
                    max_length = int(user_input.split(maxsplit=1)[1])
                    print(f"Max length set to {max_length}")
                except (ValueError, IndexError):
                    print("Invalid length value. Please provide an integer (e.g., 'len 150').")
                continue
            
            if user_input.startswith('samples '):
                try:
                    num_samples = int(user_input.split(maxsplit=1)[1])
                    print(f"Number of samples set to {num_samples}")
                except (ValueError, IndexError):
                    print("Invalid samples value. Please provide an integer (e.g., 'samples 2').")
                continue
            
            if not user_input:
                continue
            
            # Generate text
            results = generate_text(
                model=model,
                tokenizer=tokenizer,
                prompt=user_input,
                max_length=max_length,
                temperature=temperature,
                num_samples=num_samples,
            )
            
            print("-" * 50)
            
        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break
        except Exception as e:
            print(f"An error occurred during interactive generation: {e}")


def benchmark_model(model: YourNewModel, tokenizer: YourNewModelTokenizer):
    """
    Benchmarks the model's forward pass performance across different
    sequence lengths and batch sizes.

    Measures inference time and (for CUDA) peak memory allocation.
    Results are printed to the console.

    Args:
        model (YourNewModel): The loaded LLM instance to benchmark.
        tokenizer (YourNewModelTokenizer): The tokenizer (used for vocab size for random input).
    """
    print("\n=== Model Benchmarking ===")
    
    device = next(model.parameters()).device
    
    # Test different sequence lengths
    seq_lengths = [10, 50, 100, 200, 500, 1024] # Added 1024 for common context window size
    batch_sizes = [1, 2, 4, 8, 16] # Added 16 for larger batch testing
    
    print("Forward pass benchmarks:")
    print(f"{'Seq Length':<12} | {'Batch Size':<12} | {'Time (ms)':<12} | {'Memory (MB)':<12}")
    print("-" * 55) # Adjusted separator length
    
    # Reset CUDA memory stats before benchmarking
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    for seq_len in seq_lengths:
        for batch_size in batch_sizes:
            try:
                # Create random input
                # Ensure vocab_size is accessible from tokenizer or model config
                vocab_size = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else model.config.vocab_size
                input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
                
                # Warm up run to ensure CUDA context is initialized and caches are warm
                with torch.no_grad():
                    _ = model(input_ids)
                
                # Benchmark
                if device.type == 'cuda':
                    torch.cuda.synchronize() # Ensure all previous CUDA ops are complete
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)

                    start_event.record()
                    with torch.no_grad():
                        _ = model(input_ids)
                    end_event.record()
                    torch.cuda.synchronize() # Wait for the events to complete

                    elapsed_time = start_event.elapsed_time(end_event) # Time in ms
                    
                    # Memory usage
                    # Note: torch.cuda.max_memory_allocated() might show cumulative memory.
                    # For a single forward pass, it's better to look at memory delta or
                    # manage memory more carefully if precise per-pass memory is needed.
                    # This gives current allocated.
                    memory_used = torch.cuda.memory_allocated() / 1024**2 # Current allocated memory
                    # For peak memory during the operation, you'd need to profile more deeply
                    # or reset stats around the specific operation.
                    
                else: # CPU benchmarking
                    start_time = time.perf_counter()
                    with torch.no_grad():
                        _ = model(input_ids)
                    elapsed_time = (time.perf_counter() - start_time) * 1000 # Convert to ms
                    memory_used = 0 # CPU memory tracking is more complex and less direct via torch

                print(f"{seq_len:<12d} | {batch_size:<12d} | {elapsed_time:<11.2f} | {memory_used:<11.1f}")
                
            except RuntimeError as e: # Catch CUDA out of memory or other runtime errors
                print(f"{seq_len:<12d} | {batch_size:<12d} | {'OOM/Error':<11} | {'N/A':<11} | Error: {e}")
                if "out of memory" in str(e) and device.type == 'cuda':
                    torch.cuda.empty_cache() # Try to clear cache on OOM
            except Exception as e:
                print(f"{seq_len:<12d} | {batch_size:<12d} | {'ERROR':<11} | {'N/A':<11} | {e}")


def main():
    """
    Main function to parse command-line arguments and run the LLM inference.

    Handles loading the model and tokenizer, then directs control to either
    interactive generation, single prompt generation, or benchmarking based
    on the provided arguments.
    """
    parser = argparse.ArgumentParser(description="Generate text with Dense LLM")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (e.g., 'model_checkpoint.pt')")
    parser.add_argument("--prompt", type=str, help="Text prompt for generation (required if not in interactive mode)")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum generation length (including prompt tokens). Defaults to 100.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature. Higher values increase randomness. Defaults to 1.0.")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of independent samples to generate. Defaults to 1.")
    parser.add_argument("--interactive", action="store_true", help="Start interactive text generation mode.")
    parser.add_argument("--benchmark", action="store_true", help="Run model performance benchmarks (forward pass time and memory).")
    parser.add_argument("--device", type=str, default="auto", help="Device to use for inference ('auto', 'cpu', 'cuda'). Defaults to 'auto'.")
    
    args = parser.parse_args()
    
    # Load model
    try:
        model = load_model(args.checkpoint, args.device)
    except Exception as e:
        print(f"Failed to load model. Exiting. Error: {e}")
        return # Exit if model loading fails
    
    # Create tokenizer
    # Assuming YourNewModelTokenizer can be initialized with vocab_size from model config
    # and that it's a basic tokenizer for demonstration.
    # In a real scenario, you'd load a pre-trained tokenizer (e.g., from files).
    try:
        tokenizer = YourNewModelTokenizer(vocab_size=model.config.vocab_size)
    except AttributeError:
        print("Error: Model config does not have 'vocab_size' attribute. Cannot initialize YourNewModelTokenizer.")
        return
    except Exception as e:
        print(f"Error initializing YourNewModelTokenizer: {e}")
        return

    if args.benchmark:
        benchmark_model(model, tokenizer)
    
    if args.interactive:
        interactive_generation(model, tokenizer)
    elif args.prompt:
        results = generate_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            num_samples=args.num_samples,
        )
        
        for i, result in enumerate(results):
            print(f"\n=== Sample {i+1} ===")
            print(f"Prompt: {result['prompt']}")
            print(f"Generated: {result['generated_text']}")
    else:
        print("\nNo action specified. Please provide --prompt, --interactive, or --benchmark.")
        parser.print_help()


if __name__ == "__main__":
    main()
