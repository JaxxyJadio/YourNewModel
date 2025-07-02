<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Dense LLM Development Instructions

This project implements a dense transformer-based language model from scratch. When working on this codebase:

## Code Style and Architecture
- Follow PyTorch best practices and conventions
- Use type hints for all function parameters and return values
- Implement proper error handling and validation
- Use descriptive variable names that reflect the tensor shapes and purposes
- Add comprehensive docstrings explaining tensor dimensions (e.g., [batch_size, seq_len, hidden_dim])

## Model Implementation Guidelines
- Keep the transformer architecture modular and extensible
- Implement proper weight initialization (Xavier/Kaiming)
- Add support for gradient checkpointing to save memory
- Use torch.nn.functional for mathematical operations when possible
- Implement proper masking for attention mechanisms

## Training Infrastructure
- Support both single-GPU and distributed training
- Implement proper checkpointing and resuming
- Add comprehensive logging and metrics tracking
- Use mixed precision training (torch.cuda.amp) for efficiency
- Implement gradient clipping and learning rate scheduling

## Data Processing
- Handle tokenization efficiently with proper batching
- Implement data streaming for large datasets
- Add proper data validation and preprocessing
- Support multiple data formats (text, JSON, HuggingFace datasets)

## Testing and Validation
- Write unit tests for all core components
- Test with small synthetic datasets first
- Validate gradients and forward/backward passes
- Check memory usage and performance benchmarks

## Performance Optimization
- Profile code regularly to identify bottlenecks
- Use torch.compile() for optimization when available
- Implement efficient attention mechanisms (flash attention if possible)
- Consider model parallelism for very large models
