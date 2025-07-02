"""
YourNewModel: Dense Transformer Language Model Implementation (Refactored)

All modifiable hyperparameters are grouped at the top for easy editing.
"""

import math
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# =====================
# CONFIGURABLE VALUES (EDIT THESE)
# =====================
USER_MODEL_CONFIG = {
    'vocab_size': 175000,      # Vocabulary size
    'n_positions': 12088,      # Max sequence length
    'n_embd': 6144,            # Hidden dimension
    'n_layer': 24,            # Number of transformer layers
    'n_head': 24,             # Number of attention heads
    'n_inner': None,          # Feedforward inner dim (None = 4 * n_embd)
    'activation_function': "gelu_new",  # Activation function
    'attn_pdrop': 0.1,        # Attention dropout
    'embd_pdrop': 0.1,        # Embedding dropout
    'resid_pdrop': 0.1,       # Residual dropout
    'layer_norm_epsilon': 1e-5,
    'initializer_range': 0.02,
    'use_cache': True,
    'scale_attn_weights': True,
    'scale_attn_by_inverse_layer_idx': False,
    'reorder_and_upcast_attn': False,
}


@dataclass
class YourNewModelConfig:
    """Configuration class for the dense transformer model."""
    vocab_size: int
    n_positions: int
    n_embd: int
    n_layer: int
    n_head: int
    n_inner: Optional[int] = None
    activation_function: str = "gelu_new"
    attn_pdrop: float = 0.1
    embd_pdrop: float = 0.1
    resid_pdrop: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    use_cache: bool = True
    scale_attn_weights: bool = True
    scale_attn_by_inverse_layer_idx: bool = False
    reorder_and_upcast_attn: bool = False

    def __post_init__(self):
        if self.n_inner is None:
            self.n_inner = 4 * self.n_embd
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        )


class YourNewModelAttention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, config: YourNewModelConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_dim = self.n_embd // self.n_head
        self.split_size = self.n_embd
        
        if self.head_dim * self.n_head != self.n_embd:
            raise ValueError(
                f"`n_embd` must be divisible by `n_head` (got `n_embd`: {self.n_embd} and `n_head`: {self.n_head})."
            )
        
        # Combined query, key, value projection
        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd, bias=True)
        # Output projection
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=True)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # Causal mask
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((config.n_positions, config.n_positions), dtype=torch.bool)).view(
                1, 1, config.n_positions, config.n_positions
            ),
            persistent=False,
        )
        
        # Scale factor
        self.scale_attn_weights = config.scale_attn_weights
        if self.scale_attn_weights:
            self.scale_attn = 1.0 / math.sqrt(self.head_dim)
            if config.scale_attn_by_inverse_layer_idx and layer_idx is not None:
                self.scale_attn /= float(layer_idx + 1)
    
    def _attn(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Compute attention weights and apply to values."""
        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        
        if self.scale_attn_weights:
            attn_weights = attn_weights * self.scale_attn
        
        # Apply causal mask
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.full([], mask_value, dtype=attn_weights.dtype, device=attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights.to(attn_weights.dtype), mask_value)
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.type(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, value)
        
        return attn_output, attn_weights
    
    def _split_heads(self, tensor: torch.Tensor, num_heads: int, attn_head_size: int):
        """Split the last dimension into (num_heads, attn_head_size)."""
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)
    
    def _merge_heads(self, tensor: torch.Tensor, num_heads: int, attn_head_size: int):
        """Merge attention heads back."""
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
    ):
        # Combined QKV projection
        qkv = self.c_attn(hidden_states)
        query, key, value = qkv.split(self.split_size, dim=2)
        
        query = self._split_heads(query, self.n_head, self.head_dim)
        key = self._split_heads(key, self.n_head, self.head_dim)
        value = self._split_heads(value, self.n_head, self.head_dim)
        
        if layer_past is not None:
            past_key, past_value = layer_past
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)
        
        if use_cache:
            present = (key, value)
        else:
            present = None
        
        attn_output, attn_weights = self._attn(query, key, value, attention_mask)
        
        attn_output = self._merge_heads(attn_output, self.n_head, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output, present


class YourNewModelMLP(nn.Module):
    """Feed-forward network."""
    
    def __init__(self, config: YourNewModelConfig):
        super().__init__()
        n_embd = config.n_embd
        n_inner = getattr(config, 'n_inner', 4 * n_embd)  # Safe fallback
        
        self.c_fc = nn.Linear(n_embd, n_inner, bias=True)
        self.c_proj = nn.Linear(n_inner, n_embd, bias=True)
        self.act = self._get_activation_function(config.activation_function)
        self.dropout = nn.Dropout(config.resid_pdrop)
    
    def _get_activation_function(self, activation_function: str):
        """Get activation function by name."""
        if activation_function == "relu":
            return nn.ReLU()
        elif activation_function == "gelu":
            return nn.GELU()
        elif activation_function == "gelu_new":
            # Uses a slightly different GELU implementation
            def gelu_new(x):
                return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
            return gelu_new
        elif activation_function == "swish":
            return nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation function: {activation_function}")
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class YourNewModelBlock(nn.Module):
    """Transformer block."""
    
    def __init__(self, config: YourNewModelConfig, layer_idx: Optional[int] = None):
        super().__init__()
        hidden_size = config.n_embd
        
        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = YourNewModelAttention(config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.mlp = YourNewModelMLP(config)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_outputs = self.attn(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            use_cache=use_cache,
        )
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]
        
        # Residual connection
        hidden_states = attn_output + residual
        
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        
        # Residual connection
        hidden_states = residual + feed_forward_hidden_states
        
        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]
        
        return outputs  # hidden_states, present, (attentions, cross_attentions)


class YourNewModel(nn.Module):
    """Dense transformer language model."""
    
    def __init__(self, config: YourNewModelConfig):
        super().__init__()
        self.config = config
        
        # Token and position embeddings
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # Token embeddings
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)  # Position embeddings
        
        self.drop = nn.Dropout(config.embd_pdrop)
        
        # Transformer blocks
        self.h = nn.ModuleList([
            YourNewModelBlock(config, layer_idx=i) for i in range(config.n_layer)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        
        # Language modeling head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report number of parameters
        print(f"Number of parameters: {self.get_num_params():,}")
    
    def _init_weights(self, module):
        """Initialize weights following standard initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        
        # Residual projections get scaled by 1/sqrt(N) where N is the number of residual layers
        for name, p in module.named_parameters():
            if name.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=self.config.initializer_range / math.sqrt(2 * self.config.n_layer))
    
    def get_input_embeddings(self):
        return self.wte
    
    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else True
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
            device = input_ids.device
        else:
            raise ValueError("You have to specify input_ids")
        if past_key_values is None:
            past_length = 0
            past_key_values = [None] * len(self.h)
        else:
            past_length = past_key_values[0][0].size(-2) if past_key_values[0] is not None else 0
        if position_ids is None:
            position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0)
        # Token and position embeddings
        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.drop(hidden_states)
        presents: List = []
        all_hidden_states: List = []
        for i, (block, layer_past) in enumerate(zip(self.h, past_key_values)):
            if return_dict:
                all_hidden_states.append(hidden_states)
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                use_cache=use_cache,
            )
            hidden_states = outputs[0]
            if use_cache:
                presents.append(outputs[1])
        hidden_states = self.ln_f(hidden_states)
        if return_dict:
            all_hidden_states.append(hidden_states)
        # Language modeling head
        lm_logits = self.lm_head(hidden_states)
        if not return_dict:
            return (lm_logits, presents)
        return {
            "logits": lm_logits,
            "past_key_values": presents,
            "hidden_states": all_hidden_states,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 500,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate text using the model."""
        self.eval()
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize past key values for efficient generation
        past_key_values = None
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.shape[1]):
                # Forward pass
                outputs = self.forward(
                    input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                
                logits = outputs["logits"][:, -1, :]  # Get last token logits
                past_key_values = outputs["past_key_values"]
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float("-inf")
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    logits[indices_to_remove] = float("-inf")
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
                
                # Check for end of sequence
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
                if pad_token_id is not None and (next_token == pad_token_id).all():
                    break
        
        return input_ids
    
    def get_num_params(self, non_embedding: bool = True) -> int:
        """Get the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.wte.weight.numel()
            n_params -= self.wpe.weight.numel()
        return n_params


def create_model(model_size: str = "small") -> YourNewModel:
    """Create a dense transformer model with predefined configurations."""
    configs = {
        "tiny": YourNewModelConfig(
            vocab_size=175000,
            n_positions=1024,
            n_embd=512,
            n_layer=8,
            n_head=8,
            n_inner=2048,
        ),
        "small": YourNewModelConfig(
            vocab_size=175000,
            n_positions=1536,
            n_embd=1288,
            n_layer=12,
            n_head=12,
            n_inner=4096,
        ),
        "medium": YourNewModelConfig(
            vocab_size=175000,
            n_positions=2048,
            n_embd=1536,
            n_layer=18,
            n_head=12,
            n_inner=6144,
        ),
        "large": YourNewModelConfig(
            vocab_size=175000,
            n_positions=3072,
            n_embd=2048,
            n_layer=24,
            n_head=12,
            n_inner=8192,
        ),
        "xl": YourNewModelConfig(
            vocab_size=175000,
            n_positions=4096,
            n_embd=3072,
            n_layer=48,
            n_head=32,
            n_inner=12288,
        ),
    }
    config = configs.get(model_size, configs["small"])
    return YourNewModel(config)


def create_custom_model(config_dict: dict = USER_MODEL_CONFIG) -> nn.Module:
    """
    Create a model with user-specified config. Pass a dict with any of the config keys above.
    """
    config = YourNewModelConfig(**config_dict)
    return YourNewModel(config)


ModelConfig = YourNewModelConfig

if __name__ == "__main__":
    # Test the model
    model = create_model("small")
    print(f"Model parameters: {model.get_num_params():,}")
    
    # Test forward pass
    input_ids = torch.randint(0, 175000, (1, 10))
    outputs = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {outputs['logits'].shape}")
    
    # Test generation
    generated = model.generate(input_ids, max_length=20)
    print(f"Generated shape: {generated.shape}")

