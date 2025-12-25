"""
Patch for transformers 4.51.3 flash attention bugs and missing dependency.
This module patches the missing _flash_supports_window_size variable and provides 
an SDPA fallback for _flash_attention_forward when flash_attn is not installed.
"""
import sys
import torch
import transformers.modeling_flash_attention_utils as flash_utils

# --- Part 1: Patch _flash_supports_window_size ---

# Check if flash_attn is available
try:
    import flash_attn
    flash_attn_version = flash_attn.__version__
    # Check if flash attention supports window size (version >= 2.1.0)
    _flash_supports_window_size = tuple(int(x) for x in flash_attn_version.split(".")[:3]) >= (2, 1, 0)
    print(f"Reflected flash_attn version: {flash_attn_version}")
except ImportError:
    _flash_supports_window_size = False
    print("flash_attn package not found.")

# Patch the transformers module variable
if not hasattr(flash_utils, '_flash_supports_window_size'):
    flash_utils._flash_supports_window_size = _flash_supports_window_size
    print(f"Patched transformers flash attention utils: _flash_supports_window_size = {_flash_supports_window_size}")

# --- Part 2: Patch _flash_attention_forward with SDPA fallback ---

# If flash_attn_func is None (meaning flash-attn is not installed or failed to load),
# we must provide a fallback because the GLM-ASR model code calls _flash_attention_forward unconditionally.
if getattr(flash_utils, 'flash_attn_func', None) is None:
    print("Patching _flash_attention_forward with SDPA fallback (because flash_attn_func is None)")
    
    def _flash_attention_forward_fallback(
        query_states,
        key_states,
        value_states,
        attention_mask,
        query_length,
        is_causal,
        dropout,
        position_ids=None,
        use_top_left_mask=False,
        **kwargs
    ):
        """
        Fallback implementation using torch.nn.functional.scaled_dot_product_attention
        Input shapes for Flash Attention: (batch_size, seq_len, num_heads, head_dim)
        Input shapes for SDPA: (batch_size, num_heads, seq_len, head_dim)
        """
        # Transpose to (batch, heads, seq, dim)
        q = query_states.transpose(1, 2)
        k = key_states.transpose(1, 2)
        v = value_states.transpose(1, 2)
        
        # Prepare arguments for SDPA
        sdpa_kwargs = {
            "dropout_p": dropout,
            "is_causal": is_causal,
        }
        
        # Handle attention mask
        # If is_causal is True, SDPA handles the causal mask internally.
        # If an explicit attention_mask is provided, we use it.
        # Note: Flash attention mask handling is complex, but for basic usage:
        if attention_mask is not None and not is_causal:
             sdpa_kwargs["attn_mask"] = attention_mask
        
        # Call SDPA
        attn_output = torch.nn.functional.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)
        
        # Transpose back to (batch, seq, heads, dim)
        attn_output = attn_output.transpose(1, 2)
        
        return attn_output

    flash_utils._flash_attention_forward = _flash_attention_forward_fallback
    print("Successfully patched _flash_attention_forward")
else:
    print("flash_attn_func is available, skipping _flash_attention_forward patch.")
