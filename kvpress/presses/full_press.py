
from dataclasses import dataclass
from kvpress.presses.base_press import BasePress

@dataclass
class FullPress(BasePress):
    def __post_init__(self):
        super().__post_init__()
        # Initialize tokenizer and input_tokens for step tracking
        self.tokenizer = None
        self.input_tokens = None
    
    def compress_decoding(self, module, hidden_states, keys, values, attentions, kwargs):
        # Track generation steps for FullPress (all tokens retained)
        kv_len = keys.shape[2]
        layer_idx = getattr(module, "layer_idx", 0)
        
        # Track at first layer only to avoid duplicates
        if layer_idx == 0:
            if kv_len <= len(self.input_tokens):
                all_token_ids = self.input_tokens[:kv_len].cpu().tolist()
                retained_token_ids = all_token_ids.copy()  # All tokens retained in FullPress
            else:
                # If kv_len > input_tokens, we have generated tokens
                # Use input tokens + placeholder indices for generated tokens
                all_token_ids = self.input_tokens.cpu().tolist() + list(range(len(self.input_tokens), kv_len))
                retained_token_ids = all_token_ids.copy()  # All tokens retained
            self.track_generation_step(all_token_ids, retained_token_ids, self.tokenizer)
        return keys, values
    
    def compress_prefilling(self, module, hidden_states, keys, values, attentions, kwargs):
        return keys, values