
from kvpress.presses.base_press import BasePress

class FullPress(BasePress):
    def compress_decoding(self, module, hidden_states, keys, values, attentions, kwargs):
        return keys, values
    
    def compress_prefilling(self, module, hidden_states, keys, values, attentions, kwargs):
        return keys, values