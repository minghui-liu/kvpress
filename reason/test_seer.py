from transformers import AutoTokenizer, AutoConfig
from seer_attn import SeerAttnLlamaForCausalLM ## Sparse Prefill Modeling
from seer_attn import SeerDecodingQwen3ForCausalLM ##  Sparse Decoding Modeling
import torch

## SeerAttention-R: sparse decoding 
model_name = "SeerAttention/SeerAttention-Decode-Qwen3-4B-AttnGates"
config = AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(
    config.base_model, 
    padding_side="left",
)
## Token budget based sparsity selection. You can also use threshold method
model = SeerDecodingQwen3ForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    seerattn_sparsity_method='token_budget', 
    seerattn_token_budget = 4096, 
).cuda()

prompt="""
Darrell and Allen's ages are in the ratio of 7:11. If their total age now is 162, calculate Allen's age 10 years from now.\nSolve the problem step by step. Wrap your final answer in \"\\boxed{}\".
"""
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(
    inputs["input_ids"],
    attention_mask=inputs["attention_mask"],
    max_new_tokens=100,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))