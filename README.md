# Consistent Rotation Base for Dynamic NTK Scaling RoPE

## Inconsistent problem
Weeks ago, [u/emozilla](https://www.reddit.com/user/emozilla) proposed an improvement on NTK-Aware RoPR in this [post](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/), later named DynamicNTKScalingRotaryEmbedding. The main idea behind Dynamic NTK involves incorporating a scaling factor relative to the present decoding sequence length to improve the base functionality.
However, there is actually a subtle gap between how we compute perplexity and how the LLM actually generates tokens. 

If you are using the DynamicNTKRope implemented by [Huggingface](https://github.com/huggingface/transformers/blob/b257c46a075419c09e5ce5c5aa39bc346ecdb9a5/src/transformers/models/llama/modeling_llama.py#L147), the sequence length remains fixed when calculating perplexity, and no keys is cached. As a result, there are no rotation base inconsistencies between keys.

However, when LLM starts generation token by token beyond its maximum trained length, the sequence length increases and each key is continually pushed into the key-value cache during decoding. Consequently, we have such a rotation inconsistency problem between keys.

To put it clearly, the current DynamicNTKRope is implemented as

<img src="doc/eq1.png" width="1000" height="60">

From my understanding, we should keep the rotation base consistent,

When decoding `sequence length = seq2`

<img src="doc/eq2.png" width="600" height="60">

As decoding sequence length increases to `seq3`,

<img src="doc/eq3.png" width="850" height="60">

Please check [this post](https://www.reddit.com/r/LocalLLaMA/comments/155bexn/a_potential_rotation_inconsistency_of_dynamically/) for more details.


## How to use
```python

from transformers import AutoTokenizer, LlamaForCausalLM
import torch
from scale_rope.consistent_rope_for_llama_patch import replace_llama_attn_with_consistent_ntk_rope

model = LlamaForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")

replace_llama_attn_with_consistent_ntk_rope()
# Generate
generate_ids = model.generate(inputs.input_ids, max_length=30)
tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

```
