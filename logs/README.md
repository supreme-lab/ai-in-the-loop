---
base_model: meta-llama/Llama-Guard-3-8B
library_name: transformers
model_name: logs
tags:
- generated_from_trainer
- sft
- trl
licence: license
---

# Model Card for logs

This model is a fine-tuned version of [meta-llama/Llama-Guard-3-8B](https://huggingface.co/meta-llama/Llama-Guard-3-8B).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- TRL: 0.25.1
- Transformers: 4.57.1
- Pytorch: 2.7.0
- Datasets: 3.5.0
- Tokenizers: 0.22.1

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```