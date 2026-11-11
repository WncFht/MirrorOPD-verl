---
license: apache-2.0
---

# Llama-3.2-3B-Instruct-arithmetic-full

This is a version of `meta-llama/Llama-3.2-3B-Instruct` fine-tuned for an arithmetic task.

## Model Details

- **Base Model:** `meta-llama/Llama-3.2-3B-Instruct`
- **Fine-tuning method:** Full fine-tuning
- **Dataset:** (Please fill in - inferred as 'arithmetic' from path)
- **Learning Rate:** 0.0001
- **Training Samples:** 20000

This model was trained as part of an experiment run on 2025-04-19.

## How to use

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "YOUR_HF_USERNAME/Llama-3.2-3B-Instruct-arithmetic-full" # Replace with your repo ID
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# ... your code to use the model
``` 