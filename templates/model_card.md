---
{{ card_data }}
---

# Model Card for {{ model_id | default("Model name", true) }}

This is a multi-modal and multi-task model.

## Model Details

### Model Description

- **Developed by:** The JAT Team
- **License:** Apache 2.0

### Model Sources

- **Repository:** <https://github.com/huggingface/jat>
- **Paper:** Coming soon
- **Demo:** Coming soon

## Training

The model was trained on the following tasks:

{% for task in tasks -%}
- {{ task }}
{% endfor %}
## How to Get Started with the Model

Use the code below to get started with the model.

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("{{ model_name | default("[More Information Needed]", true)}}")
```

