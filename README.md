# Text-to-Text Transfer Transformer Quantized Model for Text Summarization

This repository hosts a quantized version of the T5 model, fine-tuned for text summarization tasks. The model has been optimized for efficient deployment while maintaining high accuracy, making it suitable for resource-constrained environments.

## Model Details

- **Model Architecture:** T5  
- **Task:** Text Summarization  
- **Dataset:** Hugging Face's `cnn_dailymail'  
- **Quantization:** Float16  
- **Fine-tuning Framework:** Hugging Face Transformers  

## Usage

### Installation

```sh
pip install transformers torch
```

### Loading the Model

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "AventIQ-AI/t5-text-summarizer"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

def test_summarization(model, tokenizer):
    user_text = input("\nEnter your text for summarization:\n")
    input_text = "summarize: " + user_text
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)

    output = model.generate(
        **inputs,
        max_new_tokens=100,
        num_beams=5,
        length_penalty=0.8,
        early_stopping=True
    )

    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary

print("\n📝 **Quantized Model Summary:**")
print(test_summarization(model, tokenizer))
```

# 📊 ROUGE Evaluation Results
 
After fine-tuning the **T5-Small** model for text summarization, we obtained the following **ROUGE** scores:

| **Metric**  | **Score**  | **Meaning** |
|-------------|-----------|-------------|
| **ROUGE-1** | **0.3061** (~30%) | Measures overlap of **unigrams (single words)** between the reference and generated summary. |
| **ROUGE-2** | **0.1241** (~12%) | Measures overlap of **bigrams (two-word phrases)**, indicating coherence and fluency. |
| **ROUGE-L** | **0.2233** (~22%) | Measures **longest matching word sequences**, testing sentence structure preservation. |
| **ROUGE-Lsum** | **0.2620** (~26%) | Similar to ROUGE-L but optimized for summarization tasks. |
 

## Fine-Tuning Details

### Dataset

The Hugging Face's `cnn_dailymail` dataset was used, containing the text and their summarization examples.

### Training

- Number of epochs: 3
- Batch size: 4  
- Evaluation strategy: epoch  
- Learning rate: 3e-5  

### Quantization

Post-training quantization was applied using PyTorch's built-in quantization framework to reduce the model size and improve inference efficiency.

## Repository Structure

```
.
├── model/               # Contains the quantized model files
├── tokenizer_config/    # Tokenizer configuration and vocabulary files
├── model.safetensors/   # Quantized Model
├── README.md            # Model documentation
```

## Limitations

- The model may not generalize well to domains outside the fine-tuning dataset.  
- Quantization may result in minor accuracy degradation compared to full-precision models.  

## Contributing

Contributions are welcome! Feel free to open an issue or submit a pull request if you have suggestions or improvements.

