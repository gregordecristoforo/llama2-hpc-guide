from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

hf_token = "***********************"
custom_directory = "/home/gregor/git/llama_2/"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hf_token, cache_dir=custom_directory)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hf_token, cache_dir=custom_directory)

# Encode some text to test the model (replace 'Your text here' with your input)
input_text = "What is the capital of China?"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Generate text using the model
with torch.no_grad():
    output = model.generate(input_ids, max_length=50)

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
