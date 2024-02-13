from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

hf_token = "****************************"
custom_directory = "/cluster/projects/nn9997k/decristo/"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hf_token, cache_dir=custom_directory)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hf_token, cache_dir=custom_directory)

# Move the model to GPU
model = model.to('cuda')

# Encode some text to test the model (replace 'Your text here' with your input)
input_text = "What is the capital of China?"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Move the input tensors to GPU
input_ids = input_ids.to('cuda')

print("Model device:", model.device)
print("Input tensor device:", input_ids.device)

# Generate text using the model
with torch.no_grad():
    output = model.generate(input_ids, max_length=50)

if torch.cuda.is_available():
    print("Current GPU device:", torch.cuda.current_device())
    print("Current GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))

# Move the generated tensor back to CPU for decoding
output = output.to('cpu')

# Decode the generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)


