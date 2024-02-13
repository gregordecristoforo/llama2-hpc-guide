# Finetune llama-2 model on SAGA and Lumi

## Get access to llama2
* create Hugging Face account: https://huggingface.co/
* Request access to Llama on https://llama.meta.com/llama-downloads/ 
* Create Access token in personal settings on Hugging Face

## Use llama2 on local machine with conda
* set up pip empty conda environment with:
    `conda create --name llama2 python`
* install required packages: 
    `pip install torch transformers`
* set up script for downloading model:
    ```python    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    hf_token = "*********************************"
    custom_directory = "~/llama_2/"

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hf_token, cache_dir=custom_directory)
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", token=hf_token, cache_dir=custom_directory)
    ```
    setting a custom directory where the model is stored is not necessary, I however prefer having an explicit path when working on HPC devices

