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
* run llama2 with selected tokenizer and model:
    ```python
    # Encode some text to test the model (replace 'Your text here' with your input)
    input_text = "What is the capital of China?"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    # Generate text using the model
    with torch.no_grad():
        output = model.generate(input_ids, max_length=50)

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    print(generated_text)
    ```

    ## Use llama2 on SAGA
There are multiple ways to install python packages on SAGA. I choose `virtualenv` following this part of the documentation: https://documentation.sigma2.no/software/userinstallsw/python.html
* Run the following command since we will run our model on A100 GPUS: `module --force swap StdEnv Zen2Env`
* Load a compatible python version: `module load Python/3.10.8-GCCcore-12.2.0`
* Create the virtual environment: `python -m venv llama2`
* Activate the environment: `source llama2/bin/activate`
* Install dependencies of the project. I advise putting all dependencies into a `requirements.txt` file. That way we make sure that the build stays reproducable. The used `requirements.txt` file is in this repository. To install the dependencies run: `pip install -r requirements.txt`
* Modify python script to make use of GPU. For this we need to move `model` and the `input_ids` to the GPU and the `output` back to the CPU:
    ```python
    # Move the model to GPU
    model = model.to('cuda')

    # Move the input tensors to GPU
    input_ids = input_ids.to('cuda')

    # Move the generated tensor back to CPU for decoding
    output = output.to('cpu')
    ```
    Take whole script is available as `llama2.py` in this repository
* Set up job script. The job script generator was very useful: https://open.pages.sigma2.no/job-script-generator/
  However, we need to set `#SBATCH --partition=a100` since the GPUs from the `accel` have to little memory. Also, make sure to use enough RAM since llama2 uses around 25GB. The batch script I used, `llama2.sh`, is in this repository.
* Submit the job to run llama2: `sbatch llama2.sh`

