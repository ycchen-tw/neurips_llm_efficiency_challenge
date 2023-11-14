# NeurIPS LLM Efficiency Challenge Solution
This repository presents the training and inference code developed by team ycchen.

We are graduate students at National Taiwan University.

## Method Introduction

Our best submission was an 8-bit GPTQ quantized Qwen-14B, which did not undergo fine-tuning. Interestingly, our fine-tuned model yielded a lower score, which was an unexpected outcome 😢.

We precisely adjusted the generation settings (such as maximum token length and stop criteria) to ensure the model produced outputs that met the challenge requirements.

## Training Code
The training code is available in the folder named 'training_code'. 
It utilizes auto-gptq to quantize Qwen-14B and uploads it to the Hugging Face Hub.
To start, replace "YOUR_TOKEN" and "YOUR_USERNAME/YOUR_REPO" in the Dockerfile with your personal token and repository details. Execute the following commands:
```bash
docker build -f ./Dockerfile -t qwen_quant .
docker run --gpus "device=0" --rm -ti qwen_quant
```
The program can complete in approximately 2 hours using a single RTX 3090 GPU.

## Data Format

The submissions for the 4090 challenge are contained within the '4090_submissions' folder, which includes the following files:

- 4090_submissions/
  - 4090_submissions_1.zip
  - 4090_submissions_2.zip
  - 4090_submissions_3.zip

# NeurIPS 1 LLM 1 GPU Challenge
