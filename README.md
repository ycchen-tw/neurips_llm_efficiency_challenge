# NeurIPS LLM Efficiency Challenge Solution
This repo presents the training and inference code developed by team ycchen.

We are graduate students at National Taiwan University.

Our most effective submission was an 8-bit quantized GPTQ model based on Qwen-14B, which did not undergo fine-tuning. (Interestingly, our fine-tuned model yielded a lower score 😢).

## Training Code
The training code can be found in the folder named 'training_code'. 
To begin, replace "YOUR_TOKEN" and "YOUR_USERNAME/YOUR_REPO" in the Dockerfile with your personal token and repository details. Execute the following commands:
```bash
docker build -f ./Dockerfile -t qwen_quant .
docker run --gpus "device=0" --rm -ti qwen_quant
```

## Data Format

The submissions for the 4090 challenge are contained within the folder '4090_submissions', which includes the following files:

- 4090_submissions/
  - 4090_submissions_1.zip
  - 4090_submissions_2.zip
  - 4090_submissions_3.zip

# Neurips 1 LLM 1 GPU Challenge
