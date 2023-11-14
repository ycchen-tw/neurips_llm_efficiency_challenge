# Challenge solution
We are graduate "students" of National Taiwan Unerversity.

Our best submission is a 8bit GPTQ quantized Qwen-14B model without fine-tuning. (Fine-tuned model get lower score😢) 

# Training code
First replace "YOUR_TOKEN" and "YOUR_USERNAME/YOUR_REPO" in Dockerfile to your token and repo.
```bash
docker build -f ./Dockerfile -t qwen_quant .
docker run --gpus "device=0" --rm -ti qwen_quant
```

## Data Format

Three 4090 submissions are in the 4090_submissions folder:

4090_submissions/

-> 4090_submissions_1.zip

-> 4090_submissions_2.zip

-> 4090_submissions_3.zip

# Neurips 1 LLM 1 GPU Challenge
