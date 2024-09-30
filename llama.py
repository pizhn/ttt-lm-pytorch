import transformers
import torch

if __name__ == '__main__':
    model_id = 'meta-llama/Llama-3.2-1B'
    
    pipeline = transformers.pipeline("text-generation", model=model_id, model_kwargs={"torch_dtype": torch.bfloat16}, device_map="auto")
    pipeline("Hey how are you doing today?")
