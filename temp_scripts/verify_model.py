from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

model_path = "d:/03_Development/translate_akkadian/input/datasets/byt5-akkadian-optimized-34x"

print(f"Loading model from: {model_path}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Tokenizer loaded successfully")
    print(f"Vocab size: {len(tokenizer)}")
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    print(f"Model loaded successfully")
    print(f"Model type: {type(model).__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"Model moved to GPU: {torch.cuda.get_device_name(0)}")
    
    print("\nModel verification successful!")
    
except Exception as e:
    print(f"Error loading model: {e}")
