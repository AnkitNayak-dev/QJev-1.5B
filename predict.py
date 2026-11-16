import sys
import os
import argparse
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from model import QwenDecisionModel
from dataset import format_prompt

def load_decision_model(model_path="models/Qwen2.5-1.5B-Instruct", checkpoint_path="checkpoints/decision_head_100k.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Qwen Decision Model on {device}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = QwenDecisionModel(model_path, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
    
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["head_state_dict"] if "head_state_dict" in ckpt else ckpt
        model.head.load_state_dict(state_dict)
        print(f"Loaded decision head from: {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found!")

    model.to(device)
    model.eval()
    return model, tokenizer, device

def predict_decision(model, tokenizer, device, state: str, question: str, options: list):
    prompt = format_prompt(tokenizer, state, question, options)
    inputs = tokenizer([prompt], return_tensors="pt").to(device)
    num_choices = torch.tensor([len(options)], dtype=torch.long, device=device)

    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"], num_choices=num_choices)
        probs = torch.softmax(logits[0, :len(options)], dim=-1).cpu().tolist()

    best_idx = int(torch.argmax(logits[0, :len(options)]).item())
    
    return {
        "best_choice": options[best_idx],
        "confidence": round(probs[best_idx] * 100, 2),
        "distribution": {opt: round(p * 100, 2) for opt, p in zip(options, probs)}
    }

if __name__ == "__main__":
    model, tokenizer, device = load_decision_model()
    
    print("\n--- Example Decision Query ---")
    state = "The patient presents with sudden severe chest pain, elevated cardiac troponin, and ST-segment elevation on ECG."
    question = "What is the immediate recommended clinical action?"
    options = [
        "Discharge with oral analgesics",
        "Perform emergency cardiac catheterization / PCI",
        "Wait 24 hours and re-evaluate"
    ]
    
    print(f"Context: {state}")
    print(f"Question: {question}")
    result = predict_decision(model, tokenizer, device, state, question, options)
    print("\nResult:")
    print(json.dumps(result, indent=2))

