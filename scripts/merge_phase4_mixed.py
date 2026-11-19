import os
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def merge_phase4_mixed():
    base_dir = "export/QJev-1.5B-Phase3"
    lora_dir = "checkpoints/phase4_mixed/lora_adapter"
    head_src = "checkpoints/phase4_mixed/decision_head.safetensors"
    
    out_dir = "export/QJev-1.5B-Phase4-Mixed"
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 60)
    print("MERGING BALANCED PHASE 4 ADAPTER INTO STANDALONE MODEL")
    print("=" * 60)
    
    print(f"Loading base model from: {base_dir}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_dir,
        torch_dtype=torch.bfloat16,
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_dir)
    
    print(f"Loading and merging LoRA adapter from: {lora_dir}...")
    peft_model = PeftModel.from_pretrained(base_model, lora_dir)
    merged_model = peft_model.merge_and_unload()
    
    print(f"Saving merged transformer weights to: {out_dir}...")
    merged_model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    
    # Copy decision head
    shutil.copy(head_src, os.path.join(out_dir, "decision_head.safetensors"))
    
    # Copy serving and config scripts
    for f in ["qjev.py", "serve.py", "Modelfile", "handler.py", "LICENSE", "decision_config.json", "requirements.txt"]:
        src_f = os.path.join(base_dir, f)
        if os.path.exists(src_f):
            shutil.copy(src_f, os.path.join(out_dir, f))
            
    print(f"Balanced Phase 4 Model successfully merged and packaged at: {out_dir}")
    return out_dir

if __name__ == "__main__":
    merge_phase4_mixed()
