import os
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from safetensors.torch import save_file, load_file

def merge_phase3():
    base_dir = "export/Qwen2.5-1.5B-Decision-Head-Merged"
    lora_dir = "checkpoints/phase3_consistency/lora_adapter"
    head_src = "checkpoints/phase3_consistency/decision_head.safetensors"
    
    out_dir = "export/QJev-1.5B-Phase3"
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 60)
    print("MERGING PHASE 3 CONSISTENCY ADAPTER INTO BASE MODEL")
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
    
    # Copy qjev.py, handler.py, and configs
    for f in ["qjev.py", "handler.py", "LICENSE", "decision_config.json"]:
        src_f = os.path.join(base_dir, f)
        if os.path.exists(src_f):
            shutil.copy(src_f, os.path.join(out_dir, f))
            
    print(f"Phase 3 Model successfully merged and packaged at: {out_dir}")
    return out_dir

if __name__ == "__main__":
    merge_phase3()
