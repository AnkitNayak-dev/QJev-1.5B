"""
GGUF Export Helper for QJev-1.5B.
Converts QJev weights to GGUF format for llama.cpp, Ollama, LM Studio, and Jan.ai.

Usage:
    python scripts/export_gguf.py --model-dir export/QJev-1.5B-Phase3 --out-file qjev-1.5b-f16.gguf
"""

import os
import sys
import argparse
import subprocess

def export_to_gguf(model_dir: str, out_file: str):
    print("=" * 70)
    print("QJev-1.5B GGUF EXPORT TOOL (llama.cpp & Ollama Compatible)")
    print(f"Source Model: {model_dir}")
    print(f"Output File:  {out_file}")
    print("=" * 70)

    # Check if convert_hf_to_gguf exists locally or in PATH
    script_candidates = [
        "convert_hf_to_gguf.py",
        "llama.cpp/convert_hf_to_gguf.py",
        os.path.expanduser("~/llama.cpp/convert_hf_to_gguf.py")
    ]
    
    found_script = None
    for cand in script_candidates:
        if os.path.exists(cand):
            found_script = cand
            break
            
    if found_script:
        print(f"Found llama.cpp converter: {found_script}")
        cmd = [sys.executable, found_script, model_dir, "--outfile", out_file, "--outtype", "f16"]
        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"GGUF Export Complete: {out_file}")
    else:
        print("\n[NOTE] llama.cpp convert script not found locally.")
        print("To convert QJev to GGUF in 2 simple terminal commands:")
        print("-------------------------------------------------------")
        print("1. Clone llama.cpp:")
        print("   git clone https://github.com/ggerganov/llama.cpp")
        print("   pip install -r llama.cpp/requirements.txt")
        print()
        print(f"2. Convert QJev to GGUF:")
        print(f"   python llama.cpp/convert_hf_to_gguf.py {model_dir} --outfile {out_file}")
        print()
        print("3. (Optional) Quantize to 4-bit (1.1 GB):")
        print(f"   ./llama.cpp/llama-quantize {out_file} qjev-1.5b-q4_k_m.gguf Q4_K_M")
        print("-------------------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default="export/QJev-1.5B-Phase3")
    parser.add_argument("--out-file", default="qjev-1.5b-f16.gguf")
    args = parser.parse_args()
    export_to_gguf(args.model_dir, args.out_file)
