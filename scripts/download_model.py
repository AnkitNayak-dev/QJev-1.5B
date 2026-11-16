import os
import time
from huggingface_hub import snapshot_download

def download_qwen_model(model_id="Qwen/Qwen2.5-1.5B-Instruct", local_dir="models/Qwen2.5-1.5B-Instruct"):
    print(f"=== Downloading Full Model: {model_id} ===")
    print(f"Destination: {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    
    start_time = time.time()
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
    )
    elapsed = time.time() - start_time
    print(f"\nSuccessfully downloaded model in {elapsed/60:.2f} minutes!")

if __name__ == "__main__":
    download_qwen_model()

