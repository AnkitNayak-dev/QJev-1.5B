import os
import time
from datasets import load_dataset
from huggingface_hub import snapshot_download

def download_full_dataset(dataset_name="SargeDev/jev-distill-corpus-v3", local_dir="data/jev-distill-corpus-v3"):
    print(f"=== Downloading Full Dataset: {dataset_name} ===")
    print(f"Destination: {local_dir}")
    os.makedirs(local_dir, exist_ok=True)
    
    start_time = time.time()
    # Download all splits in full to disk
    dataset = load_dataset(dataset_name)
    print("\nDataset structure downloaded:")
    print(dataset)

    print(f"\nSaving dataset to disk at {local_dir}...")
    dataset.save_to_disk(local_dir)
    
    elapsed = time.time() - start_time
    print(f"\nSuccessfully downloaded and saved full dataset in {elapsed/60:.2f} minutes!")

if __name__ == "__main__":
    download_full_dataset()

