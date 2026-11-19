import os
import sys
from huggingface_hub import HfApi

def upload_models():
    api = HfApi()
    user = api.whoami()
    username = user["name"]
    print(f"Authenticated as Hugging Face user: {username}")
    
    models_to_upload = [
        ("AnkitNayak-dev/QJev-1.5B", "export/QJev-1.5B-Phase3", "Upload Phase 3 Consistency Standalone Model (62.53% JevBench)"),
        ("AnkitNayak-dev/QJev-1.5B-LoRA", "export/QJev-1.5B-LoRA", "Update Phase 3 LoRA Adapter and Decision Head"),
        ("AnkitNayak-dev/QJev-1.5B-4bit", "export/QJev-1.5B-4bit", "Update Phase 3 4-Bit NF4 Quantized Edition")
    ]
    
    for repo_id, folder_path, commit_msg in models_to_upload:
        print("\n" + "=" * 70)
        print(f"UPLOADING: {repo_id}")
        print(f"Source Folder: {folder_path}")
        print("=" * 70)
        
        if not os.path.exists(folder_path):
            print(f"ERROR: Folder {folder_path} does not exist, skipping.")
            continue
            
        print("Pushing files to Hugging Face Hub...")
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_msg
        )
        print(f"SUCCESS: {repo_id} is live at https://huggingface.co/{repo_id}")
        
    print("\n" + "=" * 70)
    print("ALL MODELS SUCCESSFULLY PUBLISHED TO HUGGING FACE!")
    print("=" * 70)

if __name__ == "__main__":
    upload_models()
