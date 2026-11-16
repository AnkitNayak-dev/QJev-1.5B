import argparse
import os
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser(description="Upload QJev-1.5B to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repo id, e.g. 'devankit/QJev-1.5B'")
    parser.add_argument("--folder", type=str, default="export/Qwen2.5-1.5B-Decision-Head-Merged", help="Folder to upload")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token (optional if already logged in)")
    parser.add_argument("--private", action="store_true", help="Create as private repository")
    args = parser.parse_args()

    print(f"=== Uploading QJev-1.5B to Hugging Face ===")
    print(f"Target Repo: {args.repo_id}")
    print(f"Local Folder: {args.folder}")

    if not os.path.exists(args.folder):
        raise FileNotFoundError(f"Export directory not found: {args.folder}")

    api = HfApi(token=args.token)
    
    print("\n[1/2] Creating/verifying repository on Hugging Face...")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True
    )
    print(f"Repository ready: https://huggingface.co/{args.repo_id}")

    print("\n[2/2] Uploading model files (including 3GB model.safetensors, LoRA adapter, tokenizer, and README)...")
    api.upload_folder(
        folder_path=args.folder,
        repo_id=args.repo_id,
        repo_type="model"
    )
    print("\n============================================================")
    print(f"SUCCESS! QJev-1.5B is live at: https://huggingface.co/{args.repo_id}")
    print("============================================================")

if __name__ == "__main__":
    main()
