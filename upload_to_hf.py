#!/usr/bin/env python3
"""
Upload QJev-1.5B standalone package to Hugging Face Hub.

Usage:
    python upload_to_hf.py --repo-id <username>/QJev-1.5B [--token <hf_token>] [--private]
"""

import os
import argparse
from pathlib import Path
from huggingface_hub import HfApi, login

def main():
    parser = argparse.ArgumentParser(description="Upload QJev-1.5B to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face repo ID, e.g., 'your-username/QJev-1.5B'")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face user access token (write access)")
    parser.add_argument("--folder", type=str, default="export/Qwen2.5-1.5B-Decision-Head-Merged", help="Folder path to upload")
    parser.add_argument("--private", action="store_true", help="Set repository to private")
    
    args = parser.parse_args()
    
    token = args.token or os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
        print("[+] Logged into Hugging Face Hub.")
    
    api = HfApi(token=token)
    user = api.whoami()
    print(f"[+] Authenticated as: {user['name']} ({user.get('email', 'no email')})")
    
    folder_path = Path(args.folder)
    if not folder_path.exists():
        raise FileNotFoundError(f"Export directory not found: {folder_path.resolve()}")
    
    print(f"[+] Creating repository '{args.repo_id}' (exist_ok=True, private={args.private})...")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        private=args.private,
        exist_ok=True
    )
    
    print(f"[+] Uploading directory '{folder_path}' to '{args.repo_id}'...")
    print("    This uploads model.safetensors (3.08 GB), lora_adapter/ (16.6 MB), and configs.")
    api.upload_folder(
        folder_path=str(folder_path),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message="feat: initial release of QJev-1.5B (merged weights + lora adapter)"
    )
    
    print("\n" + "=" * 60)
    print(f" SUCCESS! Model published to:")
    print(f" https://huggingface.co/{args.repo_id}")
    print("=" * 60)

if __name__ == "__main__":
    main()
