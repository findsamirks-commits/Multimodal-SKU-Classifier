import os
import sys
from huggingface_hub import HfApi, login

# Read token from environment variable or prompt
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Notice: HF_TOKEN environment variable not set. Using cached credentials or prompt.")

api = HfApi()

REPO_ID = "findsamirks/Retail-Multimodal-Classifier"
CHECKPOINT_PATH = "checkpoints/best_multimodal_model.pt"

print(f"Creating Hugging Face repository {REPO_ID}...")
try:
    api.create_repo(
        repo_id=REPO_ID, 
        repo_type="model", 
        exist_ok=True
    )
except Exception as e:
    print(f"Repo creation info: {e}")

if os.path.exists(CHECKPOINT_PATH):
    print(f"Uploading model weights from {CHECKPOINT_PATH}...")
    api.upload_file(
        path_or_fileobj=CHECKPOINT_PATH,
        path_in_repo="best_multimodal_model.pt",
        repo_id=REPO_ID,
    )
    print("✅ Upload complete! Weights are now live on Hugging Face.")
else:
    print(f"Checkpoints file not found at {CHECKPOINT_PATH}. Run training first.")
