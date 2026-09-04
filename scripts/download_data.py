"""
Download Bitext customer support dataset from HuggingFace and save to data/raw/bitext_support.csv
"""
import os
import sys
import shutil
from pathlib import Path
import requests
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.config import config

def download_dataset():
    print("Downloading Bitext customer support dataset from Hugging Face...")
    csv_url = "https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset/resolve/main/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv"
    output_path = config.DATA_RAW_DIR / "bitext_support.csv"
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # Try using datasets library first if installed, else direct stream download
    try:
        from datasets import load_dataset
        ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
        split_name = list(ds.keys())[0]
        df = pd.DataFrame(ds[split_name])
        df.to_csv(output_path, index=False)
        print(f"Success! Dataset loaded via `datasets` and saved to: {output_path}")
    except Exception:
        print(f"Downloading directly from {csv_url} ...")
        res = requests.get(csv_url, stream=True, timeout=60)
        res.raise_for_status()
        with open(output_path, "wb") as f:
            for chunk in res.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"Success! Downloaded directly to: {output_path}")

    # Inspect dataset
    if output_path.exists():
        df = pd.read_csv(output_path)
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        print("\nFirst row sample:")
        print(df.head(1).to_dict(orient="records"))

if __name__ == "__main__":
    download_dataset()
