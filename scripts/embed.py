import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import argparse

# Short aliases for available models
MODEL_ALIASES = {
    "vya": "krutrim-ai-labs/Vyakyarth",
    "l3c": "l3cube-pune/indic-sentence-similarity-sbert",
    "min": "sentence-transformers/all-MiniLM-L6-v2",
    "mpn": "sentence-transformers/all-mpnet-base-v2",
    "lbs": "sentence-transformers/LaBSE"
}

def sanitize_filename(name: str) -> str:
    """Replace slashes with underscores for safe filenames."""
    return name.replace("/", "_")

def generate_embeddings(csv_path: str, embeddings_path: str, model_name: str) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if 'text' not in df.columns:
        raise ValueError("CSV file must contain a 'text' column for embeddings.")

    print(f"[Embed] Loading model {model_name} ...")
    model = SentenceTransformer(model_name)

    print(f"[Embed] Encoding {len(df)} texts ...")
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True)

    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    np.save(embeddings_path, embeddings)
    print(f"[Embed] Embeddings saved to {embeddings_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings from NCO cleaned dataset CSV")
    parser.add_argument("--csv", type=str, default="data/processed/nco_processed.csv",
                        help="Input cleaned CSV file path")
    parser.add_argument("--model", type=str, choices=MODEL_ALIASES.keys(),
                        required=True, help=f"Model alias: {', '.join(MODEL_ALIASES.keys())}")
    
    args = parser.parse_args()

    full_model_name = MODEL_ALIASES[args.model]
    safe_name = sanitize_filename(full_model_name)
    embeddings_path = f"embeddings/nco_embeddings_{safe_name}.npy"

    generate_embeddings(args.csv, embeddings_path, full_model_name)

if __name__ == "__main__":
    main()
