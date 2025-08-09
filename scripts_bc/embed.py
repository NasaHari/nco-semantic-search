# scripts/embed.py
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss  # Add for normalization

def generate_embeddings(csv_path: str, embeddings_path: str, model_name: str = "krutrim-ai-labs/Vyakyarth") -> None:
    """
    Generate and save embeddings for the occupation texts in the CSV.
    Args:
        csv_path (str): Path to cleaned CSV containing the column 'text'.
        embeddings_path (str): Path to save the generated embeddings (.npy).
        model_name (str): SentenceTransformer model name.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Load data
    df = pd.read_csv(csv_path)
    if 'text' not in df.columns:
        raise ValueError("CSV file must contain a 'text' column for embeddings.")

    print(f"[Embed] Loading model {model_name} ...")
    model = SentenceTransformer(model_name)

    # Generate embeddings
    print(f"[Embed] Encoding {len(df)} texts ...")
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True).astype('float32')
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    # Save embeddings
    os.makedirs(os.path.dirname(embeddings_path), exist_ok=True)
    np.save(embeddings_path, embeddings)
    print(f"[Embed] Embeddings saved to {embeddings_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate embeddings from NCO cleaned dataset CSV")
    parser.add_argument("--csv", type=str, default="data/processed/nco_processed.csv", help="Input cleaned CSV file path")
    parser.add_argument("--output", type=str, default="embeddings/nco_embeddings_vya.npy", help="Output embeddings file path")
    parser.add_argument("--model", type=str, default="krutrim-ai-labs/Vyakyarth", help="SentenceTransformer model name")  # Updated default
    args = parser.parse_args()

    generate_embeddings(args.csv, args.output, args.model)
