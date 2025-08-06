import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def generate_embeddings(csv_path: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Load data and generate embeddings."""
    df = pd.read_csv(csv_path)
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df["text"].tolist(), show_progress_bar=True)
    return embeddings

if __name__ == "__main__":
    csv_path = "data/processed/nco_data.csv"
    embeddings = generate_embeddings(csv_path)
    np.save("embeddings/nco_embeddings.npy", embeddings)
    print("Embeddings generated and saved.")
