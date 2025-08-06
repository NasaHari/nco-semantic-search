import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

def search_nco(query: str, top_k: int = 5, model_name: str = "all-MiniLM-L6-v2") -> str:
    """Perform semantic search and return JSON results."""
    model = SentenceTransformer(model_name)
    query_emb = model.encode([query])
    
    index = faiss.read_index("embeddings/nco_index.faiss")
    distances, indices = index.search(query_emb.astype(np.float32), top_k)
    
    df = pd.read_csv("data/processed/nco_data.csv")
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        row = df.iloc[idx]
        score = 1 - (dist / np.max(distances))  # Simple confidence normalization
        results.append({
            "code": row["Code"],
            "title": row["Title"],
            "description": row["Description"],
            "score": float(score)
        })
    return json.dumps(results, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    args = parser.parse_args()
    print(search_nco(args.query))
