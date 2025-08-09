# scripts/index.py
import os
import numpy as np
import faiss

def build_faiss_index(embeddings_path: str, index_path: str) -> None:
    """
    Builds and saves a FAISS index from precomputed embeddings.
    Args:
        embeddings_path (str): Path to the .npy file containing embeddings.
        index_path (str): Path to save the FAISS index.
    """
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found at {embeddings_path}")
    
    print(f"[Index] Loading embeddings from {embeddings_path} ...")
    embeddings = np.load(embeddings_path).astype('float32')
    dim = embeddings.shape[1]
    print(f"[Index] Embeddings shape: {embeddings.shape}")

    # Build index (Inner Product for cosine similarity)
    index = faiss.IndexFlatIP(dim)  # Changed from L2 to IP
    print("[Index] Adding vectors to the index ...")
    index.add(embeddings)

    # Save index
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"[Index] Index saved to {index_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build FAISS index from embeddings")
    parser.add_argument("--embeddings", type=str, default="embeddings/nco_embeddings_vya.npy", help="Input embeddings file")
    parser.add_argument("--index", type=str, default="embeddings/nco_index-vya.faiss", help="Output FAISS index path")
    args = parser.parse_args()

    build_faiss_index(args.embeddings, args.index)
