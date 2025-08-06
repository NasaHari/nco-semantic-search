import faiss
import numpy as np

def build_index(embeddings_path: str) -> faiss.Index:
    """Build FAISS index from embeddings."""
    embeddings = np.load(embeddings_path)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype(np.float32))
    return index

if __name__ == "__main__":
    embeddings_path = "embeddings/nco_embeddings.npy"
    index = build_index(embeddings_path)
    faiss.write_index(index, "embeddings/nco_index.faiss")
    print("Index built and saved.")
