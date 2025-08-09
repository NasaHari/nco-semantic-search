import os
import numpy as np
import faiss
import argparse

# Short aliases for models
MODEL_ALIASES = {
    "vya": "krutrim-ai-labs/Vyakyarth",
    "l3c": "l3cube-pune/indic-sentence-similarity-sbert",
    "min": "sentence-transformers/all-MiniLM-L6-v2",
    "mpn": "sentence-transformers/all-mpnet-base-v2",
        "lbs": "sentence-transformers/LaBSE"

}

def build_faiss_index(model_name: str) -> None:
    """
    Builds and saves a FAISS index from precomputed embeddings for a given model.
    Output filenames are based on the model name.
    """
    safe_model = model_name.replace("/", "_")
    embeddings_path = f"embeddings/nco_embeddings_{safe_model}.npy"
    index_path = f"embeddings/nco_index_{safe_model}.faiss"

    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings not found at {embeddings_path}. Please run embed.py first.")

    print(f"[Index] Loading embeddings from {embeddings_path} ...")
    embeddings = np.load(embeddings_path).astype('float32')
    dim = embeddings.shape[1]
    print(f"[Index] Embeddings shape: {embeddings.shape}")

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # Build index (exact nearest neighbors with Inner Product similarity)
    index = faiss.IndexFlatIP(dim)
    print("[Index] Adding vectors to the index ...")
    index.add(embeddings)

    # Save index
    os.makedirs("embeddings", exist_ok=True)
    faiss.write_index(index, index_path)
    print(f"[Index] Index saved to {index_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS index from embeddings")
    parser.add_argument("--model", type=str, default="vya",
                        help=f"Model alias or name. Aliases: {', '.join(MODEL_ALIASES.keys())}")
    args = parser.parse_args()

    # Map alias to full name if needed
    model_name = MODEL_ALIASES.get(args.model, args.model)

    build_faiss_index(model_name)
