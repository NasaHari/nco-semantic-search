import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
from langdetect import detect

# Short aliases for models
MODEL_ALIASES = {
    "vya": "krutrim-ai-labs/Vyakyarth",
    "l3c": "l3cube-pune/indic-sentence-similarity-sbert",
    "min": "sentence-transformers/all-MiniLM-L6-v2",
    "mpn": "sentence-transformers/all-mpnet-base-v2"
}

# Global cache of preloaded searchers (models)
_loaded_searchers = {}

class NCOSearcher:
    def __init__(self,
                 model_name: str = "krutrim-ai-labs/Vyakyarth",
                 embeddings_path: str = None,
                 index_path: str = None,
                 data_csv: str = "data/processed/nco_cleaned.csv"):
        # Use model alias if given
        model_name = MODEL_ALIASES.get(model_name, model_name)
        
        safe_model = model_name.replace("/", "_")
        if embeddings_path is None:
            embeddings_path = f"embeddings/nco_embeddings_{safe_model}.npy"
        if index_path is None:
            index_path = f"embeddings/nco_index_{safe_model}.faiss"

        if not all(map(os.path.exists, [embeddings_path, index_path, data_csv])):
            raise FileNotFoundError(
                f"Missing files for model '{model_name}'. "
                f"Expected:\n  {embeddings_path}\n  {index_path}\n  {data_csv}"
            )

        # Check if the model is already loaded from the cache
        if model_name not in _loaded_searchers:
            print(f"[Search] Loading SentenceTransformer model: {model_name}")
            self.model = SentenceTransformer(model_name)
            _loaded_searchers[model_name] = self.model
        else:
            self.model = _loaded_searchers[model_name]
            print(f"[Search] Using cached model: {model_name}")
        
        # Loading the rest of the data
        print("[Search] Loading NCO data CSV...")
        self.df = pd.read_csv(data_csv)
        print(f"[Search] Dataset loaded with {len(self.df)} entries.")

        print("[Search] Loading embeddings...")
        self.embeddings = np.load(embeddings_path)

        print("[Search] Loading FAISS index...")
        self.index = faiss.read_index(index_path)

        # Synonym corpus
        self.synonym_corpus = [
            'specialist', 'data', 'medical', 'production', 'evaluation',
            'standard', 'financing', 'developer', 'security', 'credit union',
            'knowledge', 'reviews', 'mechanic', 'industrial', 'ferrous',
            'savings institution', 'apparatus', 'operation', 'banking', 'healthcare'
        ]
        print("[Search] Encoding synonym corpus...")
        self.synonym_embeddings = self.model.encode(self.synonym_corpus)

    def search(self, query: str, top_k: int = 5, use_synonyms: bool = True):
        query_emb = self.model.encode([query])

        # Detect language
        try:
            lang = detect(query)
        except Exception:
            lang = "unknown"

        query_words = query.strip().split()
        apply_synonyms = use_synonyms and (len(query_words) < 3) and (lang == "en")

        # Apply synonym-based weighting if conditions are met
        if apply_synonyms and len(self.synonym_embeddings) > 0:
            sim_scores = np.dot(self.synonym_embeddings, query_emb[0]) / (
                np.linalg.norm(self.synonym_embeddings, axis=1) * np.linalg.norm(query_emb[0]) + 1e-10
            )
            print("Using synonyms for query enhancement...")
            top_syn_indices = np.argsort(sim_scores)[-3:]
            top_syn_embs = self.synonym_embeddings[top_syn_indices]

            weight_query = 0.7
            weight_syn = 0.3 / top_syn_embs.shape[0]
            weighted_sum = query_emb * weight_query + np.sum(top_syn_embs * weight_syn, axis=0, keepdims=True)
            augmented_emb = weighted_sum.astype('float32').reshape(1, -1)
        else:
            augmented_emb = query_emb.astype('float32').reshape(1, -1)

        # Perform search in the FAISS index
        distances, indices = self.index.search(augmented_emb, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.df):
                continue
            row = self.df.iloc[idx]
            corpus_emb = self.embeddings[idx]
            score = np.dot(augmented_emb[0], corpus_emb) / (
                np.linalg.norm(augmented_emb[0]) * np.linalg.norm(corpus_emb) + 1e-10
            )
            results.append({
                "Unit_Code": row.get("Unit_Code", ""),
                "Title": row.get("Unit_Title", ""),
                "Description": row.get("Unit_Description", ""),
                "Score": float(score)
            })

        # Sort results by score
        results.sort(key=lambda x: x["Score"], reverse=True)
        return results


# Function to preload all search models
def load_all_searchers(model_names):
    """ Preload all models to avoid loading each time on request. """
    searchers = {}
    for model_name in model_names:
        searchers[model_name] = NCOSearcher(model_name=model_name)
    print("[Init] All requested searchers loaded.")

    return searchers
