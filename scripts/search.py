import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
from langdetect import detect
import argparse
from typing import List, Dict
from collections import defaultdict

MODEL_ALIASES = {
    "vya": "krutrim-ai-labs/Vyakyarth",
    "l3c": "l3cube-pune/indic-sentence-similarity-sbert",
    "min": "sentence-transformers/all-MiniLM-L6-v2",
    "mpn": "sentence-transformers/all-mpnet-base-v2",
    "lbs": "sentence-transformers/LaBSE"
}

_loaded_searchers = {}

class NCOSearcher:
    def __init__(self,
                 model_name: str = "krutrim-ai-labs/Vyakyarth",
                 embeddings_path: str = None,
                 index_path: str = None,
                 data_csv: str = "data/processed/nco_cleaned.csv"):
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

        if model_name not in _loaded_searchers:
            print(f"[Search] Loading SentenceTransformer model: {model_name}")
            self.model = SentenceTransformer(model_name)
            _loaded_searchers[model_name] = self.model
        else:
            self.model = _loaded_searchers[model_name]
            print(f"[Search] Using cached model: {model_name}")
        
        print("[Search] Loading NCO data CSV...")
        self.df = pd.read_csv(data_csv)
        print(f"[Search] Dataset loaded with {len(self.df)} entries.")

        print("[Search] Loading embeddings...")
        self.embeddings = np.load(embeddings_path)

        print("[Search] Loading FAISS index...")
        self.index = faiss.read_index(index_path)

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

        try:
            lang = detect(query)
        except Exception:
            lang = "unknown"

        query_words = query.strip().split()
        apply_synonyms = use_synonyms and (len(query_words) < 3) and (lang == "en")

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

        results.sort(key=lambda x: x["Score"], reverse=True)
        return results

def ensemble_search(searchers: Dict[str, NCOSearcher], query: str, top_k: int = 5, use_synonyms: bool = True):
    """
    Perform ensemble search across multiple models and combine results.
    Uses reciprocal rank fusion to merge rankings.
    """
    all_results = {}
    for model_name, searcher in searchers.items():
        results = searcher.search(query, top_k=top_k * 2, use_synonyms=use_synonyms)  # Fetch more to merge
        all_results[model_name] = results

    # Combine using Reciprocal Rank Fusion (RRF)
    fusion_scores = defaultdict(float)
    for model_results in all_results.values():
        for rank, res in enumerate(model_results, 1):
            key = (res["Unit_Code"], res["Title"])  # Unique key per occupation
            fusion_scores[key] += 1.0 / rank  # RRF score

    # Sort by fused score
    sorted_items = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Build final results list
    ensemble_results = []
    for (code, title), fused_score in sorted_items:
        # Get full details from one of the models (e.g., first one)
        sample_res = next((r for results in all_results.values() for r in results if r["Unit_Code"] == code), None)
        if sample_res:
            sample_res["Score"] = fused_score  # Use fused score
            ensemble_results.append(sample_res)

    return ensemble_results

def load_all_searchers(model_names):
    searchers = {}
    for model_name in model_names:
        searchers[model_name] = NCOSearcher(model_name=model_name)
    print("[Init] All requested searchers loaded.")
    return searchers

def main():
    parser = argparse.ArgumentParser(description="NCO Search CLI")
    parser.add_argument("--model", type=str, choices=MODEL_ALIASES.keys(), default="vya",
                        help="Model alias to use for search")
    parser.add_argument("--query", type=str, required=True,
                        help="Search query text")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top results to return")
    parser.add_argument("--no_synonyms", action="store_true",
                        help="Disable synonym-based query enhancement")
    parser.add_argument("--data_csv", type=str, default="data/processed/nco_cleaned.csv",
                        help="Path to cleaned NCO CSV data file")
    parser.add_argument("--embeddings_path", type=str, default=None,
                        help="Path to precomputed embeddings numpy file")
    parser.add_argument("--index_path", type=str, default=None,
                        help="Path to FAISS index file")
    
    args = parser.parse_args()

    searcher = NCOSearcher(
        model_name=args.model,
        embeddings_path=args.embeddings_path,
        index_path=args.index_path,
        data_csv=args.data_csv
    )

    results = searcher.search(args.query, top_k=args.top_k, use_synonyms=not args.no_synonyms)

    print(f"\nTop {args.top_k} results for query: \"{args.query}\"")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['Title']} (Score: {res['Score']:.4f})")
        print(f"   Code: {res['Unit_Code']}")
        print(f"   Description: {res['Description']}\n")

if __name__ == "__main__":
    main()
