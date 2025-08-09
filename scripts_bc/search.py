import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import os
from langdetect import detect

class NCOSearcher:
    def __init__(self,
                 model_name: str = "krutrim-ai-labs/Vyakyarth",
                 embeddings_path: str = "embeddings/nco_embeddings_vya.npy",
                 index_path: str = "embeddings/nco_index-vya.faiss",
                 data_csv: str = "data/processed/nco_cleaned.csv"):
        if not all(map(os.path.exists, [embeddings_path, index_path, data_csv])):
            raise FileNotFoundError("Ensure embeddings, index and CSV files exist.")
        
        print("[Search] Loading SentenceTransformer model...Backup")
        self.model = SentenceTransformer(model_name)
        print("[Search] Loading embeddings...")
        self.embeddings = np.load(embeddings_path)  # shape: (num_entries, dim)
        print("[Search] Loading FAISS index...")
        self.index = faiss.read_index(index_path)
        print("[Search] Loading NCO data CSV...")
        self.df = pd.read_csv(data_csv)
        print(f"[Search] Dataset loaded with {len(self.df)} entries.")
        
        # Pre-build synonym corpus (expand this list with 100-500 NCO phrases)
        self.synonym_corpus = ['specialist', 'data', 'medical', 'production', 'evaluation', 'standard', 'financing', 'developer', 'security', 'credit union', 'knowledge', 'reviews', 'mechanic', 'industrial', 'ferrous', 'savings institution', 'apparatus', 'operation', 'banking', 'healthcare']
        print("[Search] Encoding synonym corpus...")
        self.synonym_embeddings = self.model.encode(self.synonym_corpus)  # Pre-encode for efficiency

    def search(self, query: str, top_k: int = 5) -> str:
        use_synonyms=True
        query_emb = self.model.encode([query])
        try:
            lang = detect(query)
        except Exception:
            lang = "unknown"
        query_words = query.strip().split()
        apply_synonyms = use_synonyms and (len(query_words) < 3)  and (lang == "en")
        
        if apply_synonyms and hasattr(self, "synonym_embeddings") and len(self.synonym_embeddings) > 0:
            sim_scores = np.dot(self.synonym_embeddings, query_emb[0]) / (
                np.linalg.norm(self.synonym_embeddings, axis=1) * np.linalg.norm(query_emb[0]) + 1e-10
            )
            print("usingsyn")
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
        return json.dumps(results, indent=4)

# Global instance (singleton) remains the same
_searcher_instance = None

def get_searcher():
    global _searcher_instance
    if _searcher_instance is None:
        _searcher_instance = NCOSearcher()
    return _searcher_instance

def search_nco(query: str, top_k: int = 5) -> str:
    searcher = get_searcher()
    return searcher.search(query, top_k)
import argparse


BENCHMARK_QUERIES = [
    "data analyst",
    "statistical data processing",
    "sewing machine operator",
    "bank teller",
    "healthcare assistant",
    "mechanic",
    "software developer",
    "quality control inspector",
    "credit union officer",
    "industrial electrician",
    "tailor and dressmaker",
    "machine operator in textile industry",
    "nurse",
    "data entry operator",
    "health insurance claims processor",
    "construction laborer",
    "customer service representative",
    "project manager",
    "financial analyst",
    "social worker",
    "agricultural worker"
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search NCO dataset using a query or benchmark queries.")
    parser.add_argument("--q", type=str, help="Query string for searching. Separate multiple queries with commas.")
    parser.add_argument("--top_k", type=int, default=3, help="Number of top results to return.")
    parser.add_argument("--benchmark", action="store_true", help="Run built-in benchmark queries.")
    args = parser.parse_args()

    if args.benchmark:
        queries = BENCHMARK_QUERIES
    elif args.q:
        queries = [q.strip() for q in args.q.split(",") if q.strip()]
    else:
        print("Please provide --q for a single query or --benchmark for the test set.")
        exit(1)

    for query in queries:
        print(f"\n=== Results for query: \"{query}\" ===")
        results_json = search_nco(query, top_k=args.top_k)
        results = json.loads(results_json)
        for i, res in enumerate(results):
            print(f"{i+1}. {res['Title']}: {res['Score']:.4f}")