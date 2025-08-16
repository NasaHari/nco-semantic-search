import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pandas as pd
from langdetect import detect
import argparse
from typing import List, Dict
from collections import defaultdict
import json
from rapidfuzz import fuzz
from sklearn.metrics.pairwise import cosine_similarity
from .preprocess import  preprocess_and_save
from .index  import build_faiss_index
from .embed import   *

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
                 data_csv: str = "data/processed/nco_cleaned.csv",synonym_file: str = "data/processed/synonyms.json"):
        model_name = MODEL_ALIASES.get(model_name, model_name)
        
        safe_model = model_name.replace("/", "_")
        if embeddings_path is None:
            embeddings_path = f"embeddings/nco_embeddings_{safe_model}.npy"
        if index_path is None:
            index_path = f"embeddings/nco_index_{safe_model}.faiss"

        if not os.path.exists(embeddings_path):
               generate_embeddings(data_csv, embeddings_path, model_name)
        if not os.path.exists(index_path):
            build_faiss_index(model_name)

        preprocess_and_save(data_csv)
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
        self.check_index_alignment()

        try:
            with open(synonym_file, 'r') as f:
                self.synonym_corpus = json.load(f)  # Dictionary of term: [synonyms]
            print("[Search] Loaded synonym corpus from JSON")
        except FileNotFoundError:
            print("[Search] Synonym file not found, using empty synonym corpus")
            self.synonym_corpus = {}
        syn_embeddings_path = f"embeddings/synonym_embeddings_{safe_model}.npy"


        try:
            self.synonym_embeddings = np.load(
               syn_embeddings_path, 
                allow_pickle=True
            ).item()
            print("[Search] Loaded precomputed synonym embeddings")
        except FileNotFoundError:
            print("[Search] Precomputed synonym embeddings not found, encoding now...")
            self.synonym_embeddings = {}
            for lang, groups in self.synonym_corpus.items():
                self.synonym_embeddings[lang] = []
                for group in groups:
                    emb = self.model.encode(group)
                    self.synonym_embeddings[lang].append({"words": group, "embeddings": emb})
            np.save(syn_embeddings_path, self.synonym_embeddings)
            print(f"[Search] Saved newly encoded synonym embeddings to {syn_embeddings_path}")
        print("[Search] Synonym embeddings ready")


    def search(self, query: str, top_k: int = 5, use_synonyms: bool = True, fallback_threshold: float = 67):
        INDIC_LANGS = {"hi", "ta", "te", "kn", "ml", "bn", "mr"}
        query_emb = self.model.encode([query])

        try:
            lang = detect(query)
            if lang not in INDIC_LANGS:
                lang = "en"
            print(f'[Search] Language = {lang}')
        except Exception:
            lang = "unknown"

        # apply_synonyms = use_synonyms and (len(query_words) < 3) and (lang == "en")
        apply_synonyms = use_synonyms and len(self.synonym_corpus.get(lang, [])) > 0

        if apply_synonyms:
            print(f'[Search] Using synonyms')
            synonym_list = []
            syn_emb_list = []

            for group_dict in self.synonym_embeddings.get(lang, []):
                words = group_dict["words"]
                emb = group_dict["embeddings"]

                # Check if query matches any word in this group
                if any(fuzz.partial_ratio(word.lower(), query.lower()) > 85 for word in words):
                    # Use all other words in this group
                    other_words = [w for w in words if w.lower() != query.lower()]
                    if other_words:
                        synonym_list.extend(other_words)
                        # Get embeddings of these words
                        indices = [i for i, w in enumerate(words) if w.lower() != query.lower()]
                        syn_emb_list.append(emb[indices])

            if synonym_list:
                # Stack all synonym embeddings together
                syn_embs = np.vstack(syn_emb_list)
                # Compute similarity to query
                similarities = cosine_similarity(query_emb.reshape(1, -1), syn_embs)[0]

                threshold = 0.6
                filtered_synonyms = [s for s, sim in zip(synonym_list, similarities) if sim > threshold]
                print(f"Using synonyms for query enhancement: {filtered_synonyms}")

                if filtered_synonyms:
                    # Get embeddings of filtered synonyms (already precomputed)
                    filtered_embs = []
                    for group_dict in self.synonym_embeddings.get(lang, []):
                        for i, w in enumerate(group_dict["words"]):
                            if w in filtered_synonyms:
                                filtered_embs.append(group_dict["embeddings"][i])
                    filtered_embs = np.vstack(filtered_embs)

                    weight_query = 0.7
                    weight_syn = 0.3 / len(filtered_embs) if filtered_embs.size > 0 else 0
                    weighted_sum = query_emb * weight_query + np.sum(filtered_embs * weight_syn, axis=0, keepdims=True)
                    augmented_emb = weighted_sum.astype('float32').reshape(1, -1)
                else:
                    augmented_emb = query_emb.astype('float32').reshape(1, -1)
            else:
                augmented_emb = query_emb.astype('float32').reshape(1, -1)
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
            # Normalize score to 0-100% (cosine similarity ranges from -1 to 1)
            normalized_score = (score + 1) / 2 * 100  # Map [-1,1] to [0,100]
            results.append({
                "Unit_Code": row.get("Unit_Code", ""),
                "Title": row.get("Unit_Title", ""),
                "Description": row.get("Unit_Description", ""),
                "Score": normalized_score
            })

        # Sort results by score
        results.sort(key=lambda x: x["Score"], reverse=True)

        # Generate fallback suggestions if results are poor
        if not results or max(r["Score"] for r in results) < fallback_threshold:  # Threshold for poor results
            fallback_suggestions = self.get_fallback_suggestions(query, lang, top_k)
            return results, fallback_suggestions
        return results, []
    
    def check_index_alignment(self, sample_size=5):
        print("\n[Check] Verifying FAISS index ↔ DataFrame alignment...")
        num_vectors = self.index.ntotal
        num_rows = len(self.df)

        if num_vectors != num_rows:
            print(f"[Warning] Index has {num_vectors} vectors but DataFrame has {num_rows} rows!")
            print("         This usually means the CSV has changed but embeddings/index were not rebuilt.")
            return False

        # Sample a few random rows from FAISS and compare with DataFrame
        import random
        sample_idxs = random.sample(range(num_rows), min(sample_size, num_rows))

        for idx in sample_idxs:
            row = self.df.iloc[idx]
            corpus_emb = self.embeddings[idx]

            # Search using the exact embedding for this row
            distances, indices = self.index.search(corpus_emb.reshape(1, -1), 1)
            faiss_idx = int(indices[0][0])

            if faiss_idx != idx:
                print(f"[Mismatch] DF row {idx} ({row.get('Unit_Code')}) "
                    f"→ FAISS thinks it’s row {faiss_idx} ({self.df.iloc[faiss_idx].get('Unit_Code')})")
                return False

            print("[Check] FAISS index and DataFrame appear aligned ✅")
            return True

    def get_fallback_suggestions(self, query: str, lang: str, top_k: int) -> List[str]:
        suggestions = []

        # Try to match Unit_Title
        for title in self.df["Unit_Title"].dropna():
            score = fuzz.partial_ratio(query.lower(), title.lower())
            suggestions.append((title, score))

        
        # If still no suggestions, match description
        if not suggestions:
            for desc in self.df["Unit_Description"].dropna():
                score = fuzz.partial_ratio(query.lower(), desc.lower())
                suggestions.append((title, score))

        
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return [s[0] for s in suggestions[:top_k]]

    
def ensemble_search(searchers: Dict[str, NCOSearcher], query: str, top_k: int = 5, use_synonyms: bool = True):
    all_results = {}
    for model_name, searcher in searchers.items():
        results, _ = searcher.search(query, top_k=top_k * 2, use_synonyms=use_synonyms)
        all_results[model_name] = results

    # Combine using Reciprocal Rank Fusion (RRF)
    fusion_scores = defaultdict(float)
    for model_results in all_results.values():
        for rank, res in enumerate(model_results, 1):
            key = (res["Unit_Code"], res["Title"])
            fusion_scores[key] += 1.0 / rank

    sorted_items = sorted(fusion_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    ensemble_results = []
    for (code, title), fused_score in sorted_items:
        sample_res = next((r for results in all_results.values() for r in results if r["Unit_Code"] == code), None)
        if sample_res:
            # Normalize fused score (assuming max RRF score ~1 for top_k=5)
            sample_res["Score"] = min(fused_score * 50, 100)  # Rough normalization to 0-100
            ensemble_results.append(sample_res)

    # Fallback suggestions if ensemble results are poor
    first_searcher = list(searchers.values())[0]
    if not ensemble_results or max(r["Score"] for r in ensemble_results) < 60:
        fallback_suggestions = first_searcher.get_fallback_suggestions(query, detect(query), top_k)
        return ensemble_results, fallback_suggestions
    return ensemble_results, []

def load_all_searchers(model_names):
    searchers = {}
    for model_name in model_names:
        searchers[model_name] = NCOSearcher(model_name=model_name)
    print("[Init] All requested searchers loaded.")
    return searchers

def main():
    parser = argparse.ArgumentParser(description="NCO Search CLI")
    parser.add_argument("--model", type=str, choices=MODEL_ALIASES.keys(), default="vya")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--no_synonyms", action="store_true")
    parser.add_argument("--data_csv", type=str, default="data/processed/nco_cleaned.csv")
    parser.add_argument("--embeddings_path", type=str, default=None)
    parser.add_argument("--index_path", type=str, default=None)
    
    args = parser.parse_args()

    searcher = NCOSearcher(
        model_name=args.model,
        embeddings_path=args.embeddings_path,
        index_path=args.index_path,
        data_csv=args.data_csv
    )

    results, fallback_suggestions = searcher.search(
        args.query, top_k=args.top_k, use_synonyms=not args.no_synonyms
    )

    print(f"\nTop {args.top_k} results for query: \"{args.query}\"")
    for i, res in enumerate(results, 1):
        print(f"{i}. {res['Title']} (Confidence Score: {res['Score']:.2f}%)")
        print(f"   Code: {res['Unit_Code']}")
        print(f"   Description: {res['Description']}\n")
    
    if fallback_suggestions:
        print("Did you mean one of these?")
        for i, suggestion in enumerate(fallback_suggestions, 1):
            print(f"{i}. {suggestion}")

if __name__ == "__main__":
    main()