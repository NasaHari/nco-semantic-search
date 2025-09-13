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
    "min": "/home/harikrishnan/Coding/Statathon/nco-semantic-search/models/finetunedenglishmodel",
"mino":'sentence-transformers/all-MiniLM-L6-v2',    "mpn": "sentence-transformers/all-mpnet-base-v2",
    "lbs": "sentence-transformers/LaBSE","sml":"intfloat/multilingual-e5-small","bge":"BAAI/bge-small-en-v1.5"
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
        preprocess_and_save(data_csv)

        if embeddings_path is None:
            embeddings_path = f"embeddings/nco_embeddings_{safe_model}.npy"
        if index_path is None:
            index_path = f"embeddings/nco_index_{safe_model}.faiss"

        if not os.path.exists(embeddings_path):
               generate_embeddings(data_csv, embeddings_path, model_name)
        if not os.path.exists(index_path):
            build_faiss_index(model_name)

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

                threshold = 0.7
                filtered_synonyms = [s for s, sim in zip(synonym_list, similarities) if sim > threshold]
               # print(f"Using synonyms for query enhancement: {filtered_synonyms}")

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

    


def load_all_searchers(model_names):
    searchers = {}
    for model_name in model_names:
        searchers[model_name] = NCOSearcher(model_name=model_name)
    print("[Init] All requested searchers loaded.")
    return searchers

import time
import psutil
import numpy as np
import threading
import argparse
import contextlib
import os
import sys

@contextlib.contextmanager
def suppress_stdout():
    """Context manager to suppress print statements."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def run_queries(searcher, queries, top_k=5, use_synonyms=True):
    process = psutil.Process()
    first=True
    for query in queries:
        # Track CPU usage in a separate thread
        cpu_samples = []
        stop_event = threading.Event()

        def monitor_cpu(interval=0.05):
            psutil.cpu_percent(interval=None)  # Warm-up

            while not stop_event.is_set():
                cpu_samples.append(psutil.cpu_percent(interval=interval))

        cpu_thread = threading.Thread(target=monitor_cpu)
        cpu_thread.start()

        # Track memory before search
        mem_before = process.memory_info().rss

        # Run search
        start_time = time.perf_counter()
        results, fallback_suggestions = searcher.search(
            query, top_k=top_k, use_synonyms=use_synonyms
        )
        end_time = time.perf_counter()

        # Stop CPU monitoring
        stop_event.set()
        cpu_thread.join()

        mem_after = process.memory_info().rss

        # Compute metrics
        latency = end_time - start_time
        avg_cpu = np.mean(cpu_samples) if cpu_samples else 0
        mem_used_mb = (mem_after - mem_before) / (1024 ** 2)
        
        # Print results
        print(f"\nTop {top_k} results for query: \"{query}\"")
        for i, res in enumerate(results, 1):
            print(f"{i}. {res['Title']} (Confidence Score: {res['Score']:.2f}%)")

        if fallback_suggestions:
            print("Did you mean one of these?")
            for i, suggestion in enumerate(fallback_suggestions, 1):
                print(f"{i}. {suggestion}")

        print(f"\n--- Efficiency Metrics ---")
        print(f"Latency: {latency:.3f} s")
        if first:
            print(f"Average CPU: {avg_cpu:.2f} %")
            print(f"Memory used: {mem_used_mb:.2f} MB")
            first=False
        print("-" * 50)

def main(model_name="vya", queries=None, top_k=5, use_synonyms=True, 
         embeddings_path=None, index_path=None, data_csv="data/processed/nco_cleaned.csv"):
    
    if queries is None:
        queries = [
            "software engineer",
            "data analyst",
            "marketing manager",
            "teacher",
            "civil engineer"
        ]

    # --- Initialize searcher silently ---
    with suppress_stdout():
        searcher = NCOSearcher(
            model_name=model_name,
            embeddings_path=embeddings_path,
            index_path=index_path,
            data_csv=data_csv
        )
    print(f"MODEL={model_name}")
    # --- Run queries ---
    run_queries(searcher, queries, top_k=top_k, use_synonyms=use_synonyms)
hindi_queries = [
    "सॉफ्टवेयर इंजीनियर",
    "डेटा एनालिस्ट",
    "स्कूल टीचर",
    "किसान मजदूर",
    "अस्पताल में डॉक्टर",
    "व्यक्ति जो ऐप्स कोड करता है और बग्स ठीक करता है",
    "एनालिस्ट जो बिजनेस इनसाइट्स के लिए नंबर्स क्रंच करता है",
    "शिक्षक जो बच्चों को गणित और विज्ञान पढ़ाता है",
    "किसान जो फसलें उगाता है और पशुओं का प्रबंधन करता है",
    "मेडिकल प्रोफेशनल जो मरीजों का सर्जरी से इलाज करता है",
    "इंजीनियर जो सर्किट डिजाइन करता है और एम्बेडेड सिस्टम बनाता है",
    "शोधकर्ता जो मशीन लर्निंग मॉडल्स को स्पीड के लिए ऑप्टिमाइज करता है",
    "कलाकार जो डिजिटल इलस्ट्रेशन्स और एनिमेशन्स बनाता है",
    "मैनेजर जो सेल्स टीमों और मार्केटिंग कैंपेन का निरीक्षण करता है",
    "मैकेनिक जो कारों और भारी मशीनरी की मरम्मत करता है",
    "नाभिकीय भौतिक विज्ञानी",
    "वैज्ञानिक जो नाभिकीय भौतिकी और परमाणु कणों का अध्ययन करता है",
    "वैज्ञानिक जो चिकित्सीय और ऊर्जा अनुप्रयोगों के लिए नाभिकीय अभिक्रियाओं का विश्लेषण करता है",
    "व्यक्ति जो समस्थानिकों और एक्स-रे क्रिस्टलोग्राफी के साथ प्रयोग करता है",
    "वैज्ञानिक जो परमाणु कार्य करता है",
    "परमाणु भौतिकी में शोधकर्ता",
    "नाभिकीय प्रयोगों के लिए वैज्ञानिक"
]
english_queries = [
    "Software Engineer",
    "Data Analyst",
    "School Teacher",
    "Farm Worker",
    "Doctor in Hospital",
    "Person who codes apps and fixes bugs",
    "Analyst who crunches numbers for business insights",
    "Teacher educating kids in math and science",
    "Farmer growing crops and managing livestock",
    "Medical professional treating patients with surgery",
    "Engineer designing circuits and embedded systems",
    "Researcher optimizing machine learning models for speed",
    "Artist creating digital illustrations and animations",
    "Manager overseeing sales teams and marketing campaigns",
    "Mechanic repairing cars and heavy machinery",
    "Nuclear Physicist",
    "Scientist who studies nuclear physics and atomic particles",
    "Scientist analyzing nuclear reactions for medical and energy applications",
    "Person performing experiments with isotopes and X-ray crystallography",
    "Scientist with atom work",
    "Researcher in atomic physics",
    "Scientist for nuclear experiments"
]
tamil_queries = [
    "சாப்ட்வேர் பொறியாளர்",
    "தரவு பகுப்பாய்வாளர்",
    "பள்ளி ஆசிரியர்",
    "திருப்பணியாளர்",
    "மருத்துவர்",
    "ஆப்ஸ்களை குறியீட்டு செய்பவர் மற்றும் பிழைகளை சரிசெய்பவர்",
    "கணிதம் மற்றும் அறிவியல் கற்பிக்கும் ஆசிரியர்",
    "சாகுபடிகள் வளர்க்கும் விவசாய்",
    "மருத்துவம் செய்யும் நிபுணர்",
    "சர்க்கிட் வடிவமைக்கிறார் மற்றும் உள்ளமைக்கப்பட்ட அமைப்புகளை உருவாக்குகிறார் என்ஜினியர்",
    "மெஷின் கற்றல் மாதிரிகளை வேகமாக உறுதிப்படுத்துவதில் ஆராய்ச்சியாளர்கள்",
    "டிஜிட்டல் ஓவியங்கள் மற்றும் அனிமேஷன்களை உருவாக்கும் கலைஞர்",
    "கார்கள் மற்றும் கனமான இயந்திரங்களைக் குணப்படுத்தும் மெக்கானிக்",
    "அணி நाभிகி ஆய்வாளர்",
    "நाभிகி மற்றும் அணு கூறுகளைப் படிப்பவர் விஞ்ஞானி",
    "மருத்துவ மற்றும் ஆற்றல் பயன்பாடுகளுக்காக அணு எதிர்வினைகளை பகுப்பாய்வு செய்வவர் விஞ்ஞானி",
    "இசோடோப்கள் மற்றும் எக்ஸ்-ரே கிரிஸ்டலோகிராஃபி நிகழ்த்தும் நபர்",
    "அணுக் காரியங்களில் ஆராய்ச்சியாளர்",
    "அணு நிபுணரான ஆராய்ச்சியாளர்",
    "நுபிக்கல் நிகழ்வுகளுக்கான விஞ்ஞானி"
]
# Example usage:
if __name__ == "__main__":
    main(
        model_name="sml",
        queries= hindi_queries
,
        top_k=5,
        use_synonyms=0
    )