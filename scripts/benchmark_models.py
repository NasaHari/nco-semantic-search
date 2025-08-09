import argparse
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
from typing import Dict, List

# --- MODEL CONFIG ---
MODEL_ALIASES = {
   # "vya": ("krutrim-ai-labs/Vyakyarth", "embeddings/nco_index_krutrim-ai-labs_Vyakyarth.faiss", "embeddings/nco_embeddings_krutrim-ai-labs_Vyakyarth.npy"),
    "min": ("sentence-transformers/all-MiniLM-L6-v2", "embeddings/nco_index_sentence-transformers_all-MiniLM-L6-v2.faiss", "embeddings/nco_embeddings_sentence-transformers_all-MiniLM-L6-v2.npy"),
    "lbs": ("sentence-transformers/LaBSE","embeddings/nco_index_sentence-transformers_LaBSE.faiss"," embeddings/nco_embeddings_sentence-transformers_LaBSE.npy")

}
BENCHMARK_QUERIES = [
    # English (10)
    "Statistical Data Analyst",
    "Data Science Consultant",
    "Retail Bank Cashier",
    "Healthcare Support Worker",
    "Registered Nurse",
    "Automobile Mechanic",
    "Application Software Engineer",
    "Elementary School Teacher",
    "Executive Chef",
    "Machine Learning Engineer",

    # Hindi (10)
    "सांख्यिकी डेटा विश्लेषक",
    "डेटा विज्ञान सलाहकार",
    "बैंक कैशियर",
    "स्वास्थ्य सहायक",
    "पंजीकृत नर्स",
    "गाड़ी मैकेनिक",
    "सॉफ़्टवेयर इंजीनियर",
    "प्राथमिक विद्यालय शिक्षक",
    "मुख्य रसोइया",
    "मशीन लर्निंग इंजीनियर",

    # Malayalam (10)
    "സംഖ്യാ ഡാറ്റ വിശകലന വിദഗ്ധൻ",
    "ഡാറ്റ സയൻസ് കൺസൾട്ടന്റ്",
    "ബാങ്ക് കാഷിയർ",
    "ആരോഗ്യ സഹായി",
    "രജിസ്റ്റർഡ് നഴ്‌സ്",
    "വാഹനം മെക്കാനിക്ക്",
    "ആപ്ലിക്കേഷൻ സോഫ്റ്റ്‌വെയർ എൻജിനീയർ",
    "പ്രാഥമിക വിദ്യാലയം അധ്യാപകൻ",
    "എക്സിക്യൂട്ടീവ് ഷെഫ്",
    "മെഷീൻ ലേണിംഗ് എഞ്ചിനീയർ",

    # Tamil (10)
    "கணினி தரவு பகுப்பாய்வு வல்லுநர்",
    "டேட்டா சயின்ஸ் ஆலோசகர்",
    "வங்கிக் காசியர்",
    "சுகாதார உதவியாளர்",
    "பதிவு செய்யப்பட்ட செவிலியர்",
    "வாகன மெக்கானிக்",
    "மென்பொருள் பொறியாளர்",
    "ஆரம்பப் பள்ளி ஆசிரியர்",
    "முதன்மை சமையல்காரர்",
    "மெஷின் லர்னிங் பொறியாளர்",

    # Telugu (10)
    "సంఖ్యాత్మక డేటా విశ్లేషకుడు",
    "డేటా సైన్స్ కన్సల్టెంట్",
    "బ్యాంకు కాషియర్",
    "ఆరోగ్య సహాయకుడు",
    "నమోదు నర్సు",
    "యంత్రం మెకానిక్",
    "సాఫ్ట్వేర్ ఇంజినీర్",
    "ప్రాథమిక పాఠశాల ఉపాధ్యాయుడు",
    "ప్రధాన వంటకారు",
    "మిషన్ లెర్నింగ్ ఇంజినీర్",

    # Kannada (10)
    "ಸಂಖ್ಯಾ ಡೇಟಾ ವಿಶ್ಲೇಷಕ",
    "ಡೇಟಾ ಸೈನ್ಸ್ ಸಲಹೆಗಾರ",
    "ಬ್ಯಾಂಕ್ ಕ್ಯಾಸಿಯರ್",
    "ಆರೋಗ್ಯ ಸಹಾಯಕ",
    "ನೋಂದಣಿಯ ನರ್ಸ್",
    "ಕಾರು ಮೆಕಾನಿಕ್",
    "ಸಾಫ್ಟ್‌ವೇರ್ ಇಂಜಿನಿಯರ್",
    "ಪ್ರಾಥಮಿಕ ಶಾಲೆ ಶಿಕ್ಷಕ",
    "ಮುಖ್ಯ ಬಂತುಹಾರ",
    "ಮೆಷಿನ್ ಲರ್ನಿಂಗ್ ಇಂಜಿನಿಯರ್",

    # Bengali (10)
    "পরিসংখ্যান ডেটা বিশ্লেষক",
    "ডেটা সায়েন্স পরামর্শক",
    "ব্যাংক ক্যাশিয়ার",
    "স্বাস্থ্য সহায়ক",
    "নিবন্ধিত নার্স",
    "গাড়ি মেকানিক",
    "সফটওয়্যার ইঞ্জিনিয়ার",
    "প্রাথমিক বিদ্যালয়ের শিক্ষক",
    "প্রধান রাঁধুনি",
    "মেশিন লার্নিং ইঞ্জিনিয়ার",

    # Marathi (10)
    "सांख्यिकी डेटा विश्लेषक",
    "डेटा सायन्स सल्लागार",
    "बँक कॅशियर",
    "आरोग्य सहाय्यक",
    "नोंदणीकृत नर्स",
    "गाडी मॅकेनिक",
    "सॉफ़्टवेअर अभियंता",
    "प्राथमिक शाळेचा शिक्षक",
    "मुख्य स्वयंपाकी",
    "मशीन लर्निंग अभियंता",
]




# --- SYNONYMS MAP ---
SYNONYMS = {
    "data analyst": ["statistician", "statistical assistant", "research analyst"],
    "sewing machine operator": ["tailor", "stitching machine operator", "garment worker"],
    "bank cashier": ["bank teller", "cashier bank", "bank clerk"],
    "healthcare assistant": ["general duty assistant", "medical assistant", "nursing attendant"],
    "mechanic": ["vehicle mechanic", "diesel mechanic", "automobile fitter"],
    "software developer": ["programmer", "application developer", "software engineer"],
    "teacher": ["school teacher", "lecturer", "instructor"],
    "graphic designer": ["visual designer", "creative designer", "illustrator"],
    "electrician": ["electrical technician", "wireman", "line man"],
    "driver": ["truck driver", "heavy vehicle driver", "bus driver"],
    "sales executive": ["salesman", "sales associate", "retail sales"],
    "cook": ["chef", "kitchen assistant", "culinary worker"]
}

TOP_K = 5
FUZZY_THRESHOLD = 85  # rapidfuzz score threshold

# --- LOAD INDEX ---
def load_index(index_path: str):
    return faiss.read_index(index_path)

# --- FLEXIBLE MATCH CHECK ---
def is_match(retrieved, ideal_list: List[str]) -> bool:
    if not isinstance(retrieved, str):
        retrieved = "" if pd.isna(retrieved) else str(retrieved)
    retrieved_lower = retrieved.lower()
    for ideal in ideal_list:
        score = fuzz.token_set_ratio(retrieved_lower, str(ideal).lower())
        if score >= FUZZY_THRESHOLD:
            return True
    return False


def benchmark_model(alias: str, csv_path: str) -> Dict[str, Dict]:
    model_name, index_path, embeddings_path = MODEL_ALIASES[alias]
    print(f"\n[Benchmark] Loading {alias} → {model_name}")
    model = SentenceTransformer(model_name)
    index = load_index(index_path)
    df = pd.read_csv(csv_path)
    if "Unit_Title" in df.columns:
        df["Unit_Title"] = df["Unit_Title"].fillna("").astype(str)
    if "text" in df.columns:
        df["text"] = df["text"].fillna("").astype(str)

    results = {}
    for query in BENCHMARK_QUERIES:
        query_vec = model.encode([query])
        scores, idxs = index.search(query_vec, TOP_K)

        if "Unit_Title" in df.columns:
            retrieved = [df["Unit_Title"].iloc[i] for i in idxs[0]]
        else:
            retrieved = [" ".join(df["text"].iloc[i].split()[:6]) for i in idxs[0]]

        results[query] = {
            "retrieved": retrieved
        }

    return results

def ensemble_retrieval(csv_path: str):
    df = pd.read_csv(csv_path)
    selected_aliases = ["lbs", "min"]  # Only these models

    models = {
        alias: SentenceTransformer(MODEL_ALIASES[alias][0], device="cpu")
        for alias in selected_aliases
    }
    indexes = {
        alias: load_index(MODEL_ALIASES[alias][1])
        for alias in selected_aliases  # <-- also restrict indexes here
    }
    results = {}
    for query in BENCHMARK_QUERIES:
        all_scores = {}
        for alias in selected_aliases:  # <-- loop over selected only
            q_vec = models[alias].encode([query])
            scores, idxs = indexes[alias].search(q_vec, TOP_K)
            for s, i in zip(scores[0], idxs[0]):
                title = df["Unit_Title"].iloc[i] if "Unit_Title" in df.columns else df["text"].iloc[i]
                all_scores[title] = all_scores.get(title, 0) + s  # sum scores for ranking

        sorted_items = sorted(all_scores.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
        retrieved = [title for title, _ in sorted_items]

        results[query] = {
            "retrieved": retrieved
        }
    return results



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="data/processed/nco_processed.csv")
    args = parser.parse_args()

    final_rows = []
    # for alias in MODEL_ALIASES:
    #     res = benchmark_model(alias, args.csv)
    #     for q, metrics in res.items():
    #         final_rows.append({
    #             "model": alias,
    #             "query": q,
    #             "retrieved": metrics["retrieved"]
    #         })

    ensemble_res = ensemble_retrieval(args.csv)
    for q, metrics in ensemble_res.items():
        final_rows.append({
            "model": "ensemble[vya+min]",
            "query": q,
            "retrieved": metrics["retrieved"]
        })

    df_out = pd.DataFrame(final_rows)
    df_out.to_csv("tests/benchmark_results.csv", index=False)
    print(df_out)


if __name__ == "__main__":
    main()
