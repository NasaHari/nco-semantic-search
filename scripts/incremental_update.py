import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import pandas as pd
import re
import unicodedata

# Load the CSV
df = pd.read_csv('/home/harikrishnan/Statathon/nco-semantic-search/data/processed/nco_cleaned.csv')

# List the text columns you want to normalize
text_columns = [
    'Unit_Title', 'Unit_Description', 'Division', 'Division_Description',
    'Sub_Division', 'Sub_Division_Description', 'Group', 'Group_Description',
    'Family_Description', 'text'
    # Add/remove columns as needed
]

def normalize_text(text):
    if pd.isna(text): return ""
    text = str(text)
    # Lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Standardize punctuation spacing
    text = re.sub(r'\s*([,./\-:;])\s*', r'\1 ', text)
    text = re.sub(r'\s+', ' ', text)
    # Remove non-ASCII/control characters but keep Indian languages
    text = ''.join([c for c in text if unicodedata.category(c)[0] != 'C'])
    # Remove leading/trailing spaces
    text = text.strip()
    return text




def incremental_update(
    data_csv: str = "data/processed/nco_cleaned.csv",
    embeddings_path: str = "embeddings/nco_embeddings_vya.npy",
    index_path: str = "embeddings/nco_index-vya.faiss",
    model_name: str = "krutrim-ai-labs/Vyakyarth",  # Match your model
    new_rows: list[dict] = None  # List of new row dicts (e.g., from Streamlit form)
) -> None:
    """
    Incrementally update CSV, embeddings, and FAISS index with new data.
    
    Args:
        data_csv: Path to existing CSV.
        embeddings_path: Path to existing embeddings .npy.
        index_path: Path to existing FAISS index.
        model_name: SentenceTransformer model.
        new_rows: List of dicts with new data (must include 'text' for embedding).
    """
    if new_rows is None or not new_rows:
        print("No new rows provided. Nothing to update.")
        return
    
    # Load existing data
    if not os.path.exists(data_csv):
        raise FileNotFoundError(f"CSV not found: {data_csv}")
    df = pd.read_csv(data_csv)
    
    # Append new rows
    new_df = pd.DataFrame(new_rows)
    for col in text_columns:
        if col in new_df:
            df[col] = new_df[col].apply(normalize_text)
    if 'text' not in new_df.columns:
        raise ValueError("New rows must include 'text' column for embeddings.")
    df = pd.concat([df, new_df], ignore_index=True)
    df.to_csv(data_csv, index=False)
    print(f"[Update] Appended {len(new_rows)} new rows to {data_csv}. Total rows: {len(df)}")

    # Load model
    model = SentenceTransformer(model_name)

    # Generate embeddings ONLY for new data
    new_texts = new_df['text'].tolist()
    new_embeddings = model.encode(new_texts, show_progress_bar=True).astype('float32')
    
    # Normalize if using cosine (match your setup)
    faiss.normalize_L2(new_embeddings)

    # Load and append to existing embeddings
    if os.path.exists(embeddings_path):
        existing_embeddings = np.load(embeddings_path)
        updated_embeddings = np.vstack([existing_embeddings, new_embeddings])
    else:
        updated_embeddings = new_embeddings
    np.save(embeddings_path, updated_embeddings)
    print(f"[Update] Added {len(new_embeddings)} new embeddings to {embeddings_path}. Total: {len(updated_embeddings)}")

    # Load existing index and add new vectors
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Index not found: {index_path}")
    index = faiss.read_index(index_path)
    index.add(new_embeddings)
    faiss.write_index(index, index_path)
    print(f"[Update] Added {len(new_embeddings)} vectors to index {index_path}. Total vectors: {index.ntotal}")


