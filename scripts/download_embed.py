from pathlib import Path
import numpy as np
import requests

embeddings_info = {
    "nco_all-MiniLM-L6-v2": "https://huggingface.co/datasets/hariv2/embeddings/resolve/main/nco_embeddings_sentence-transformers_all-MiniLM-L6-v2.npy",
    "nco_LaBSE": "https://huggingface.co/datasets/hariv2/embeddings/resolve/main/nco_embeddings_sentence-transformers_LaBSE.npy",
    "synonym_all-MiniLM-L6-v2": "https://huggingface.co/datasets/hariv2/embeddings/resolve/main/synonym_embeddings_sentence-transformers_all-MiniLM-L6-v2.npy",
    "synonym_LaBSE": "https://huggingface.co/datasets/hariv2/embeddings/resolve/main/synonym_embeddings_sentence-transformers_LaBSE.npy",
}

embeddings_folder = Path("embeddings")
embeddings_folder.mkdir(parents=True, exist_ok=True)

loaded_embeddings = {}
for name, url in embeddings_info.items():
    file_path = embeddings_folder / f"{name}.npy"
    
    if not file_path.exists():
        print(f"Downloading {name}...")
        r = requests.get(url)
        r.raise_for_status()
        file_path.write_bytes(r.content)
    
    loaded_embeddings[name] = np.load(file_path)
    print(f"{name} loaded! Shape: {loaded_embeddings[name].shape}")
