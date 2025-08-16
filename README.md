# AI-Enabled Semantic Search for NCO

## Overview
This Proof-of-Concept (PoC) implements an AI-driven semantic search for the National Classification of Occupation (NCO-2015). It ingests NCO data, generates embeddings using Sentence Transformers, indexes with FAISS, and provides a Streamlit UI for querying occupations semantically (e.g., "sewing machine operator" matches relevant codes with confidence scores).

Built for STATATHON PS-5. Key features: Data preprocessing, embedding generation, fast similarity search, and basic UI.

## Setup
1. Clone the repo: `git clone https://github.com/NasaHari/nco-semantic-search.git`.
2. Create a vitual envinroment 
3. Activate the env,inside the env
4. Install dependencies: `pip install -r requirements.txt`.
5. Launch app: `streamlit run app/app.py`.

## Usage
- Query via UI: Enter a job description and get ranked NCO matches.



## License
MIT License.
