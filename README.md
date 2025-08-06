# AI-Enabled Semantic Search for NCO

## Overview
This Proof-of-Concept (PoC) implements an AI-driven semantic search for the National Classification of Occupation (NCO-2015). It ingests NCO data, generates embeddings using Sentence Transformers, indexes with FAISS, and provides a Streamlit UI for querying occupations semantically (e.g., "sewing machine operator" matches relevant codes with confidence scores).

Built for STATATHON PS-5. Key features: Data preprocessing, embedding generation, fast similarity search, and basic UI.

## Setup
1. Clone the repo: `git clone <your-repo-url>`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Download NCO data: Place raw files (e.g., NCO-2015.pdf) in `data/raw/`.
4. Run preprocessing: `python scripts/preprocess.py`.
5. Generate embeddings and index: `python scripts/embed.py` then `python scripts/index.py`.
6. Launch app: `streamlit run app/app.py`.

## Usage
- Query via UI: Enter a job description and get ranked NCO matches.
- CLI example: `python scripts/search.py --query "doctor"`.
- Update data: `./scripts/update_data.sh`.

## Development
- Experiment in `notebooks/`.
- Lint code: `flake8 scripts/`.
- Format: `black .`.
- Run tests: `pytest`.

## License
MIT License.
