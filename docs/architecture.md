# System Architecture

## Overview
This PoC follows a modular pipeline: Data Ingestion → Preprocessing → Embedding Generation → Indexing → Query Processing → UI Rendering.

## Diagram
[Insert Draw.io link or embed image here] (e.g., create via draw.io and export as PNG to docs/architecture.png)

## Components
- **Preprocessing**: PDF extraction and text cleaning.
- **Embeddings**: Sentence Transformers for semantic vectors.
- **Indexing**: FAISS for efficient similarity search.
- **Search**: Query encoding and ranking with confidence scores.
- **UI**: Streamlit for interactive queries.

## Data Flow
1. Raw NCO PDF → Processed CSV.
2. CSV → Embeddings (.npy) → FAISS Index.
3. User Query → Encoded → Searched → JSON Results.
