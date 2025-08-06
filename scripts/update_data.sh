#!/bin/bash

echo "Updating data..."
python scripts/preprocess.py
python scripts/embed.py
python scripts/index.py
echo "Update complete. Run 'streamlit run app/app.py' to test."
