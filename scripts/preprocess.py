import pandas as pd
import pdfplumber
import os

def extract_from_pdf(pdf_path: str) -> pd.DataFrame:
    """Extract tables from NCO PDF and structure into DataFrame."""
    with pdfplumber.open(pdf_path) as pdf:
        data = []
        for page in pdf.pages:
            table = page.extract_table()
            if table:
                data.extend(table[1:])  # Skip headers
    df = pd.DataFrame(data, columns=["Code", "Title", "Description", "Hierarchy"])
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean text: lowercase, remove punctuation, combine fields."""
    df["text"] = df["Title"] + " " + df["Description"]
    df["text"] = df["text"].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    return df

if __name__ == "__main__":
    raw_path = "data/raw/NCO-2015.pdf"  # Adjust as needed
    df = extract_from_pdf(raw_path)
    df = clean_data(df)
    df.to_csv("data/processed/nco_data.csv", index=False)
    print("Preprocessing complete.")
