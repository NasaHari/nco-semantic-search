import pandas as pd

def preprocess_and_save(raw_csv_path):
    df = pd.read_csv(raw_csv_path)

    # Combine relevant text fields into one 'text' column
    df['text'] = (
    df['Unit_Title'].fillna('') + '. ' +
    df['Unit_Description'].fillna('')).str.lower().str.replace(r'[^\w\s]', '', regex=True)
    df.to_csv(raw_csv_path, index=False)
    print(f"Preprocessed data saved to {raw_csv_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/processed/nco_cleaned.csv")
    parser.add_argument("--output", type=str, default="data/processed/nco_cleaned.csv")
    args = parser.parse_args()

    preprocess_and_save(args.input)
