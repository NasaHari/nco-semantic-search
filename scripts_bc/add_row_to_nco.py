import pandas as pd
import os

def add_row_to_nco_cleaned(
        csv_path: str,
        new_row: dict,
        save_path: str = None
    ):
    """
    Load existing NCO cleaned CSV, append a new row, and save back.
    
    Args:
        csv_path (str): Path to existing CSV file.
        new_row (dict): Dictionary with keys matching CSV columns.
        save_path (str): Where to save updated CSV. If None, overwrite csv_path.
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file does not exist at {csv_path}")

    # Load existing dataset
    df = pd.read_csv(csv_path)

    # Convert the new_row dict keys and columns to same case/order if needed
    # Assuming keys in new_row exactly match dataset columns.

    # Append new row
    df = df.append(new_row, ignore_index=True)

    # Save updated dataset
    save_path = save_path if save_path else csv_path
    df.to_csv(save_path, index=False)
    print(f"New row appended and saved to {save_path}")

if __name__ == "__main__":
    # Example new row data (replace the values with actual data)
    new_row_data = {
        "Division": "1",
        "Division_Description": "Managers plan, direct, coordinate and evaluate...",
        "Sub_Division": "11",
        "Sub_Division_Description": "Production and Specialized Services Managers plan...",
        "Group": "111",
        "Group_Description": "Managers who manage production resources...",
        "Family": "1111",
        "Family_Description": "General Managers manage business or operations...",
        "Unit_Code": "1111.0100",
        "Unit_Title": "General Manager",
        "Unit_Description": "General Managers plan, direct, coordinate, and evaluate operational activities...",
        "NCO_2004": "1234.56",
        "QP_NOS Reference": "MGR/Q0001",
        "QP_NOS Name": "General Manager QPs",
        "NSQF_Level": 8
    }

    csv_file = "data/processed/nco_cleaned.csv"

    add_row_to_nco_cleaned(csv_file, new_row_data)
