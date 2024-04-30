"""
gen_metadata_SDF_stuides.py

Creates metadata files in SDF User Study Data, with similar format to the metadata 
files generated by FIVRS as of the 10/26/23. Note addtiional fields are also provided

Author: Nimesh Nagururu
"""

import pandas as pd
import json
import os

def create_metadata_file(directory, metadata):
    """
    Create a metadata file in JSON format in the specified directory.

    Args:
        directory (str): The directory where the metadata file will be created.
        metadata (dict): Dictionary containing metadata information.

    Returns:
        None
    """
    metadata_filepath = os.path.join(directory, 'metadata.json')
    
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata, f, indent=4)

def main(csv_filepath):
    """
    Main function to create metadata files for SDF User Study Data.

    Args:
        csv_filepath (str): Path to the CSV file containing metadata.

    Returns:
        None
    """

    # Load the CSV containing the metadata about SDF User Study runs
    df = pd.read_csv(csv_filepath)

    for _, row in df.iterrows():
        metadata = {
            "participant_name": str(row['participant']),
            "volume_adf": None,  # Assuming ADF information is not in the CSV
            "volume_name": row['anatomy'],
            "assist_mode": row['assist_mode'],
            "trial": int(row['trial']),
            "pupil_data": bool(row['pupil_data']),
            "notes": None  # Assuming notes are not in the CSV
        }
        
        create_metadata_file(row['exp_dir'], metadata)

csv_filepath = '/Users/nimeshnagururu/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/exp_dirs.csv'

if __name__ == "__main__":
    main(csv_filepath)
