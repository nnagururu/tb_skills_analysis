import pandas as pd
import json
import os

def create_metadata_file(directory, metadata):
    # Create the full filepath for the new metadata.json file
    metadata_filepath = os.path.join(directory, 'metadata.json')
    
    # Write the metadata to the new file
    with open(metadata_filepath, 'w') as f:
        json.dump(metadata, f, indent=4)

def main(csv_filepath):
    # Load the CSV data into a DataFrame
    df = pd.read_csv(csv_filepath)

    # Iterate over the rows of the DataFrame
    for _, row in df.iterrows():
        # Construct the metadata dictionary
        metadata = {
            "participant_name": str(row['participant']),
            "volume_adf": None,  # Assuming ADF information is not in the CSV
            "volume_name": row['anatomy'],
            "assist_mode": row['assist_mode'],
            "trial": int(row['trial']),
            "pupil_data": bool(row['pupil_data']),
            "notes": None  # Assuming notes are not in the CSV
        }
        
        # Create the metadata.json file in the corresponding directory
        create_metadata_file(row['exp_dir'], metadata)

# Path to the CSV file
csv_filepath = '/Users/nimeshnagururu/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/exp_dirs.csv'

# Execute the main function
if __name__ == "__main__":
    main(csv_filepath)
