import pandas as pd
from pathlib import Path
from argparse import ArgumentParser
import os
import re



def find_exp_dirs(exp_folder):
    # Defining experiments as those with a .hdf5 file
    exp_dirs = []

    # Traverse the directory tree
    for current_dir, _, _ in os.walk(exp_folder):
        for file in os.listdir(current_dir):
            if file.endswith('.hdf5'):
                exp_dir = os.path.abspath(current_dir)
                exp_dirs.append(current_dir)
                break
    
    return exp_dirs

def exp_dirs_to_csv(exp_dirs):
    anatomy = []
    assist_mode = []
    participant = []
    trial = []
    pupil_data = []

    for exp_dir in exp_dirs:
        exp_dir = Path(exp_dir)
        exp_name = exp_dir.parts[-1]
        parts = exp_name.split('_')

        PT_match = re.match(r'P(\d+)T(\d+)', parts[3])
        anatomy.append(parts[1].split('anat')[-1])
        assist_mode.append(parts[2])
        participant.append(PT_match.group(1))
        trial.append(PT_match.group(2))

        pupil_path0 = exp_dir / '000/eye0.mp4'
        pupil_path1 = exp_dir / '000/eye1.mp4'
        if pupil_path0.exists() and pupil_path1.exists(): 
            pupil_data.append(1)
        else:
            pupil_data.append(0)
        
    df = pd.DataFrame({'exp_dir': exp_dirs, 'anatomy': anatomy, 'assist_mode': assist_mode,
                       'participant': participant, 'trial': trial, 'pupil_data': pupil_data})
    df = df.astype({'participant': 'int32', 'trial': 'int32', 'pupil_data': 'int8'})
    df = df.sort_values(by=['participant', 'trial'])
    df.reset_index(drop=True, inplace=True)

    return df

def write_csv(df, csv_file):
    df.to_csv(csv_file, index=True)


def main():
    parser = ArgumentParser()
    parser.add_argument("--exp_folder", 
                        action="store", 
                        dest="exp_folder", 
                        help="Specify experiments directory", 
                        default = '/Users/nimeshnagururu/Documents/tb_skills_analysis/data/SDF_UserStudy_Data')
    parser.add_argument("--csv_file",
                        action="store",
                        dest="csv_file",
                        help="Specify the csv file to write the data to",
                        default = '/Users/nimeshnagururu/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/exp_dirs.csv')
    args = parser.parse_args()
    
    exp_dirs = find_exp_dirs(args.exp_folder) 
    df = exp_dirs_to_csv(exp_dirs)
    write_csv(df, args.csv_file)


if __name__ == "__main__":
    main()
