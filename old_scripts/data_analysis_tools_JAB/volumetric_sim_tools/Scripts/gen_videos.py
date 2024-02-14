from pathlib import Path
import click
from volumetric_sim_tools.DataUtils.Recording import Recording
from volumetric_sim_tools.DataUtils.VideoUtils import generate_video_from_recording

@click.command()
@click.option("--rec_path", required=True, help="path to recordings directory")

def gen_videos(rec_path: str):
    rec_path = Path(rec_path)
    if rec_path.is_dir():
        subfolders = [sub for sub in rec_path.iterdir() if sub.is_dir()]
        for subfolder in subfolders:
            path = subfolder.resolve()
            try:
                print(f"processing {path}")
                root = Path(path)
                recording = Recording(root, participant_id="None", anatomy="None", guidance_modality="None")
                generate_video_from_recording(recording)
            except Exception as e:
                print(f"error with {path}")
                print(e)
    else:
        print(f"The parent folder '{rec_path}' does not exist or is not a directory.")


def main():
    gen_videos()
    

if __name__ == "__main__":
    main()