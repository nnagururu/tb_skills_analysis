from argparse import ArgumentParser
from pathlib import Path
from collections import OrderedDict
from natsort import natsorted
import numpy as np
from exp_reader import exp_reader 
import cv2

def main():
    parser = ArgumentParser()
    parser.add_argument("--exp_dir", 
                            action="store", 
                            dest="exp_dir", 
                            help="Specify experiments directory", 
                            default = '/Users/nimeshnagururu/Documents/tb_skills_analysis/SDF_UserStudy_Data/Participant_8/2023-02-13 09:39:29_anatT_haptic_P8T9')
    
    args = parser.parse_args()
    exp_dir = Path(args.exp_dir)

    output_vid_f = exp_dir / ('000/' + 'world.mp4')
    output_timestamps_f = exp_dir / ('000/' +'world_timestamps.npy')
    
    reader = exp_reader(exp_dir)
    od = reader._data

    world_timestamps = od['data']['time']
    l_imgs = od['data']['l_img']

    frate = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(str(output_vid_f), fourcc, frate, (l_imgs[0].shape[1], l_imgs[0].shape[0]))

    # time_diffs = np.diff(world_timestamps, prepend=world_timestamps[0])
    time_diffs = np.diff(world_timestamps)
    # print(time_diffs[:10])
    frames_per_timestamp = (time_diffs * frate).round().astype(int)  # Calculate number of frames to replicate
    # print(frames_per_timestamp[:10])

    upsampled_images = []
    upsampled_timestamps = []

    current_timestamp = world_timestamps[0]
    for i in range(len(time_diffs)):
        for _ in range(frames_per_timestamp[i]):
            upsampled_images.append(l_imgs[i])
        interval_timestamps = np.linspace(current_timestamp, 
                                        current_timestamp + time_diffs[i], 
                                        frames_per_timestamp[i], 
                                        endpoint=False)
        upsampled_timestamps.extend(interval_timestamps)
        current_timestamp += time_diffs[i]

    # print(len(upsampled_images))
    # print(len(upsampled_timestamps))
    # print(upsampled_timestamps[:10])
    # print(max(upsampled_timestamps) - min(upsampled_timestamps))
    # print(max(world_timestamps) - min(world_timestamps))

    for img in upsampled_images:
        video.write(img)

    video.release()

    np.save(output_timestamps_f, np.array(upsampled_timestamps))

    # for img in l_imgs:
    #     video.write(img)
    # video.release()
    # np.save('world_timestamps.npy', np.array(world_timestamps))



if __name__ == "__main__":
	main()