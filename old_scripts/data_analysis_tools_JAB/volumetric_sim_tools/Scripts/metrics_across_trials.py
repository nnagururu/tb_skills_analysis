import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pathlib import Path, PurePath
from volumetric_sim_tools.DataUtils.Recording import Recording
from volumetric_sim_tools.Metrics.PerformanceMetrics import PerformanceMetrics
from volumetric_sim_tools.DataUtils.AnatomicalVolume import AnatomicalVolume
from volumetric_sim_tools.DataUtils.DataMerger import DataMerger
from itertools import combinations

from collections import Counter

def check_path(path: str):
    path = Path(path)
    if not path.exists():
        print(f"Path {path} does not exist")
        exit(0)

    return path
    
class TrialMetrics:
    def __init__(self, parsed_args):
        self.rec_dir = check_path(parsed_args.rec_dir)
        self.vol_dir = check_path(parsed_args.vol_dir)


        self.trial_meta_data = {
        "participant_id": "pilot participant",
        "guidance_modality": "Baseline",
        "anatomy": PurePath(parsed_args.vol_dir).parent.name
        }

        self.trial_paths = [check_path(sub) for sub in self.rec_dir.iterdir() if sub.is_dir()]

    def basic_metrics(self):
        completion_times = []
        collision_dicts = []
        for t in self.trial_paths:
            with Recording(t, **self.trial_meta_data) as recording:
                print(f"Read {len(recording)} h5 files for {recording.participant_id}")
                metrics = PerformanceMetrics(recording, generate_first_vid=False)
                # metrics.metrics_report()
                completion_times.append(metrics.completion_time)
                collision_dicts.append(metrics.collision_dict)
        
        print("Collision Time Mean: {}".format(np.mean(completion_times)))
        print("Collision Time Standard Deviation: {}".format(np.std(completion_times)))
        print(collision_dicts)
    
    def rm_voxel(self, vol, rec_path: Path):
        trial = DataMerger()
        trial.get_merged_data(rec_path) 
        removed_voxels = trial.get_removed_voxels()

        vol.remove_voxels(removed_voxels)

        return vol
    
    def binarize(self, vol, RGBA):
        vol = vol.reshape(-1,4)
        vol_bin = []
        for i in vol:
            if np.array_equal(i, [0., 0., 0., 0.]):
                continue
            if np.array_equal(i, RGBA):
                vol_bin.append(True)
            else:
                vol_bin.append(False)

        return np.asarray(vol_bin)
        
    def dice(self, rec1_path: Path, rec2_path: Path):
        rm_vox_color = [255., 0., 0., 50.]
        
        vol1 = AnatomicalVolume.from_png_list(self.vol_dir)
        vol2 = AnatomicalVolume.from_png_list(self.vol_dir)

        vol1  = self.rm_voxel(vol1, rec1_path).anatomy_matrix
        vol2  = self.rm_voxel(vol2, rec2_path).anatomy_matrix
        
        vol1_b = self.binarize(vol1, rm_vox_color)
        vol2_b = self.binarize(vol2, rm_vox_color)

        intersection = np.logical_and(vol1_b, vol2_b)
        dice = 2. * intersection.sum()/(vol1_b.sum() + vol2_b.sum())

        return dice
        
    def pairwise_dice(self):
        results = []
        pairs = list(combinations(self.trial_paths, 2))
        for pair in pairs:
            print(pair[0])
            print(pair[1])
            results.append(self.dice(check_path(pair[0]), check_path(pair[1])))
        print(results)
        print(np.mean(results))
        print(np.std(results))

    def avg_postop_vol(self, visualize: bool):
        rm_vox_color = [255., 0., 0., 50.]
        
        vol = AnatomicalVolume.from_png_list(self.vol_dir)

        for i in range(len(self.trial_paths)):
            vol_temp = self.rm_voxel(vol, self.trial_paths[i]).anatomy_matrix
            vol_temp_bin = []
            for j in vol_temp:
                if np.array_equal(j, [0., 0., 0., 0.]):
                    continue
                if np.array_equal(j, rm_vox_color):
                    vol_temp_bin.append(np.append(j,1.))       
            if i == 0:
                vol_avg = vol_temp_bin
            else:
                vol_avg += vol_temp_bin
            vol = AnatomicalVolume.from_png_list(self.vol_dir)
        
        vol_avg = np.array(vol_avg)/len(self.trial_paths)
        print(np.shape(vol_avg))

        if visualize:
            vol_avg_lin = vol_avg.reshape(-1,4)
            mask = ~np.all(vol_avg_lin == [0., 0., 0., 0.], axis=1)
            vol_avg_lin_wozero = vol_avg_lin[mask]

            X = vol_avg_lin_wozero[:,0]
            Y = vol_avg_lin_wozero[:,1]
            Z = vol_avg_lin_wozero[:,2]
            prob = vol_avg_lin_wozero[:,3]

            print(np.ptp(X))
            print(np.ptp(Y))
            print(np.ptp(Z))
            print(np.ptp(prob))


            # fig = plt.figure(figsize=(8,6))
            # ax = fig.add_subplot(111,projection='3d')
            # cm = plt.cm.get_cmap('RdYlBu')

            # def randrange(n, vmin, vmax):
            #     return (vmax-vmin)*np.random.rand(n) + vmin
            # n = 100

            # # xs = randrange(n, 23, 32)
            # # ys = randrange(n, 0, 100)
            # # zs = randrange(n, 0, 100)
            # # col = randrange(n, 0, 100)

            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            # ax.set_zlabel('Z Label')
            # sc = ax.scatter(X, Y, Z, c=prob, cmap = cm, marker='o')
            # # sc = ax.scatter(xs, ys, zs, c=col, cmap = cm, marker='o')

            # plt.colorbar(sc)            

            # plt.show()


        return vol_avg 
    

def main():
    parser = ArgumentParser()
    parser.add_argument("-rec_dir", action = "store", dest="rec_dir", help = "Specify directory in which trial recordings are")
    parser.add_argument("-vol_dir", action = "store", dest="vol_dir", help = "Specify directory for volume PNGs")
    parsed_args = parser.parse_args()
    print(parsed_args)

    t_metrics = TrialMetrics(parsed_args)
    # t_metrics.basic_metrics()
    # t_metrics.pairwise_dice()
    t_metrics.avg_postop_vol(True)

if __name__ == "__main__":
    main()


    





