import numpy as np
import pandas as pd
from collections import defaultdict
from exp_reader import ExpReader
from scipy import integrate
from scipy.spatial.transform import Rotation as R
from hdf5_utils import save_dict_to_hdf5, load_dict_from_hdf5
from pathlib import Path
from plotting_skills_analysis import StrokeMetricsVisualizer, plot_3d_vx_rmvd

#TODO figure out why there are more voxels removed after last stroke_endtime

# {short_name: [color, "long_name"]}
anatomy_dict = {
    "Bone": ["255 249 219", "Bone"],
    "Malleus": ["233 0 255", "Malleus"],
    "Incus": ["0 255 149", "Incus"],
    "Stapes": ["63 0 255", "Stapes"],
    "BonyLabyrinth": ["91 123 91", "Bony_Labyrinth"],
    "IAC": ["244 142 52", "IAC"],
    "SuperiorVestNerve": ["255 191 135", "Superior_Vestibular_Nerve"],
    "InferiorVestNerve": ["121 70 24", "Inferior_Vestibular_Nerve"],
    "CochlearNerve": ["219 244 52", "Cochlear_Nerve"],
    "FacialNerve": ["244 214 49", "Facial_Nerve"],
    "Chorda": ["151 131 29", "Chorda_Tympani"],
    "ICA": ["216 100 79", "ICA"],
    "SigSinus": ["110 184 209", "Sinus_+_Dura"],
    "VestAqueduct": ["91 98 123", "Vestibular_Aqueduct"],
    "TMJ": ["100 0 0", "TMJ"],
    "EAC": ["255 225 214", "EAC"],
}

class StrokeExtractor:
    def __init__(self, exp):
        self.exp = exp

        self.data_vrm_mask  = self._filter_pose_ts_within_periods(self.exp.data_ts, self.exp.v_rm_ts)
        self.stroke_ends = self._get_strokes(self.exp.d_poses, self.exp.data_ts, self.data_vrm_mask)
        
    def _find_drilling_seq(self, v_rm_ts, threshold=0.2):
        """
        Find sequences of time points where each consecutive pair is less than `threshold` seconds

        Parameters:
        - time_points: A sorted NumPy array of time points.
        - threshold (seconds): The maximum allowed difference between consecutive time points to consider them a sequence.

        Returns:
        - A list of tuples, where each tuple contains the start and end of a sequence.
        """

        # Calculate differences between consecutive time points
        diffs = np.diff(v_rm_ts)

        # Identify where differences are greater than or equal to the threshold
        breaks = np.where(diffs >= threshold)[0]

        # Calculate start and end indices for sequences
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [len(v_rm_ts) - 1]))

        # Filter out sequences where the start and end are the same (i.e., no actual sequence)
        drill_periods = [(v_rm_ts[start], v_rm_ts[end]) for start, end in zip(starts, ends) if start != end]

        return drill_periods

    def _filter_pose_ts_within_periods(self, data_ts, v_rm_ts):
        """
        Filters and retains time points that fall within any of the specified time periods.

        Parameters:
        - time_points: A numpy array of time points.
        - periods: A list of tuples, where each tuple contains a start and end time point of a period.

        Returns:
        - Boolean mask of driving the time points that fall within any of the specified periods.
        """
        periods = self._find_drilling_seq(v_rm_ts)

        # Initialize an empty array to mark time points within periods
        is_within_period = np.zeros(data_ts.shape, dtype=bool)

        # Iterate through each period and mark time points within the period
        for start, end in periods:
            is_within_period |= (data_ts >= start) & (data_ts <= end)

        return is_within_period

    def _get_strokes(self, d_poses, data_ts, data_vrm_mask, k=3):
        '''
        Returns a list of 1's and 0's indicating whether a stroke has
        ended at the timestamp at its index and the timestamps.

            Parameters:
                stream (np.ndarray): Drill poses over course of procedure
                timepts (np.ndarray): Time stamps of all drill poses

            Returns:
                F_c (np.ndarray): List of 1's and 0's indicating whether a
                                stroke has ended at the timestamp at its index
                st (np.ndarray): Timestamps of stroke ends
        '''
    
        d_pos = d_poses[data_vrm_mask][:,:3]
        data_ts = data_ts[data_vrm_mask]
        K_p = []

        # Compute k-cosines for each pivot point
        for pivot, P in enumerate(d_pos):

            # Cannot consider edge points as central k's
            if (pivot - k < 0) or (pivot + k >= d_pos.shape[0]):
                continue

            # Get vector of previous k points and pivot and after k points and pivot
            v_bf = d_pos[pivot] - d_pos[pivot - k]
            v_af = d_pos[pivot] - d_pos[pivot + k]
            
            cos_theta = np.dot(v_bf, v_af) / (np.linalg.norm(v_bf) * np.linalg.norm(v_af))
            theta = np.arccos(cos_theta) * 180 / np.pi
            K = 180 - theta # now in degrees
            K_p.append(K)

        # Detect pivot points as points with k-cosines greater than mu + sig
        mu = np.mean(K_p)
        sig = np.std(K_p)

        for i in range(k):
            K_p.insert(0, mu)
            K_p.append(mu)

        stroke_ends = [1 if k_P > mu + sig else 0 for k_P in K_p]
        stroke_ends = np.array(stroke_ends)
    
        # Calculate speeds as tiebreak for consecutive pivot points
        position_diffs = np.diff(d_pos, axis=0)
        dists = np.linalg.norm(position_diffs, axis=1)
        time_diffs = np.diff(data_ts)
        speeds = dists/time_diffs
        speeds = np.insert(speeds, 0, 0)   

        pivot_props = np.where(stroke_ends == 1)[0] # indices where stroke_ends == 1

        # Iterate over pivot_props and eliminate consecutive ones
        i = 0
        while i < len(pivot_props) - 1:
            # Check for consecutive indices
            if pivot_props[i] + 1 == pivot_props[i + 1]:
                # Find the start and end of the consecutive ones sequence
                start = i
                while i < len(pivot_props) - 1 and pivot_props[i] + 1 == pivot_props[i + 1]:
                    i += 1
                end = i
                
                # Identify the index with the minimum corresponding speed in the sequence
                min_val_index = np.argmin(speeds[pivot_props[start:end + 1]])
                
                # Set all ones in the sequence to 0 except the minimum speed
                for j in range(start, end + 1):
                    if j != start + min_val_index:
                        stroke_ends[pivot_props[j]] = 0
            i += 1

        stroke_ends[len(stroke_ends)-1] = 1
            
        return stroke_ends

class StrokeMetrics:
    def __init__(self, stroke_extr, num_buckets=10):
        self.exp = stroke_extr.exp
        self.stroke_ends = stroke_extr.stroke_ends
        self.data_vrm_mask = stroke_extr.data_vrm_mask
        self.stroke_endtimes = self.get_stroke_endtimes(self.stroke_ends, self.data_vrm_mask, self.exp.data_ts)
        self.num_buckets = num_buckets  

    def get_stroke_endtimes(self, stroke_ends, data_vrm_mask, data_ts):
        # Adds the first timestamp to the list of stroke end times
        data_ts_vrm = data_ts[data_vrm_mask]
        stroke_endtimes = data_ts_vrm[stroke_ends.astype(bool)]
        stroke_endtimes = np.insert(stroke_endtimes, 0, min(data_ts_vrm))

        return stroke_endtimes       

    def stroke_force(self, forces, forces_ts, stroke_endtimes):
        avg_stroke_force = []
        stroke_forces = 0
        
        forces = forces[:,:3] # excluding torques
        forces_ts = forces_ts[:forces.shape[0]]

        # if forces_ts[0] > 5:
        #     print('Forces_ts recorded properly')
        #     forces_ts = forces_ts[:forces.shape[0]] # subset because of mismatch in shapes
        # else:
        #     print('Forces_ts recorded improperly, correcting...')
        #     forces_ts = forces_ts[:forces.shape[0]] * 1e9 # because of recording error
        
        for i in range(len(stroke_endtimes) - 1):
            stroke_mask = [f_ts >= stroke_endtimes[i] and f_ts < 
                           stroke_endtimes[i+1] for f_ts in forces_ts]
            stroke_forces = np.linalg.norm(forces[stroke_mask], axis=1)
            avg_stroke_force.append(np.mean(stroke_forces))
                
        return np.array(avg_stroke_force)
    
    def stroke_length(self, stroke_ends, d_poses):
        # note that this path length not euclidean legnth
        d_pos = d_poses[:, :3]

        lens = []
        inds = np.insert(np.where(stroke_ends == 1), 0, 0)
        
        for i in range(sum(stroke_ends)):
            stroke_len = 0
            curr_stroke = d_pos[inds[i]:inds[i+1]]
            for j in range(1, len(curr_stroke)):
                stroke_len += np.linalg.norm(curr_stroke[j-1] - curr_stroke[j])

            lens.append(stroke_len)

        return np.array(lens)

    def extract_kinematics(self, d_poses, data_ts, stroke_ends, data_vrm_mask):
        
        drill_pose = d_poses[data_vrm_mask]
        data_ts = data_ts[data_vrm_mask]
        stroke_indices = np.insert(np.where(stroke_ends == 1)[0] + 1, 0, 0)[:-1]
        
        # Extract x, y, and z data from drill pose data
        x = [i[0] for i in drill_pose]
        y = [i[1] for i in drill_pose]
        z = [i[2] for i in drill_pose]

        stroke_vx = []
        stroke_vy = []
        stroke_vz = []
        stroke_t = []
        stroke_velocities = []

        # Store velocity information for acceleration and calculate average velocity
        for i in range(len(stroke_indices)):
            stroke_start = stroke_indices[i]
            next_stroke = len(data_ts)
            if i != len(stroke_indices) - 1:
                next_stroke = stroke_indices[i + 1]
            # Split up positional data for each stroke
            stroke_x = x[stroke_start:next_stroke]
            stroke_y = y[stroke_start:next_stroke]
            stroke_z = z[stroke_start:next_stroke]
            t = data_ts[stroke_start:next_stroke]

            stroke_vx.append(np.gradient(stroke_x, t))
            stroke_vy.append(np.gradient(stroke_y, t))
            stroke_vz.append(np.gradient(stroke_z, t))
            stroke_t.append(t)

            # Calculate distance traveled during stroke and use to calculate velocity
            curr_stroke = [[stroke_x[k], stroke_y[k], stroke_z[k]]
                        for k in range(len(stroke_x))]
            dist = 0
            for l in range(1, len(curr_stroke)):
                dist += np.linalg.norm(np.subtract(
                    curr_stroke[l], curr_stroke[l - 1]))
            stroke_velocities.append(dist / np.ptp(t))

        # Calculate average acceleration using velocity information, and store for jerk
        stroke_accelerations = []
        for i in range(len(stroke_vx)):
            curr_stroke = [[stroke_vx[i][j], stroke_vy[i][j], stroke_vz[i][j]]
                        for j in range(len(stroke_vx[i]))]
            vel = 0
            for k in range(1, len(curr_stroke)):
                vel += np.linalg.norm(np.subtract(
                    curr_stroke[k], curr_stroke[k - 1]))
            stroke_accelerations.append(vel / np.ptp(stroke_t[i]))

        stroke_ax = []
        stroke_ay = []
        stroke_az = []

        # Store acceleration information for jerk
        for i in range(len(stroke_vx)):
            stroke_ax.append(np.gradient(stroke_vx[i], stroke_t[i]))
            stroke_ay.append(np.gradient(stroke_vy[i], stroke_t[i]))
            stroke_az.append(np.gradient(stroke_vz[i], stroke_t[i]))

        # Calculate average jerk using acceleration information
        stroke_jerks = []
        for i in range(len(stroke_ax)):
            curr_stroke = [[stroke_ax[i][j], stroke_ay[i][j], stroke_az[i][j]]
                        for j in range(len(stroke_ax[i]))]
            acc = 0
            for k in range(1, len(curr_stroke)):
                acc += np.linalg.norm(np.subtract(
                    curr_stroke[k], curr_stroke[k - 1]))
            stroke_jerks.append(acc / np.ptp(stroke_t[i]))
        

        return np.array(stroke_velocities), np.array(stroke_accelerations), np.array(stroke_jerks)

    def extract_curvature(self, d_poses, data_ts, stroke_ends, data_vrm_mask):
        
        drill_pose = d_poses[data_vrm_mask]
        data_ts = data_ts[data_vrm_mask]
        stroke_indices = np.insert(np.where(stroke_ends == 1)[0] + 1, 0, 0)[:-1]


        # Get preprocessed, x, y, and z position data
        x = [i[0] for i in drill_pose]
        y = [i[1] for i in drill_pose]
        z = [i[2] for i in drill_pose]

        curvatures = []
        stroke_curvatures = []
        for i in range(len(stroke_indices)):
            stroke_start = stroke_indices[i]
            next_stroke = len(data_ts)
            if i != len(stroke_indices) - 1:
                next_stroke = stroke_indices[i + 1]

            # Split up positional data for each stroke
            stroke_x = x[stroke_start:next_stroke]
            stroke_y = y[stroke_start:next_stroke]
            stroke_z = z[stroke_start:next_stroke]
            stroke_t = data_ts[stroke_start:next_stroke]

            # Calculate velocity and acceleration for each stroke
            stroke_vx = np.gradient(stroke_x, stroke_t)
            stroke_vy = np.gradient(stroke_y, stroke_t)
            stroke_vz = np.gradient(stroke_z, stroke_t)
            stroke_ax = np.gradient(stroke_vx, stroke_t)
            stroke_ay = np.gradient(stroke_vy, stroke_t)
            stroke_az = np.gradient(stroke_vz, stroke_t)
            curvature = []
            stroke_t_copy = [t for t in stroke_t]

            for j in range(len(stroke_vx)):
                # Calculate r' and r'' at specific time point
                r_prime = [stroke_vx[j], stroke_vy[j], stroke_vz[j]]
                r_dprime = [stroke_ax[j], stroke_ay[j], stroke_az[j]]

                # Potentially remove
                if np.linalg.norm(r_prime) == 0:
                    stroke_t_copy.pop(j)
                    continue
                k = np.linalg.norm(np.cross(r_prime, r_dprime)) / \
                    ((np.linalg.norm(r_prime)) ** 3)
                curvature.append(k)

            # Average value of function over an interval is integral of function divided by length of interval
            stroke_curvatures.append(integrate.simpson(
                curvature, stroke_t_copy) / np.ptp(stroke_t_copy))

        return np.array(stroke_curvatures)

    def contact_orientation(self, stroke_endtimes, d_poses, data_ts,
                        data_vrm_mask,forces, forces_ts):

        d_poses = d_poses[data_vrm_mask]
        data_ts = data_ts[data_vrm_mask]
        forces = forces[:,:3]
        forces_ts = forces_ts[:forces.shape[0]]
                              
        avg_angles_per_stroke = []

        for i in range(len(stroke_endtimes) - 1):
            stroke_mask = (forces_ts >= stroke_endtimes[i]) & (forces_ts < stroke_endtimes[i + 1])
            stroke_forces = forces[stroke_mask]
            stroke_forces_ts = forces_ts[stroke_mask]

            if stroke_forces.size == 0:
                avg_angles_per_stroke.append(np.nan)  # Append NaN for strokes with no data, unlikely tho
                continue

            closest_indices = np.abs(np.subtract.outer(data_ts, stroke_forces_ts)).argmin(axis=0)
            closest_d_poses = d_poses[closest_indices]

            stroke_angles = []

            for force, pose in zip(stroke_forces, closest_d_poses):
                drill_vec =  R.from_quat(pose[3:]).apply([-1, 0, 0]) # assuming the drill is pointing in the x direction
                drill_vec = drill_vec / np.linalg.norm(drill_vec)
                
                force_norm = np.linalg.norm(force)
                if force_norm <= 0:
                    continue
                normal = force / force_norm
                
                angle = np.arccos(np.clip(np.dot(normal, drill_vec), -1.0, 1.0)) * (180 / np.pi)
                if angle > 90:
                    angle = 180 - angle

                stroke_angles.append(90 - angle)
            
            avg = np.mean(stroke_angles)
            avg_angles_per_stroke.append(avg)
        
        return np.array(avg_angles_per_stroke)

    def orientation_wrt_camera(self, stroke_ends, stroke_endtimes, d_poses, cam_poses, data_ts, data_vrm_mask):
        d_poses = d_poses[data_vrm_mask]
        data_ts = data_ts[data_vrm_mask]
        cam_poses = cam_poses[data_vrm_mask]

        avg_angles = []

        for i in range(len(stroke_endtimes) - 1):  # Ensuring stroke_ends[i+1] is valid
            stroke_mask = (data_ts >= stroke_endtimes[i]) & (data_ts < stroke_endtimes[i+1])
            stroke_poses = d_poses[stroke_mask]
            stroke_cam_poses = cam_poses[stroke_mask]

            if len(stroke_poses) == 0:
                avg_angles.append(np.nan)
                continue

            # Calculate orientations
            drill_quats = stroke_poses[:, 3:]
            cam_quats = stroke_cam_poses[:, 3:]

            drill_rot = R.from_quat(drill_quats).apply([-1, 0, 0])
            cam_rot = R.from_quat(cam_quats).apply([-1, 0, 0])

            # Normalize
            drill_rot = drill_rot / np.linalg.norm(drill_rot, axis=1)[:, np.newaxis]
            cam_rot = cam_rot / np.linalg.norm(cam_rot, axis=1)[:, np.newaxis]

            # Calculate angles between vectors
            angles = np.arccos(np.clip(np.einsum('ij,ij->i', drill_rot, cam_rot), -1.0, 1.0)) * (180 / np.pi)
            angles = np.where(angles > 90, 180 - angles, angles)
            avg_angle = np.mean(90 - angles)

            avg_angles.append(avg_angle)

        avg_angle_wrt_camera = np.array(avg_angles)

        return avg_angle_wrt_camera

    def voxels_removed(self, stroke_endtimes, v_rm_ts, v_rm_locs):
        #TODO
        cum_vxl_rm_stroke_end = self.gen_cum_vxl_rm_stroke_end(stroke_endtimes, v_rm_ts, v_rm_locs)
        vxls_removed = np.diff(cum_vxl_rm_stroke_end, prepend=0)

        return vxls_removed

    def gen_cum_vxl_rm_stroke_end(self, stroke_endtimes, v_rm_ts, v_rm_locs):
        stroke_endtimes = stroke_endtimes[1:] # remove first timestamp which is min(v_rm_ts) and not needed
        closest_vxl_rm_ts_to_stroke_end = np.searchsorted(v_rm_ts, stroke_endtimes)
        cum_vxl_rm_stroke_end = np.searchsorted(v_rm_locs[:,0], closest_vxl_rm_ts_to_stroke_end, side='right') -1 
        
        return np.array(cum_vxl_rm_stroke_end)
    
    def calc_metrics (self):
        length = self.stroke_length(self.stroke_ends, self.exp.d_poses)
        velocity, acceleration, jerk = self.extract_kinematics(self.exp.d_poses, self.exp.data_ts, self.stroke_ends, self.data_vrm_mask)
        curvature = self.extract_curvature(self.exp.d_poses, self.exp.data_ts, self.stroke_ends, self.data_vrm_mask)
        orientation_wrt_camera = self.orientation_wrt_camera(self.stroke_ends, self.stroke_endtimes, self.exp.d_poses, self.exp.cam_poses, self.exp.data_ts, self.data_vrm_mask)
        voxels_removed = self.voxels_removed(self.stroke_endtimes, self.exp.v_rm_ts, self.exp.v_rm_locs)

        force = self.stroke_force(self.exp.forces, self.exp.forces_ts, self.stroke_endtimes)
        contact_angle = self.contact_orientation(self.stroke_endtimes, self.exp.d_poses, self.exp.data_ts, self.data_vrm_mask, self.exp.forces, self.exp.forces_ts)
        
        if not sum(np.isnan(force)) > 0.5*len(force):
            metrics_dict = {'length': length, 'velocity': velocity, 'acceleration': acceleration, 'jerk': jerk, 'vxls_removed': voxels_removed, 
                            'curvature': curvature, 'force': force, 'angle_wrt_bone': contact_angle, 'angle_wrt_camera': orientation_wrt_camera}
        else:
            metrics_dict = {'length': length, 'velocity': velocity, 'acceleration': acceleration, 'jerk': jerk, 'vxls_removed': voxels_removed,
                            'curvature': curvature, 'angle_wrt_camera': orientation_wrt_camera}
       
        return metrics_dict

    def assign_strokes_to_voxel_buckets(self):
        num_buckets = self.num_buckets

        # Generate cumulative voxels removed at stroke ends
        cum_vxl_rm_stroke_end = self.gen_cum_vxl_rm_stroke_end(self.stroke_endtimes, self.exp.v_rm_ts, self.exp.v_rm_locs)
        total_voxels = cum_vxl_rm_stroke_end[-1]
        
        # Determine the range of each bucket
        bucket_size = total_voxels / num_buckets
        bucket_ranges = [(int(i * bucket_size), int((i + 1) * bucket_size - 1)) for i in range(num_buckets)]
        
        # Assign each stroke to a bucket
        bucket_assignments = np.zeros(len(cum_vxl_rm_stroke_end), dtype=int)
        
        for i, voxel_count in enumerate(cum_vxl_rm_stroke_end):
            # Find the bucket index; max is to handle the last bucket edge case
            bucket_index = min(int(voxel_count / bucket_size), num_buckets - 1)
            bucket_assignments[i] = bucket_index

        bucket_dict = {'bucket_assignments': bucket_assignments, 'bucket_ranges': bucket_ranges}  
        
        return bucket_dict
    
    def save_stroke_metrics_and_buckets(output_path, self):
        metrics = self.calc_metrics()
        bucket_dict = self.assign_strokes_to_voxel_buckets()

        save_dict_to_hdf5(metrics, output_path / 'stroke_metrics.hdf5')
        save_dict_to_hdf5(bucket_dict, output_path / 'stroke_buckets.hdf5')

        return metrics, bucket_dict

class GenMetrics:
    def __init__(self, stroke_extr, exp_dir):
        # Not sure if this lazy intiializaiton with a stroke_extr object is good practice
        self.exp = stroke_extr.exp
        self.stroke_ends = stroke_extr.stroke_ends
        self.data_vrm_mask = stroke_extr.data_vrm_mask
        self.exp_dir = exp_dir
    
    def procedure_time(self):
        # Copy of method from StrokeMetrics, should replace redundnacy
        data_ts_vrm = self.exp.data_ts[self.data_vrm_mask]
        stroke_endtimes = data_ts_vrm[self.stroke_ends.astype(bool)]
        stroke_endtimes = np.insert(stroke_endtimes, 0, min(data_ts_vrm))
        self.stroke_endtimes = stroke_endtimes

        return stroke_endtimes[-1] - stroke_endtimes[0]
    
    def num_strokes(self):
        # Just a count of strokes removed
        return sum(self.stroke_ends)
    
    def metadata_dict(self):
        # metadata dictionary has participant_name, volume_name, assist_mode, and trial_number
        with open(self.exp_dir + '/metadata.json', 'r') as f:
            metadata = f.read()

        return metadata
    
    def voxel_rmvd_dict(self):
        # Returns a dictionary with the number of voxels removed (value) for each anatomy (key)
        vxl_rmvd_dict = defaultdict(int)
        v_rm_colors = np.array(self.exp.v_rm_colors).astype(np.int32)
        v_rm_colors_df = pd.DataFrame(v_rm_colors, columns=["ts_idx", "r", "g", "b", "a"])

        # add a column with the anatomy names
        for name, anatomy_info_list in anatomy_dict.items():
            color, full_name = anatomy_info_list
            color = list(map(int, color.split(" ")))
            v_rm_colors_df.loc[
                (v_rm_colors_df["r"] == color[0])
                & (v_rm_colors_df["g"] == color[1])
                & (v_rm_colors_df["b"] == color[2]),
                "anatomy_name",
            ] = name
        
        # Count number of removed voxels of each anatomy
        voxel_summary = v_rm_colors_df.groupby(["anatomy_name"]).count()

        for anatomy in voxel_summary.index:
            vxl_rmvd_dict[anatomy] += voxel_summary.loc[anatomy, "a"]
        
        return dict(vxl_rmvd_dict)
    
    def burr_change_dict(self, threshold = 0.8):
        # returns dictionary with burr size, and percent time spent in each burr size
        #TODO change to percent voxel removed in each burr size
        if self.exp.burr_chg_sz is None:
            burr_chg_dict = {'6 mm': 1.0}
            return burr_chg_dict

        burr_chg_sz = np.array(self.exp.burr_chg_sz)
        burr_chg_ts = np.array(self.exp.burr_chg_ts)

        burr_chg_sz = np.insert(burr_chg_sz, 0, 6) # 6 is starting burr size
        burr_chg_ts = np.append(burr_chg_ts, self.stroke_endtimes[-1]) # changing time stamps to represent the time at which the burr size changes from corresponding size (not to) 

    
        # calculate differences between consecutive changes
        diffs = np.diff(burr_chg_ts)
        diffs = np.append(diffs, True) # keep last change

        # select elements where the difference is >= 0.8s
        burr_chg_sz = burr_chg_sz[diffs >= threshold]
        burr_chg_ts = burr_chg_ts[diffs >= threshold]

        burr_sz_duration = np.diff(burr_chg_ts, prepend=self.stroke_endtimes[0])
        relative_burr_duration = burr_sz_duration / self.procedure_time()


        burr_chg_dict = {str(burr_size) + ' mm': 0 for burr_size in np.unique(burr_chg_sz)}
        for i in range(len(burr_chg_ts)):
            burr_size_str = str(burr_chg_sz[i]) + ' mm'
            burr_chg_dict[burr_size_str] += relative_burr_duration[i]
        
        return dict(burr_chg_dict)
    
    def calc_metrics(self):
        procedure_time = self.procedure_time()
        num_strokes = self.num_strokes()
        metadata = self.metadata_dict()
        vxl_rmvd_dict = self.voxel_rmvd_dict()
        burr_chg_dict = self.burr_change_dict()

        self.gen_metrics_dict = {'procedure_time': procedure_time, 'num_strokes': num_strokes, 'metadata': metadata, 'vxl_rmvd_dict': vxl_rmvd_dict, 'burr_chg_dict': burr_chg_dict}

        return self.gen_metrics_dict
    
    def save_gen_metrics(self):
        save_dict_to_hdf5(self.gen_metrics_dict, self.exp_dir + '/gen_metrics.hdf5')
        


def main():
    exp_csv = pd.read_csv('/Users/nimeshnagururu/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/exp_dirs_DONOTOVERWRITE.csv')
    exps = exp_csv['exp_dir']

    # for i in range(len(exps)):
    novice_exp = ExpReader('/Users/nimeshnagururu/Documents/tb_skills_analysis/data/SDF_UserStudy_Data/Participant_8/2023-02-08 10:45:02_anatM_baseline_P8T5', verbose = True)
    # novice_exp = ExpReader(exps[49], verbose = True)
    
    novice_stroke_extr = StrokeExtractor(novice_exp)
    novice_stroke_metr = StrokeMetrics(novice_stroke_extr)

    # novice_stroke_metr.gen_cum_vxl_rm_stroke_end(novice_stroke_metr.stroke_endtimes, novice_stroke_metr.exp.v_rm_ts, novice_stroke_metr.exp.v_rm_locs)
    novice_metrics_dict = novice_stroke_metr.calc_metrics()
    novice_bucket_dict = novice_stroke_metr.assign_strokes_to_voxel_buckets()

    
    visualizer = StrokeMetricsVisualizer(novice_metrics_dict, novice_bucket_dict, novice_metrics_dict, novice_bucket_dict, plot_previous_bucket=True)
    visualizer.interactive_plot_buckets() 

    # novice_gen_metr = GenMetrics(novice_stroke_extr, exps[46])
    # plot_3d_vx_rmvd(novice_exp)
    

    # att_exp = exp_reader(exps[46], verbose = True)
    # att_stroke_extr = stroke_extractor(att_exp)
    # att_stroke_metr = stroke_metrics(att_exp, att_stroke_extr.stroke_ends, att_stroke_extr.data_vrm_mask)
    # att_metrics_dict = att_stroke_metr.calc_metrics()

    # stroke_metr.plot_metrics(metrics_dict)
    # novice_stroke_metr.interactive_plot(window_percentage=30, overlap_percentage=80)
        
    # break
        

if __name__ == "__main__":
    main()