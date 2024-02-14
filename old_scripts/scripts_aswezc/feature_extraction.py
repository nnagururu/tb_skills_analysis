import numpy as np
import math

from scipy.spatial.transform import Rotation as R
from scipy import signal
from scipy import fft
from scipy import integrate


def stats_per_stroke(stroke_arr: np.ndarray):
    '''
    The mean, median, and max of a list of values.

        Parameters:
            stroke_arr (np.ndarray): Values across strokes of a procedure

        Returns:
            mean (np.ndarray): mean of values
            med_ (np.ndarray): median of values
            max_ (np.ndarray): maximum of values
    '''

    if not stroke_arr.any():
        return 0, 0, 0

    mean = np.mean(stroke_arr)
    med_ = np.median(stroke_arr)
    max_ = np.max(stroke_arr)

    return mean, med_, max_


def get_strokes(stream: np.ndarray, timepts: np.ndarray, k=6):
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

    stream = stream[:, :3]
    X_P = []

    # Compute k-cosines for each pivot point
    for j, P in enumerate(stream):

        # Cannot consider edge points as central k's
        if (j - k < 0) or (j + k >= stream.shape[0]):
            continue

        P_a = stream[j - k]
        P_c = stream[j + k]

        k_cos = np.dot(P_a, P_c) / (np.linalg.norm(P_a) * np.linalg.norm(P_c))
        k_cos = max(min(k_cos, 1), -1)
        X_P.append(180 - (math.acos(k_cos) * (180/np.pi)))

    # Detect pivot points
    mu = np.mean(X_P)
    sig = np.std(X_P)

    for i in range(k):

        X_P.insert(0, mu)
        X_P.append(mu)

    X_P = np.array(X_P)

    F_c = [1 if x_P > mu + sig else 0 for x_P in X_P]

    j = 0
    for i in range(1, len(F_c)):

        if F_c[i] == 1 and F_c[i-1] == 0:
            j += 1
        elif sum(F_c[i:i+k]) == 0 and j != 0:
            ind = math.floor(j/2)
            F_c[i-j:i] = [0] * j
            F_c[i-ind] = 1
            j = 0
        elif j != 0:
            j += 1

    st = np.insert(timepts[[s == 1 for s in F_c]], 0, min(timepts))

    return F_c, st


def stroke_force(strokes: np.ndarray, stroke_times: np.ndarray,
                 force_stream: np.ndarray, force_times: np.ndarray):
    '''
    Returns a list of stroke forces representing mean force of
    each stroke of the procedure.

        Parameters:
            strokes (np.ndarray): List of 1's and 0's indicating whether a stroke has ended at the timestamp at its index
            stroke_times (np.ndarray): Time stamps of stroke boundaries
            force_stream (np.ndarray): Force vectors over course of procedure
            force_times (np.ndarray): Time stamps of all force vectors

        Returns:
            forces (np.ndarray): Average stroke forces for each stroke in procedure
    '''

    avg_stroke_force = []
    for i in range(sum(strokes)):

        stroke_mask = [ft >= stroke_times[i] and ft <
                       stroke_times[i+1] for ft in force_times]
        stroke_forces = np.linalg.norm(force_stream[stroke_mask], axis=1)
        avg_stroke_force.append(np.mean(stroke_forces))

    return np.array(avg_stroke_force)


def stroke_length(strokes: np.ndarray, stream: np.ndarray):
    '''
    Returns a list of stroke lengths for each stroke of the procedure.

        Parameters:
            strokes (np.ndarray): List of 1's and 0's indicating whether a stroke has ended at the timestamp at its index
            stream (np.ndarray): Drill poses over course of procedure

        Returns:
            lens (np.ndarray): Length for each stroke in procedure in units of drill position
    '''

    stream = stream[:, :3]

    lens = []
    inds = np.insert(np.where(strokes == 1), 0, 0)
    for i in range(sum(strokes)):
        stroke_len = 0
        curr_stroke = stream[inds[i]:inds[i+1]]
        for j in range(1, len(curr_stroke)):

            stroke_len += np.linalg.norm(curr_stroke[j-1] - curr_stroke[j])

        lens.append(stroke_len)

    return np.array(lens)


def bone_removal_rate(strokes: np.ndarray, stroke_times: np.ndarray,
                      stream: np.ndarray, voxel_times: np.ndarray):
    '''
    Returns a list of bone removal rates representing mean rate of
    each stroke of the procedure.

        Parameters:
            strokes (np.ndarray): List of 1's and 0's indicating whether a stroke has ended at the timestamp at its index
            stroke_times (np.ndarray): Time stamps of stroke boundaries
            stream (np.ndarray): Drill poses over course of procedure
            voxel_times (np.ndarray): Time stamps of all removed voxels

        Returns:
            rate (np.ndarray): Average bone removal rate for each stroke in procedure
    '''

    vox_rm = []
    for i in range(sum(strokes)):

        stroke_voxels = [vt >= stroke_times[i] and vt <
                         stroke_times[i+1] for vt in voxel_times]
        vox_rm.append(sum(stroke_voxels))

    rate = np.divide(np.array(vox_rm), stroke_length(
        np.array(strokes), stream))

    return rate


def procedure_duration(timepts: np.ndarray):
    '''
    Returns a duration of procedure from first to last voxel removal.
    NOTE: Requires that voxels are removed

        Parameters:
            timepts (np.ndarray): Time stamps of all voxel removals

        Returns:
            int: Duration of surgucal procedure given
    '''

    return (max(timepts) - min(timepts))


def drill_orientation(strokes: np.ndarray, stroke_times: np.ndarray,
                      stream: np.ndarray, timepts: np.ndarray,
                      force_stream: np.ndarray, force_times: np.ndarray):
    '''
    Returns a list of drill angles representing mean angle of
    each stroke of the procedure.

        Parameters:
            strokes (np.ndarray): List of 1's and 0's indicating whether a stroke has ended at the timestamp at its index
            stroke_times (np.ndarray): Time stamps of stroke boundaries
            stream (np.ndarray): Drill poses over course of procedure
            timepts (np.ndarray): Time stamps of all drill poses
            force_stream (np.ndarray): Force vectors over course of procedure
            force_times (np.ndarray): Time stamps of all force vectors

        Returns:
            angles (np.ndarray): Average drill angles for each stroke in procedure
    '''

    angles = []
    angle_times = []
    normals = []
    drill_vecs = []

    forces = force_stream[np.linalg.norm(force_stream, axis=1) > 0]
    if len(forces) <= 0:
        return np.array([])
    med = np.median(np.linalg.norm(forces, axis=1))

    for i, t in enumerate(timepts):
        ind = np.argmin(np.abs(force_times - t))
        if np.isclose(np.abs(force_times[ind] - t), 0) and np.linalg.norm(force_stream[ind]) > med:

            angle_times.append(t)

            normal = force_stream[ind]
            normal = np.divide(normal, np.linalg.norm(normal))
            normals.append(normal)

            v = R.from_quat(stream[i, 3:]).apply([-1, 0, 0])
            v = np.divide(v, np.linalg.norm(v))
            drill_vecs.append(v)

    for i, v in enumerate(drill_vecs):

        n = normals[i]

        angle = (np.arccos(np.clip(np.dot(n, v), -1.0, 1.0)) * (180/np.pi))
        if angle > 90:
            angle = 180 - angle

        angles.append(90 - angle)

    avg_stroke_angle = []
    for i in range(sum(strokes)):

        stroke_mask = [at >= stroke_times[i] and at <
                       stroke_times[i+1] for at in angle_times]
        stroke_angles = np.array(angles)[stroke_mask]
        avg_stroke_angle.append(np.mean(stroke_angles)
                                if sum(stroke_mask) > 0 else np.nan)

    A = np.array(avg_stroke_angle)
    avg_stroke_angle = A[~np.isnan(A)]
    return avg_stroke_angle


def get_stroke_indices(stroke_cutoffs):
    '''
    Returns a list of timestamp indices indicating the beginning of a new stroke

        Parameters:
            stroke_cutoffs (list): List of 1's and 0's indicating whether a stroke has ended at the timestamp at its index

        Returns:
            indices (list): List of integer indices naming the indices at which a new stroke is initiated
    '''

    # Find  all indices of stroke_cutoff where the value is a 1
    indices = []
    indices.append(0)
    for i in range(len(stroke_cutoffs)):
        if stroke_cutoffs[i] == 1:
            indices.append(i)

    # 0 index is always the start of a stroke
    return indices


def extract_kinematics(drill_pose, timestamps, stroke_indices):
    '''
    Returns mean, median, and max velocity values across all strokes from drill pose data

    Parameters:
        drill_pose (list): Drill pose data directly extracted from hdf5 file
        timestamps (list): Timestamp data directly extracted from hdf5 file
        stroke_indices (list): List of integer indices naming indices of timestamps at which a new stroke is initiated

    Returns:
        velocities (list): List containing mean, median, and max velocity
        accelerations (list): List containing mean, median, and max acceleration
    '''

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
        next_stroke = len(timestamps)
        if i != len(stroke_indices) - 1:
            next_stroke = stroke_indices[i + 1]
        # Split up positional data for each stroke
        stroke_x = x[stroke_start:next_stroke]
        stroke_y = y[stroke_start:next_stroke]
        stroke_z = z[stroke_start:next_stroke]
        t = timestamps[stroke_start:next_stroke]

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

    # Calculate average acceleration using velocity information
    stroke_accelerations = []
    for i in range(len(stroke_vx)):
        curr_stroke = [[stroke_vx[i][j], stroke_vy[i][j], stroke_vz[i][j]]
                       for j in range(len(stroke_vx[i]))]
        vel = 0
        for k in range(1, len(curr_stroke)):
            vel += np.linalg.norm(np.subtract(
                curr_stroke[k], curr_stroke[k - 1]))
        stroke_accelerations.append(vel / np.ptp(stroke_t[i]))

    return np.array(stroke_velocities), np.array(stroke_accelerations)


def preprocess(drill_pose):
    '''
    Applies second order Butterworth filter and FFT for curve smoothing and pattern repetition isolation

    Parameters:
        drill_pose(list): Drill pose data directly extracted from hdf5 file

    Returns:
        x (list): Preprocessed x position data
        y (list): Preprocessed y position data
        z (list): Preprocessed z position data
    '''
    # Extract x, y, and z data from drill pose data
    x_raw = [i[0] for i in drill_pose]
    y_raw = [i[1] for i in drill_pose]
    z_raw = [i[2] for i in drill_pose]

    # Smooth data using a second order Butterworth filter
    b, a = signal.butter(2, [0.005, 1], 'bandpass', analog=False)
    x_filt = signal.filtfilt(b, a, x_raw)
    y_filt = signal.filtfilt(b, a, y_raw)
    z_filt = signal.filtfilt(b, a, z_raw)

    # Apply a FFT to data
    x = fft(x_filt)
    y = fft(y_filt)
    z = fft(z_filt)

    return x, y, z


def extract_jerk(drill_pose, timestamps, stroke_indices):
    '''
    Returns mean, median, and max jerk across all strokes from drill pose data

    Parameters:
        drill_pose (list): Drill pose data directly extracted from hdf5 file
        timestamps (list): Timestamp data directly extracted from hdf5 file
        stroke_indices (list): List of integer indices naming indices of timestamps at which a new stroke is initiated

    Returns:
        jerks (list): List containing mean, median, and max jerk
    '''

    # Get preprocessed, x, y, and z position data
    x = [i[0] for i in drill_pose]
    y = [i[1] for i in drill_pose]
    z = [i[2] for i in drill_pose]

    stroke_vx = []
    stroke_vy = []
    stroke_vz = []
    stroke_t = []

    # Store velocity information for acceleration
    for i in range(len(stroke_indices)):
        stroke_start = stroke_indices[i]
        next_stroke = len(timestamps)
        if i != len(stroke_indices) - 1:
            next_stroke = stroke_indices[i + 1]
        # Split up positional data for each stroke
        stroke_x = x[stroke_start:next_stroke]
        stroke_y = y[stroke_start:next_stroke]
        stroke_z = z[stroke_start:next_stroke]
        t = timestamps[stroke_start:next_stroke]

        stroke_vx.append(np.gradient(stroke_x, t))
        stroke_vy.append(np.gradient(stroke_y, t))
        stroke_vz.append(np.gradient(stroke_z, t))

        stroke_t.append(t)

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

    return np.array(stroke_jerks)


def extract_curvature(drill_pose, timestamps, stroke_indices):
    '''
    Returns mean, median, and max spaciotemporal curvatures across all strokes from drill pose data

    Parameters:
        drill_pose (list): Drill pose data directly extracted from hdf5 file
        timestamps (list): Timestamp data directly extracted from hdf5 file
        stroke_indices (list): List of integer indices naming indices of timestamps at which a new stroke is initiated

    Returns:
        curvatures (list): List containing mean, median, and max spaciotemporal curvature
    '''

    # Get preprocessed, x, y, and z position data
    x = [i[0] for i in drill_pose]
    y = [i[1] for i in drill_pose]
    z = [i[2] for i in drill_pose]

    curvatures = []
    stroke_curvatures = []
    for i in range(len(stroke_indices)):
        stroke_start = stroke_indices[i]
        next_stroke = len(timestamps)
        if i != len(stroke_indices) - 1:
            next_stroke = stroke_indices[i + 1]

        # Split up positional data for each stroke
        stroke_x = x[stroke_start:next_stroke]
        stroke_y = y[stroke_start:next_stroke]
        stroke_z = z[stroke_start:next_stroke]
        stroke_t = timestamps[stroke_start:next_stroke]

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
