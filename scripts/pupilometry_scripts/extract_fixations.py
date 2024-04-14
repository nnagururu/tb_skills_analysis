# https://gist.github.com/papr/f7d5fbf7809578e26e1cbb77fdb9ad06

import argparse
import logging
import os
from collections import deque
from typing import NamedTuple

import msgpack
import numpy as np
import pandas as pd
from rich.progress import Progress, TextColumn
from rich.logging import RichHandler
from scipy.spatial.distance import pdist

logger = logging.getLogger(__name__)

# Data classes


class GazeDirection(NamedTuple):
    """Algorithm input"""

    timestamp: float
    x: float
    y: float
    z: float


class Fixation(NamedTuple):
    """Algorithm output"""

    id: int
    start_timestamp: float
    end_timestamp: float
    duration_ms: float
    dispersion_deg: float
    norm_pos_avg_x: float
    norm_pos_avg_y: float
    direction_avg_x: float
    direction_avg_y: float
    direction_avg_z: float


# Main script functionality


def main(recordings, csv_out_prefix, overwrite=False, **algorithm_kwargs):
    """Process given recordings one by one

    Iterates over each recording and handles cases where no gaze.pldata or
    gaze_timestamps.npy files could be found.

    recordings: List of recording folders
    csv_out: CSV file name under which the result will be saved
    """
    for rec in recordings:
        try:
            logger.info(f"Extracting {rec}...")
            process_recording(
                rec, csv_out_prefix, overwrite=overwrite, **algorithm_kwargs
            )
        except FileNotFoundError:
            logger.exception(
                f"The recording {rec} did not include any prerecorded gaze files!"
            )


def process_recording(
    recording, csv_out_prefix, overwrite=False, min_confidence=0.6, **algorithm_kwargs
):
    """Process a single recording

    recordings: List of recording folders
    csv_out: CSV file name prefix under which the result will be saved
    overwrite: Boolean indicating if an existing csv file should be overwritten
    """
    csv_out_path = os.path.join(recording, csv_out_prefix + ".csv")
    if os.path.exists(csv_out_path):
        if not overwrite:
            logger.warning(f"{csv_out_path} exists already! Not overwriting.")
            return
        else:
            logger.warning(f"{csv_out_path} exists already! Overwriting.")

    gaze_data = pd.DataFrame(load_and_yield_gaze_directions(recording, min_confidence))
    intrinsics = read_intrinsics(recording)
    gaze = run_fixation_detection(gaze_data, intrinsics=intrinsics, **algorithm_kwargs)
    logger.info(f"Writing fixation results to {csv_out_path}")
    gaze.to_csv(csv_out_path, index=False)


# Fixation detection algorithm


def run_fixation_detection(
    gaze_data: pd.DataFrame,
    max_dispersion_deg: float,
    min_duration_ms: int,
    max_duration_ms: int,
    intrinsics,
):
    if gaze_data.empty:
        logger.warning("No gaze data found! Skipping...")
        return

    min_duration_s = min_duration_ms / 1_000
    max_duration_s = max_duration_ms / 1_000

    gaze_data.sort_values(by="timestamp", ignore_index=True, inplace=True)

    fixation_result = Fixation_Result_Factory(intrinsics)

    working_slice = deque()

    _, rows = zip(*gaze_data.iterrows())
    remaining_slice = deque(rows)
    fixations = []
    total_gaze = len(rows)

    with Progress(
        *Progress.get_default_columns(),
        TextColumn("Fixations found: {task.fields[num_found]}"),
        expand=True,
        transient=True,
    ) as progress:
        task_processed = progress.add_task(
            "Processing gaze directions...", total=total_gaze, num_found=0
        )
        while remaining_slice:
            progress.update(
                task_processed,
                completed=total_gaze - len(remaining_slice),
                num_found=len(fixations),
            )

            # check if working_queue contains enough data
            if (
                len(working_slice) < 2
                or (working_slice[-1]["timestamp"] - working_slice[0]["timestamp"])
                < min_duration_s
            ):
                datum = remaining_slice.popleft()
                working_slice.append(datum)
                continue

            # min duration reached, check for fixation
            dispersion = vector_dispersion(working_slice)
            if dispersion > max_dispersion_deg:
                # not a fixation, move forward
                working_slice.popleft()
                continue

            left_idx = len(working_slice)

            # minimal fixation found. collect maximal data
            # to perform binary search for fixation end
            while remaining_slice:
                datum = remaining_slice[0]
                if datum["timestamp"] > working_slice[0]["timestamp"] + max_duration_s:
                    break  # maximum data found
                working_slice.append(remaining_slice.popleft())

            # check for fixation with maximum duration
            dispersion = vector_dispersion(working_slice)
            if dispersion <= max_dispersion_deg:
                fixation = fixation_result.from_data(dispersion, working_slice)
                fixations.append(fixation)
                working_slice.clear()  # discard old Q
                continue

            slicable = list(working_slice)  # deque does not support slicing
            right_idx = len(working_slice)

            # binary search
            while left_idx < right_idx - 1:
                middle_idx = (left_idx + right_idx) // 2
                dispersion = vector_dispersion(slicable[: middle_idx + 1])
                if dispersion <= max_dispersion_deg:
                    left_idx = middle_idx
                else:
                    right_idx = middle_idx

            # left_idx-1 is last valid base datum
            final_base_data = slicable[:left_idx]
            to_be_placed_back = slicable[left_idx:]
            dispersion_result = vector_dispersion(final_base_data)

            fixation = fixation_result.from_data(dispersion_result, final_base_data)
            fixations.append(fixation)
            working_slice.clear()  # clear queue
            remaining_slice.extendleft(reversed(to_be_placed_back))

        # final progress bar update
        progress.update(task_processed, completed=total_gaze, num_found=len(fixations))
    logger.info(f"{len(fixations)} fixations found")
    return pd.DataFrame(fixations)


def vector_dispersion(data):
    vectors = pd.DataFrame(data)[list("xyz")]
    distances = pdist(vectors, metric="cosine")
    dispersion = np.arccos(1.0 - distances.max())
    return np.rad2deg(dispersion)


class Fixation_Result_Factory(object):
    __slots__ = (
        "intrinsics",
        "_id_counter",
    )

    def __init__(self, intrinsics):
        self._id_counter = 0
        self.intrinsics = intrinsics

    def from_data(self, dispersion_result: float, data) -> Fixation:
        df = pd.DataFrame(data)
        start, stop = df.timestamp.iloc[[0, -1]]
        duration_ms = (stop - start) * 1_000
        direction_3d = df[list("xyz")].mean(axis=0)
        norm_2d = direction3d_to_normalized2d(direction_3d, self.intrinsics)
        fixation = Fixation(
            id=self._id_counter,
            start_timestamp=start,
            end_timestamp=stop,
            duration_ms=duration_ms,
            dispersion_deg=dispersion_result,
            norm_pos_avg_x=norm_2d[0],
            norm_pos_avg_y=norm_2d[1],
            direction_avg_x=direction_3d[0],
            direction_avg_y=direction_3d[1],
            direction_avg_z=direction_3d[2],
        )
        self._id_counter += 1
        return fixation


# Load data from recording


def load_and_yield_gaze_directions(directory, min_confidence, topic="gaze"):
    """Load and extract gaze direction data

    See the data format documentation[2] for details on the data structure.

    Adapted open-source code from Pupil Player[1] to read pldata files.
    Removed the usage of Serialized_Dicts since this script has the sole purpose
    of running through the data once.

    [1] https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/file_methods.py#L137-L153
    [2] https://docs.pupil-labs.com/developer/core/recording-format/
    """
    intrinsics = read_intrinsics(directory)
    ts_file = os.path.join(directory, topic + "_timestamps.npy")
    data_ts = np.load(ts_file)

    msgpack_file = os.path.join(directory, topic + ".pldata")
    with open(msgpack_file, "rb") as fh:
        unpacker = msgpack.Unpacker(fh, raw=False, use_list=False)
        with Progress(transient=True) as progress:
            for timestamp, (topic, payload) in progress.track(
                zip(data_ts, unpacker),
                description="Loading gaze data",
                total=len(data_ts),
            ):
                datum = deserialize_msgpack(payload)

                # custom extraction function for pupil data, see below for details
                direction, conf = extract_gaze_direction(datum, intrinsics)
                if conf < min_confidence:
                    continue

                # yield data according to csv_header() sequence
                yield GazeDirection(timestamp, *direction)


def deserialize_msgpack(msgpack_bytes):
    """Deserialize msgpack[1] data

    [1] https://msgpack.org/index.html
    """
    return msgpack.unpackb(msgpack_bytes, strict_map_key=False)


def extract_gaze_direction(gaze_datum, intrinsics):
    """Extract data for a given pupil datum

    Returns: tuple(*gaze_point_3d, confidence)
    """
    if "gaze_point_3d" in gaze_datum:
        direction = gaze_datum["gaze_point_3d"]
    else:
        direction = normalized2d_to_direction3d(gaze_datum["norm_pos"], intrinsics)
    return (direction, gaze_datum["confidence"])


# Intrinsics: Point un/projection and un/distortion


def read_intrinsics(directory):
    intrinsics_location = os.path.join(directory, "world.intrinsics")
    with open(intrinsics_location, "rb") as fh:
        intrinsics = msgpack.unpack(fh)
    intrinsics = next(values for key, values in intrinsics.items() if key != "version")
    if intrinsics["cam_type"] != "radial":
        raise ValueError(f"Unexpected camera model {intrinsics['cam_type']}")
    return intrinsics


def normalized2d_to_direction3d(point_2d, intrinsics):
    """
    Input shape: (2,)
    Output shape: (3,)
    """
    import cv2

    point_2d = np.asarray(point_2d, dtype="float64").reshape(2)
    camera_matrix = np.asarray(intrinsics["camera_matrix"])
    dist_coeffs = np.asarray(intrinsics["dist_coefs"])
    width, height = intrinsics["resolution"]

    # denormalize
    point_2d[:, 0] *= width
    point_2d[:, 1] = (1.0 - point_2d[:, 1]) * height

    # undistort
    point_2d = cv2.undistortPoints(point_2d, camera_matrix, dist_coeffs)

    # unproject
    point_3d = cv2.convertPointsToHomogeneous(point_2d)
    point_3d.shape = 3

    return point_3d


def direction3d_to_normalized2d(point_3d, intrinsics):
    """
    Input shape: (3,)
    Output shape: (2,)
    """
    import cv2

    point_3d = np.asarray(point_3d, dtype="float64").reshape((1, -1, 3))
    camera_matrix = np.asarray(intrinsics["camera_matrix"])
    dist_coeffs = np.asarray(intrinsics["dist_coefs"])
    width, height = intrinsics["resolution"]

    # rotation and translation of the camera, zero in our case
    rvec = tvec = np.zeros((1, 1, 3))

    # project and distort
    points_2d, _ = cv2.projectPoints(point_3d, rvec, tvec, camera_matrix, dist_coeffs)
    x, y = points_2d.reshape(2)

    # normalize
    x /= width
    y = 1.0 - y / height

    return x, y


if __name__ == "__main__":
    # setup logging
    logging.basicConfig(
        level=logging.NOTSET, format="%(message)s", handlers=[RichHandler()]
    )

    # setup command line interface
    parser = argparse.ArgumentParser(
        description=(
            "Extract confidence from recorded pupil data and extracts blinks with the "
            "same algorithm as the Pupil Player post-hoc blink detector. See also "
            "https://docs.pupil-labs.com/core/software/pupil-player/#blink-detector"
        )
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.6, help="Minimum gaze confidence"
    )
    parser.add_argument(
        "--max-dispersion",
        type=float,
        default=1.5,
        help="Maximum gaze dispersion in degrees",
    )
    parser.add_argument(
        "--min-duration", type=int, default=80, help="Minimum duration in milliseconds"
    )
    parser.add_argument(
        "--max-duration", type=int, default=220, help="Maximum duration in milliseconds"
    )
    parser.add_argument(
        "--out-prefix", help="CSV file name of the extracted data", default="fixations"
    )
    parser.add_argument(
        "-f",
        "--overwrite",
        action="store_true",
        help=(
            "Usually, the command refuses to overwrite existing csv files. "
            "This flag disables these checks."
        ),
    )
    parser.add_argument("recordings", nargs="+", help="One or more recordings")

    # parse command line arguments and start the main procedure
    args = parser.parse_args()
    main(
        recordings=args.recordings,
        csv_out_prefix=args.out_prefix,
        overwrite=args.overwrite,
        min_confidence=args.min_confidence,
        max_dispersion_deg=args.max_dispersion,
        min_duration_ms=args.min_duration,
        max_duration_ms=args.max_duration,
    )