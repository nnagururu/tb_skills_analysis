# https://gist.github.com/papr/40ba7d99f572bc0fe388a81aa2f87424

import argparse
import logging
import os
import traceback as tb
from typing import NamedTuple

import msgpack
import numpy as np
import pandas as pd
from rich.progress import track
from scipy.signal import fftconvolve

logger = logging.getLogger(__name__)


def main(recordings, csv_out_prefix, overwrite=False, **algorithm_kwargs):
    """Process given recordings one by one

    Iterates over each recording and handles cases where no pupil.pldata or
    pupil_timestamps.npy files could be found.

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
            logger.warning(
                f"The recording {rec} did not include any prerecorded pupil files!"
            )
            logger.debug(tb.format_exc())


def process_recording(recording, csv_out_prefix, overwrite=False, **algorithm_kwargs):
    """Process a single recording

    recordings: List of recording folders
    csv_out: CSV file name prefix under which the result will be saved
    overwrite: Boolean indicating if an existing csv file should be overwritten
    """
    csv_out_path_class = os.path.join(recording, csv_out_prefix + "-classification.csv")
    csv_out_path_interm = os.path.join(recording, csv_out_prefix + "-intermediate.csv")
    if os.path.exists(csv_out_path_class):
        if not overwrite:
            logger.warning(f"{csv_out_path_class} exists already! Not overwriting.")
            return
        else:
            logger.warning(f"{csv_out_path_class} exists already! Overwriting.")

    pupil_data = pd.DataFrame(load_and_yield_confidence_data(recording))
    blinks = run_blink_detection(pupil_data, **algorithm_kwargs)
    logger.info(f"Writing intermediate data to {csv_out_path_interm}")
    blinks.intermediate.to_csv(csv_out_path_interm, index=False)
    logger.info(f"Writing classification results to {csv_out_path_class}")
    blinks.classification.to_csv(csv_out_path_class, index=False)


class ConfidenceDatum(NamedTuple):
    timestamp: float
    eye_id: int
    detector: str
    confidence: float


def load_and_yield_confidence_data(directory, topic="pupil"):
    """Load and extract pupil diameter data

    See the data format documentation[2] for details on the data structure.

    Adapted open-source code from Pupil Player[1] to read pldata files.
    Removed the usage of Serialized_Dicts since this script has the sole purpose
    of running through the data once.

    [1] https://github.com/pupil-labs/pupil/blob/master/pupil_src/shared_modules/file_methods.py#L137-L153
    [2] https://docs.pupil-labs.com/#data-files
    """
    ts_file = os.path.join(directory, topic + "_timestamps.npy")
    data_ts = np.load(ts_file)

    msgpack_file = os.path.join(directory, topic + ".pldata")
    with open(msgpack_file, "rb") as fh:
        unpacker = msgpack.Unpacker(fh, raw=False, use_list=False)
        for timestamp, (topic, payload) in track(
            zip(data_ts, unpacker),
            description="[INFO] Load pupil confidence data",
            total=len(data_ts),
        ):
            if "2d" not in topic:
                continue
            datum = deserialize_msgpack(payload)

            # custom extraction function for pupil data, see below for details
            eye_id, conf = extract_eyeid_confidence(datum)

            # yield data according to csv_header() sequence
            yield ConfidenceDatum(timestamp, eye_id, topic.split(".")[2], conf)


def extract_eyeid_confidence(pupil_datum):
    """Extract data for a given pupil datum

    Returns: tuple(eye_id, confidence)
    """
    return (pupil_datum["id"], pupil_datum["confidence"])


def deserialize_msgpack(msgpack_bytes):
    """Deserialize msgpack[1] data

    [1] https://msgpack.org/index.html
    """
    return msgpack.unpackb(msgpack_bytes, raw=False, use_list=False)


class BlinkDatum(NamedTuple):
    id: int
    start_timestamp: float
    end_timestamp: float
    duration: float
    confidence: float
    filter_response: str
    base_data: str


class BlinkDetection(NamedTuple):
    intermediate: pd.DataFrame
    classification: pd.DataFrame


def run_blink_detection(
    pupil_data: pd.DataFrame,
    history_length_seconds: float = 0.2,
    onset_confidence_threshold=0.5,
    offset_confidence_threshold=0.5,
):
    if pupil_data.empty:
        logger.warning("No pupil data found! Skipping...")
        return

    logger.info("Calculating intermediate blink data...")
    pupil_data.sort_values(by="timestamp", ignore_index=True, inplace=True)
    ts_start, ts_stop = pupil_data.timestamp.iloc[[0, -1]]
    total_time = ts_stop - ts_start

    activity = pupil_data.confidence
    filter_size = 2 * round(len(pupil_data) * history_length_seconds / total_time / 2.0)
    blink_filter = np.ones(filter_size) / filter_size

    # This is different from the online filter. Convolution will flip
    # the filter and result in a reverse filter response. Therefore
    # we set the first half of the filter to -1 instead of the second
    # half such that we get the expected result.
    blink_filter[: filter_size // 2] *= -1

    # The theoretical response maximum is +-0.5
    # Response of +-0.45 seems sufficient for a confidence of 1.
    filter_response = fftconvolve(activity, blink_filter, "same") / 0.45

    onsets = filter_response > onset_confidence_threshold
    offsets = filter_response < -offset_confidence_threshold

    response_classification = np.zeros(filter_response.shape)
    response_classification[onsets] = 1.0
    response_classification[offsets] = -1.0

    intermediate_data = pd.concat(
        [
            pd.Series(activity, name="activity"),
            pd.Series(filter_response, name="filter_response"),
            pd.Series(response_classification, name="response_classification").map(
                {0.0: None, -1.0: "offset", 1.0: "onset"}
            ),
        ],
        axis=1,
    )

    # consolidation

    blink = None
    state = "no blink"  # others: 'blink started' | 'blink ending'
    blink_data = []
    counter = 1

    def start_blink(idx):
        nonlocal blink
        nonlocal state
        nonlocal counter
        blink = {
            "start_timestamp": pupil_data.timestamp.iloc[idx],
            "blink_id": counter,
            "__start_index__": idx,
        }
        state = "blink started"
        counter += 1

    def blink_finished(idx):
        nonlocal blink

        start_index = blink["__start_index__"]
        blink["end_timestamp"] = pupil_data.timestamp.iloc[idx]

        blink_data.append(
            BlinkDatum(
                id=blink["blink_id"],
                start_timestamp=blink["start_timestamp"],
                duration=blink["end_timestamp"] - blink["start_timestamp"],
                end_timestamp=blink["end_timestamp"],
                # blink confidence is the mean of the absolute filter response
                # during the blink event, clamped at 1.
                confidence=min(
                    float(np.abs(filter_response[start_index:idx]).mean()),
                    1.0,
                ),
                base_data=" ".join(map(str, pupil_data.timestamp[start_index:idx])),
                filter_response=" ".join(map(str, filter_response[start_index:idx])),
            )
        )

    for idx, classification in track(
        enumerate(response_classification),
        description="[INFO] Blink classification",
        total=len(response_classification),
    ):
        if state == "no blink" and classification > 0:
            start_blink(idx)
        elif state == "blink started" and classification == -1:
            state = "blink ending"
        elif state == "blink ending" and classification >= 0:
            blink_finished(idx - 1)  # blink ended previously
            if classification > 0:
                start_blink(0)
            else:
                blink = None
                state = "no blink"

    if state == "blink ending":
        # only finish blink if it was already ending
        blink_finished(idx)  # idx is the last possible idx

    return BlinkDetection(
        intermediate=intermediate_data, classification=pd.DataFrame(blink_data)
    )


if __name__ == "__main__":
    # setup logging
    logging.basicConfig(level=logging.DEBUG, format="[%(levelname)s] %(message)s")

    # setup command line interface
    parser = argparse.ArgumentParser(
        description=(
            "Extract confidence from recorded pupil data and extracts blinks with the "
            "same algorithm as the Pupil Player post-hoc blink detector. See also "
            "https://docs.pupil-labs.com/core/software/pupil-player/#blink-detector"
        )
    )
    parser.add_argument(
        "-fl",
        "--filter-length",
        type=float,
        default=0.2,
        help="The time window's length in which the detector tries to find confidence "
        "drops and gains, in seconds",
    )
    parser.add_argument(
        "-on",
        "--onset",
        type=float,
        default=0.5,
        help="The threshold that the filter response ('Activity') must rise above to "
        "classify the onset of a blink, corresponding to a sudden *drop* in 2D pupil "
        "detection confidence",
    )
    parser.add_argument(
        "-off",
        "--offset",
        type=float,
        default=0.5,
        help="The threshold that the filter response ('Activity') must fall below to "
        "classify the end of a blink, corresponding to a *rise* in 2D pupil detection "
        "confidence",
    )
    parser.add_argument(
        "--out-prefix",
        help="CSV file name prefix the extracted data",
        default="blink",
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
        history_length_seconds=args.filter_length,
        onset_confidence_threshold=args.onset,
        offset_confidence_threshold=args.offset,
    )