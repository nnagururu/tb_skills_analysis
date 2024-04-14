from pathlib import Path
import pprint
import msgpack
import numpy as np
import cv2

# intrinsics_location = Path("../world.intrinsics")
# world_lookup = Path("/Users/nimeshnagururu/Documents/skills_analysis/sample_recording_v2/world_lookup.npy")
# assert intrinsics_location.exists()

# with intrinsics_location.open("rb") as fh:
#     intrinsics = msgpack.unpack(fh)

# pprint.pprint(intrinsics)

intrinsics = {
    '(640, 480)': {
        'cam_type': 'radial',
        'camera_matrix': [
            [6668.50, 0.0, 320],
            [0.0, 6668.50, 240],
            [0.0, 0.0, 1.0]
        ],
        'dist_coefs': [
            [6.15935887e+04, 6.35386798e+04, 1.45303461e-01, -7.91757365e-01, 2.13849355e+04]
        ],
        'resolution': [640, 480]
    },
    'version': 1
}

msgpack_file = Path("../world.intrinsics")
msgpack_file.write_bytes(msgpack.packb(intrinsics))