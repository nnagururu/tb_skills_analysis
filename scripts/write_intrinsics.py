from pathlib import Path
import pprint
import msgpack
import numpy as np
import cv2

intrinsics_location = Path("/Users/nimeshnagururu/Documents/skills_analysis/world.intrinsics")
world_lookup = Path("/Users/nimeshnagururu/Documents/skills_analysis/sample_recording_v2/world_lookup.npy")
assert intrinsics_location.exists()

with intrinsics_location.open("rb") as fh:
    intrinsics = msgpack.unpack(fh)

pprint.pprint(intrinsics)

# intrinsics = {
#     '(640, 480)': {
#         'cam_type': 'radial',
#         'camera_matrix': [
#             [2666.667, 0.0, 320],
#             [0.0, 2666.667, 240],
#             [0.0, 0.0, 1.0]
#         ],
#         'dist_coefs': [
#             [-0.43738542863224966, 0.190570781428104, -0.00125233833830639, 0.0018723428760170056, -0.039219091259637684]
#         ],
#         'resolution': [640, 480]
#     },
#     'version': 1
# }

# msgpack_file = Path("/Users/nimeshnagururu/Documents/skills_analysis/world.intrinsics")
# msgpack_file.write_bytes(msgpack.packb(intrinsics))