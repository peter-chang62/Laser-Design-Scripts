"""
Demonstration of a phase bias in the linear section of a figure 9
"""

import numpy as np

rot = lambda theta: np.array(
    [
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ]
)

wp = lambda phi: np.array(
    [
        [np.exp(1j * phi / 2), 0],
        [0, np.exp(-1j * phi / 2)],
    ]
)

wp_rot = lambda wp, theta: rot(-theta) @ wp @ rot(theta)

hwp = wp(np.pi)

phase_bias = 25 * np.pi / 180
wp_pb = wp(phase_bias)

x = np.array([1, 1])
# y = (
#     wp_rot(wp_pb, -np.pi / 4)
#     @ wp_rot(hwp, np.pi / 4 / 2)  # but the angles just add
#     @ wp_rot(hwp, np.pi / 2 / 2)
#     @ wp_rot(hwp, np.pi / 4 / 2)
#     @ wp_rot(wp_pb, np.pi / 4)
#     @ x
# )

y = wp_rot(wp_pb, -np.pi / 4) @ wp_rot(hwp, np.pi / 2) @ wp_rot(wp_pb, np.pi / 4) @ x

p = abs(np.angle(y)) * 180 / np.pi
print(abs(np.diff(p)))
