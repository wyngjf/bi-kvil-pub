from .config import *
from .kvil_gui import *

priorities = {
    "p2p": 0, "p2l": 1, "p2P": 1, "p2c": 2, "p2S": 2, "free": 0, "grasping": 3
}
priority_factors = [3.0, 0.15, 0.1, 0.1, 0.1]

hand_edges = np.array([
    [4, 3], [3, 2], [2, 1], [1, 0], [0, 5], [5, 9], [9, 13], [13, 17], [17, 0], [5, 6], [6, 7], [7, 8],
    [9, 10], [10, 11], [11, 12], [13, 14], [14, 15], [15, 16], [17, 18], [18, 19], [19, 20]
])

ssh_info = {
    'hostname': '10.6.2.100',
    'port': 22,
    'username': "armar-user",
    'timeout': 5
}

human_hand_to_robot_map = {
    12: np.array([30.50455481, 8.50002948, 170.71818479]),
     9: np.array([30.50455481, -12.50002948,  70.71818479]),
     0: np.array([30.50455481, -12.50002948, -40.71818479]),
     1: np.array([30.50455481, -45.50002948, -40.71818479]),
    11: np.array([-10, 0, 50.])
}

robot_cam_intrinsic = [
    [600.187, 0.0, 640.16], [0.0, 600.05, 366.259], [0, 0, 1]
]
