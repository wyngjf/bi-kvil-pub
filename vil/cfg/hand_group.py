from typing import Dict
from marshmallow_dataclass import dataclass

from robot_utils.py.utils import default_field


@dataclass
class HandGroup:
    # hand configs
    handedness: str = None
    hand_name: str = None
    hand_idx: int = -1
    is_grasping: bool = False

    # object configs
    object_name: str = None
    object_idx: int = -1

    # task configs
    is_master: bool = False

    # TODO to be removed
    is_non_dominant: bool = None
    is_dominant: bool = None

    def __post_init__(self):
        if self.is_dominant is not None or self.is_non_dominant is not None:
            raise ValueError("You have to use is_master now, fix your config")


@dataclass
class HandGroupConfig:
    hand_group_dict: Dict[str, HandGroup] = default_field({})

    # task configs
    is_symmetric: bool = False
    is_uncoordinated: bool = False
