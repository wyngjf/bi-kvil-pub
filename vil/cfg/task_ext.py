from typing import Dict
from marshmallow_dataclass import dataclass

from vil.utils.utils import get_default_demo_path


@dataclass
class KVILTaskConfig:
    path:               str = ""
    n_demos:            int = 6
    path_postfix:       str = ""
    flag_mask:          bool = True
    flag_robot:         bool = True
    flag_stereo:        bool = False
    flag_resume:        bool = False
    additional_obj:     Dict[str, str] = None
    mask_erode_radius:  Dict[str, int] = None

    def __post_init__(self):
        self.path = get_default_demo_path() / self.path
