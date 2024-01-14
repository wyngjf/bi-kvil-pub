import os
from pathlib import Path

import numpy as np


def get_root_path():
    return Path(__file__).parent.parent.parent


def get_example_path():
    return get_root_path() / "examples"


def get_data_path():
    return get_root_path() / "data"


def get_vil_path():
    return get_root_path() / "vil"


def load_npz(filename):
    return dict(np.load(filename))


def get_default_demo_path():
    path = os.environ.get("DEFAULT_KVIL_DEMO_PATH")
    if path is None:
        raise EnvironmentError("You have to set DEFAULT_KVIL_DEMO_PATH variable pointing to the "
                               "folder that contains all the K-VIL demos")
    return Path(path)
