import os
import numpy as np
from typing import List, Union
from pathos.multiprocessing import ProcessingPool as Pool
from functools import partial
from robot_utils import console
from vil.mp.vmp import VMP


def resample_traj(
        traj: Union[str, np.ndarray],
        desired_time_steps: int = 30,
):
    if not isinstance(traj, np.ndarray):
        traj = np.load(str(traj))
    T, P, d = traj.shape
    vmp = VMP(d * P, kernel_num=30, use_out_of_range_kernel=True, elementary_type="linear")
    t = np.linspace(0, 1, T).reshape(-1, 1)
    trajs = np.concatenate((t, traj.reshape(T, P*d)), axis=-1)[np.newaxis, ...]
    vmp.train(trajs)
    # TODO fix this
    if T == 2:
        desired_time_steps = 2
    trajs = vmp.roll(start=np.array(traj[0]), goal=traj[-1], n_samples=desired_time_steps)
    return trajs[:, 1:].reshape(desired_time_steps, P, d)


def load_and_resample_all_traj(
        traj_list: List[str],  # List[(T_i, d)]
        desired_time_steps: int = 30
):
    with Pool(os.cpu_count()) as p:
        all_traj = np.array(
            p.map(partial(resample_traj, desired_time_steps=desired_time_steps), traj_list)
        )
    console.log(f"the resampled trajectory has shape {all_traj.shape}")
    return all_traj


