from pathlib import Path
from robot_utils.serialize.schema_numpy import DictNumpyArray
from robot_utils.py.filesystem import get_ordered_files, get_ordered_subdirs, copy2, create_path, validate_path
from robot_utils.serialize.dataclass import save_to_yaml, load_dict_from_yaml, load_dataclass, dump_data_to_dict, \
    dump_data_to_yaml
from vil.cfg.preprocessing import HumanProperty, HumanID
from robot_utils.py.interact import ask_checkbox
from typing import Union, List, Dict, Any
from robot_utils import console
from vil.cfg import KVILConfig
from vil.perception.viz.viz_demo import VizDemoTraj
from icecream import ic
import numpy as np
import click
import torch


def create_kvil_obj_canonical(path: Path):
    canonical_path = path / "canonical"
    kvil_canonical_path = create_path(canonical_path / "kvil")
    dcn_cfg_file = canonical_path / "dcn/canonical_cfg.yaml"
    data = load_dict_from_yaml(dcn_cfg_file)
    dcn_obj_cfg = data.get("dcn_obj_cfg", None)
    dcn_obj_list = list(dcn_obj_cfg.keys())
    # ic(dcn_obj_list)
    path_descriptor = canonical_path / "dcn/descriptors.pth"
    descriptor = torch.load(path_descriptor)
    # ic(descriptor)
    for obj in dcn_obj_list:
        obj_canonical_folder = canonical_path / f"dcn/{obj}"
        obj_uv_file = obj_canonical_folder / "uv.yaml"
        obj_coordinates_file = obj_canonical_folder / "coordinates_3d.yaml"
        uv = load_dict_from_yaml(obj_uv_file)
        coordinates = load_dict_from_yaml(obj_coordinates_file)
        # uv = np.array(uv, dtype=int)
        # coordinates = np.array(coordinates)
        obj_descriptor = descriptor[obj].numpy().tolist()
        obj_canonical_dict = {
            "uv": uv,
            "descriptors": obj_descriptor,
            "coordinates": coordinates
        }
        obj_canonical_file = kvil_canonical_path / f"{obj}.yaml"
        save_to_yaml(obj_canonical_dict, obj_canonical_file)
    return dcn_obj_list


def create_kvil_hand_canonical(task_path: Path, demo_path: Path):
    kvil_canonical_path = task_path / "canonical/kvil"
    human_path = demo_path / "human"
    result_filename = human_path / "human_id.yaml"
    data = load_dict_from_yaml(result_filename)
    human_id = load_dataclass(HumanID, data["mediapipe_humans"])
    left_hand_idx = human_id.humans[0].left_hand_idx
    right_hand_idx = human_id.humans[0].right_hand_idx
    hand_result_file = human_path / "graphormer/xyz.yaml"
    hand_uv_file = human_path / "graphormer/uv.yaml"
    xyz = DictNumpyArray.from_yaml(hand_result_file).data
    uv = DictNumpyArray.from_yaml(hand_uv_file).data
    # ic(xyz, uv)
    # ===============TODO: multi human cases
    # num_humans = len(human_id.humans)
    # for i in range(num_humans): 

    # ====temp code for kvil test
    hand_list = ["left_hand", "right_hand"]
    hand_idx_list = [left_hand_idx, right_hand_idx]
    for idx, hand in enumerate(hand_list):
        hand_idx = hand_idx_list[idx]
        hand_name = f"hand_{hand_idx:>02d}"
        ic(hand_name)
        hand_uv = uv[hand_name]
        hand_xyz = xyz[hand_name]
        ic(hand_uv.shape)
        init_uv = hand_uv[0].tolist()
        init_xyz = hand_xyz[0].tolist()
        intersection = np.arange(hand_uv.shape[1]).tolist()
        can_data_file = kvil_canonical_path / f"{hand}.yaml"
        canonical_data = {
            "uv": init_uv,
            "descriptors": intersection,
            "coordinates": init_xyz
        }
        save_to_yaml(canonical_data, can_data_file, default_flow_style=False)
    return hand_list


def allocate_the_results(obj_list, hand_list, demo_path: Path, result_folder: Path):
    demo_name = demo_path.stem
    ic(demo_name)

    # allocate hand results # TODO: change it later for multi person in the task, need auto segmentation first 
    human_path = demo_path / "human"
    result_filename = human_path / "human_id.yaml"
    data = load_dict_from_yaml(result_filename)
    human_id = load_dataclass(HumanID, data["mediapipe_humans"])
    left_hand_idx = human_id.humans[0].left_hand_idx
    right_hand_idx = human_id.humans[0].right_hand_idx
    hand_xyz_file = human_path / "graphormer/xyz.yaml"
    hand_uv_file = human_path / "graphormer/uv.yaml"
    hands_xyz = DictNumpyArray.from_yaml(hand_xyz_file).data
    hand_idx_list = [left_hand_idx, right_hand_idx]
    for idx, hand in enumerate(hand_list):
        hand_idx = hand_idx_list[idx]
        hand_name = f"hand_{hand_idx:>02d}"
        hand_traj = hands_xyz[hand_name]
        ic(hand_traj.shape)
        hand_result_file = result_folder / f"{hand}.npy"
        np.save(hand_result_file, hand_traj)

    # allocate obj results
    obj_xyz_file = demo_path / "obj/xyz.yaml"
    obj_xyz = DictNumpyArray.from_yaml(obj_xyz_file).data
    for obj in obj_list:
        obj_traj = obj_xyz[obj]
        ic(obj_traj.shape)
        obj_result_file = result_folder / f"{obj}.npy"
        np.save(obj_result_file, obj_traj)


def create_kvil_config(task_path: Path):
    config_file = create_path(task_path / "config") / "_kvil_config.yaml"
    config = KVILConfig
    dump_data_to_yaml(KVILConfig, config, config_file)


def convert_data(task_path):
    console.log(f"[bold cyan]")
    task_path = Path(task_path)
    obj_list = create_kvil_obj_canonical(task_path)
    rec = task_path / "recordings"
    trial_dirs: List[Path] = get_ordered_subdirs(validate_path(rec, throw_error=True)[0])
    to_select_from = [p.stem for p in trial_dirs]
    selected_trials = []
    while len(selected_trials) < 1:
        console.log("[bold red]You have to select at least one demo to proceed.")
        selected_trials = ask_checkbox("Select demos to proceed", ["all"] + to_select_from)

    if not (len(selected_trials) == 1 and selected_trials[0] == "all"):
        trial_dirs = [trial_dirs[to_select_from.index(i)] for i in selected_trials]
    hand_list = create_kvil_hand_canonical(task_path=task_path, demo_path=trial_dirs[0])
    ic(trial_dirs)
    num_demo = len(trial_dirs)
    create_kvil_config(task_path)
    # process_folder = create_path(task_path / f"process/demo_{num_demo:>02d}")
    # kvil_results_folder = create_path( process_folder / "preprocess_results")
    for demo_path in trial_dirs:
        result_folder = create_path(demo_path / "results")
        allocate_the_results(obj_list=obj_list, hand_list=hand_list, demo_path=demo_path, result_folder=result_folder)

        VizDemoTraj(demo_path)


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--task_path", "-p", type=str, help="the absolute path to task demo")
def main(task_path):
    convert_data(task_path)


if __name__ == "__main__":
    main()
