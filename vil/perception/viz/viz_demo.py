import click
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from pathlib import Path
from typing import Union
from robot_utils import console
from robot_utils.math.viz.polyscope import draw_frame_3d
from robot_utils.py.interact import ask_checkbox_with_all
from robot_utils.serialize.dataclass import load_dict_from_yaml
from robot_utils.py.filesystem import validate_path, validate_file, get_ordered_files, get_ordered_subdirs
from vil.hierarchy.functions import create_local_frame_on_hand
from vil.cfg import hand_edges

ps.set_autocenter_structures(False)
ps.set_autoscale_structures(False)
ps.init()
ps.set_length_scale(1.)
ps.set_up_dir("neg_y_up")
ps.set_navigation_style("free")
ps.set_ground_plane_mode("none")


class VizDemoTraj:
    def __init__(self, path: Union[str, Path], auto_viz_all: bool = False):
        console.log("visualize demo in 3D")
        path, _ = validate_path(path, throw_error=True)
        result_path, _ = validate_path(path / "results", throw_error=True)
        traj_files = get_ordered_files(result_path, ex_pattern=["uv"], pattern=[".npy"])
        # point_idx = load_dict_from_yaml(validate_file(path / "obj/inlier_idx.yaml", throw_error=True)[0])
        if not auto_viz_all:
            traj_files = ask_checkbox_with_all("Select trajectory:", traj_files)
            if len(traj_files) < 1:
                console.print(f"[red]no trajectories found/selected in {result_path}")

        self.obj_list = [t.stem for t in traj_files]
        self.n_obj = len(self.obj_list)
        console.print(f"objects in scene: {self.obj_list}")

        # origin = draw_frame_3d(np.zeros(6, dtype=float), scale=1, radius=0.005, alpha=0.8)
        self.traj = []  # (T, N, 3), usually T = 30, N = 21 or 300
        self.current_pcl_points = []
        self.current_hand_frames = []
        self.current_hand_skeletons = []
        self.hand_index_list = []
        self.frame_mat_list = []
        for i, traj_file in enumerate(traj_files):
            # obj = traj_file.stem
            traj = np.load(traj_file)
            # if "_hand" not in obj:
            #     traj = traj[:, point_idx[obj]]
            self.traj.append(traj)

            self.t_max, self.n_points = traj.shape[:2]
            self.current_pcl_points.append(
                ps.register_point_cloud(
                    f"pcl_{self.obj_list[i]}", traj[0], radius=0.01, enabled=True, point_render_mode="sphere"
            ))
            if "_hand" in str(traj_file):
                new_frame_mat = create_local_frame_on_hand(traj)
                self.frame_mat_list.append(new_frame_mat)
                self.hand_index_list.append(i)

                idx = len(self.frame_mat_list)
                collections = draw_frame_3d(
                    new_frame_mat[0], label=f"hand_frame_{idx}", scale=0.1,
                    radius=0.01, alpha=1.0, collections=None, enabled=True
                )
                self.current_hand_frames.append(collections)

                self.current_hand_skeletons.append(ps.register_curve_network(
                    f"hand_skeleton_{idx}", nodes=traj[0], edges=hand_edges, enabled=True
                ))

        ps.set_user_callback(self.callback)
        self.time_changed, self.current_time = False, 0
        ps.show()

    def callback(self):
        psim.PushItemWidth(150)
        psim.TextUnformatted("KVIL GUI")
        psim.Separator()

        self.time_changed, self.current_time = psim.SliderInt("time", v=self.current_time, v_min=0, v_max=self.t_max-1)
        if self.time_changed:
            for i in range(self.n_obj):
                self.current_pcl_points[i].update_point_positions(self.traj[i][self.current_time])
            for idx, frame_mat in enumerate(self.frame_mat_list):
                draw_frame_3d(
                    frame_mat[self.current_time], label=f"hand_frame_{idx}", scale=0.1, radius=0.01,
                    alpha=1.0, collections=self.current_hand_frames[idx], enabled=True
                )
                self.current_hand_skeletons[idx].update_node_positions(
                    self.traj[self.hand_index_list[idx]][self.current_time]
                )


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--path",      "-p",    type=str,   help="the absolute path to skill's root")
@click.option("--namespace", "-n",    type=str,   help="the namespace to viz")
def main(path, namespace):
    if not namespace:
        console.log("[bold red]You have to specify namespace with '-n'")
        exit()
    path = validate_path(path, throw_error=True)[0]

    trial_paths = get_ordered_subdirs(path / namespace / "data/")
    trial_name_list = [t.stem for t in trial_paths]
    selected_trails = ask_checkbox_with_all("select trials: ", trial_name_list)
    selected_trial_paths = [trial_paths[trial_name_list.index(t)] for t in selected_trails]

    for t in selected_trial_paths:
        viz = VizDemoTraj(t)


if __name__ == "__main__":
    main()

