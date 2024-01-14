import click
import numpy as np
import polyscope as ps
import polyscope.imgui as psim
from pathlib import Path
from typing import Union
from robot_utils import console
from robot_utils.math.viz.polyscope import draw_frame_3d
from robot_utils.py.interact import ask_checkbox
from robot_utils.py.filesystem import validate_path, get_ordered_files
from vil.hierarchy.functions import create_local_frame_on_hand

ps.set_autocenter_structures(False)
ps.set_autoscale_structures(False)
ps.init()
ps.set_length_scale(1.)
ps.set_up_dir("z_up")
ps.set_navigation_style("free")
ps.set_ground_plane_mode("none")


class VizDemoTraj:
    def __init__(self, path: Union[str, Path], auto_viz_all: bool = False):
        console.log("visualize demo in 3D")
        path, _ = validate_path(path, throw_error=True)
        result_path, _ = validate_path(path / "results", throw_error=True)
        trajs = get_ordered_files(result_path, ex_pattern=["uv"], pattern=[".npy"])
        if not auto_viz_all:
            trajs = ask_checkbox("Select trajectory:", trajs)
            if len(trajs) < 1:
                console.print(f"[red]no trajectories found/selected in {result_path}")

        self.obj_list = [t.stem for t in trajs]
        self.n_obj = len(self.obj_list)
        console.print(f"objects in scene: {self.obj_list}")

        # origin = draw_frame_3d(np.zeros(6, dtype=float), scale=1, radius=0.005, alpha=0.8)
        self.traj = []
        self.current_pcl_points = []
        self.frame_mat_list = [] 
        for i, t in enumerate(trajs):
            traj = np.load(t)
            self.traj.append(traj)
            # ic(traj.shape)
            self.t_max, self.n_points = traj.shape[:2]
            self.current_pcl_points.append(
                ps.register_point_cloud(
                    f"pcl_{self.obj_list[i]}", traj[0], radius=0.01, enabled=True, point_render_mode="quad"
            ))
            if "hand" in str(t):
                new_frame_mat = create_local_frame_on_hand(traj)
                self.frame_mat_list.append(new_frame_mat)
        # ic(trajs)
        # ic(self.frame_mat_list,len(self.frame_mat_list))
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
                config = frame_mat[self.current_time]
                # ic(config.shape)
                draw_frame_3d(config, label=f"hand_frame_{idx}", scale=0.1, radius=0.01,
                alpha=1.0, collections=None, enabled=True)


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option("--path",     "-p",    type=str,   help="the absolute path to the training root folder of dcn")
def main(path):
    viz = VizDemoTraj(path)


if __name__ == "__main__":
    main()

