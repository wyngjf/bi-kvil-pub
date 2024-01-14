# from asyncio import selector_events
import os
import matplotlib.pyplot as plt
import numpy as np
import polyscope as ps
import torch

from typing import Dict, Tuple
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_quaternion
from pathlib import Path
from functools import partial
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from sklearn.neighbors import LocalOutlierFactor
from pathos.multiprocessing import ProcessingPool as Pool

from robot_utils import console
from robot_utils.py.filesystem import get_validate_path
from robot_utils.py.utils import load_dict_from_yaml, dump_data_to_yaml, load_dataclass
from robot_utils.math.viz.polyscope import draw_frame_3d
from robot_utils.log import disable_logging, enable_logging

from vil.cfg import ObjectConfig, KVILProcessData, hand_edges, UIConfig
from vil.constraints import find_neighbors, find_local_frame, pca
from vil.mp import load_and_resample_all_traj, resample_traj


class RealObject:
    def __init__(self, config: ObjectConfig):
        self.c = config
        if not self.c.name:
            console.log(f"[bold red]{self.c.name} is invalid")
            exit(1)

        console.log(f"[bold]----")
        console.log(f"[bold green]Prepare Object [bold blue]{self.c.name}")

        # The self.c.path to a the experiment folder of a specific number of demos, e.g. .../kvil/process/demo_03
        self.kvil_root_path = self.c.path.parent.parent  # the 'kvil' folder the skill root
        self.obj_data_file = self.c.path / f"obj_data/{self.c.name}.npz"
        self.obj_cfg_file = self.c.path / f"obj_config/{self.c.name}.yaml"

        self._load_canonical_shape()

        if self.c.force_redo or not self.obj_data_file.is_file():
            # general
            # self._get_inlier()
            self._load_demo_traj()

            # for kvil
            self._update_kvil_data()
            self._compute_spatial_scale()
            self._find_neighbors()
            self._detect_local_frame()
            self._save_obj_data()
        else:
            self._load_obj_data()
            self._update_kvil_data()

        self.slave_variation: Dict[str, Dict[int, bool]] = {}

        # K-VIL
        self.d = KVILProcessData()
        self.d.data_file = get_validate_path(self.c.path / f"process") / f"{self.c.name}.npz"

        # drawing variables:
        self.skeleton = []
        self.pcl = []
        self.frame_master_global = [None] * self.n_demo
        self.demo_colors = plt.get_cmap("plasma")(np.linspace(0, 1, self.n_demo))  # (n_demos, 3)
        if self.c.is_hand:
            self.hand_edges = hand_edges

        console.log(f"[bold]Done (construct object {self.c.name})")

    def _update_kvil_data(self):
        """
        --- kvil ---
        """
        self.coordinates = self.coordinates_all[self.inlier_index]
        self.coordinates = self.coordinates - self.coordinates.mean(0)  # center the canonical shape
        self.descriptors = self.descriptors_all[self.inlier_index]
        self.uv = self.uv_all[self.inlier_index]
        console.log(
            f"[cyan]-- {self.c.name}: [green]{self.c.name}, can shape: {self.coordinates.shape}, global_traj: {self.global_traj.shape}\n"
            f"[cyan]-- {self.c.name}: [green]kvil_time_idx: {self.kvil_time_idx}, {self.inlier_index.shape} inliers")
        self.kvil_traj = self.global_traj.take(self.kvil_time_idx, 1).take(self.inlier_index, 2)  # (N, T, P, 3)
        self.n_demo, self.n_time, self.n_candidates_kvil, self.dim = self.kvil_traj.shape
        self.n_time_raw, self.n_candidates_raw = self.global_traj.shape[1:3]
        console.log(f"[cyan]-- {self.c.name}: [green]the K-VIL trajectory: (N, T, P, dim) = {self.kvil_traj.shape}")

    def _save_obj_data(self):
        """
        --- kvil ---
        """
        data_dict = dict(
            inlier_idx=self.inlier_index,
            outlier_mask=self.outlier_mask,
            global_traj=self.global_traj,
            kvil_time_idx=self.kvil_time_idx,
            hull_indices=self.hull_indices,
            neighbor_idx=self.neighbor_idx,
            frame_mat_all_demo=self.frame_mat_all_demo,
        )
        np.savez(str(self.obj_data_file), **data_dict)

    def _load_obj_data(self):
        """
        --- kvil ---
        """
        console.log(f"loading {self.c.name} from {self.obj_data_file}")
        data_dict: Dict[str, np.ndarray] = dict(np.load(str(self.obj_data_file)))
        self.inlier_index = data_dict["inlier_idx"]                 # (P, )
        self.outlier_mask = data_dict["outlier_mask"]
        self.global_traj = data_dict["global_traj"]                 # (N, T, P, dim)
        self.kvil_time_idx = data_dict["kvil_time_idx"]             # (T, )
        self.hull_indices = data_dict["hull_indices"]
        self.neighbor_idx = data_dict["neighbor_idx"]               # (P, Q)
        self.frame_mat_all_demo = data_dict["frame_mat_all_demo"]   # (N, T, P, d, d+1)

    def _load_canonical_shape(self):
        """
        --- Canonical Shape ---
        Load canonical shape from "demo_root/ref_images/obj_name/canonical.yaml", data entries including
        - uv (can be used for optical flow based method)
        - coordinates (3D coordinates in camera frame)
        - descriptors (descriptors generated by DCN model)
        """
        if self.c.is_hand:
            canonical_config_file = self.c.path.parent.parent / "canonical" / f"{self.c.original_obj}.yaml"
            canonical_cfg = load_dict_from_yaml(canonical_config_file)

            self.uv_all = np.array(canonical_cfg["uv"])
            self.coordinates_all = np.array(canonical_cfg["coordinates"])
            self.descriptors_all = np.array(canonical_cfg["descriptors"])
            self.can_inlier_idx_list = list(range(self.uv_all.shape[0]))
            return

        dcn_can_space = self.kvil_root_path.parent / f"canonical/dcn/{self.c.original_obj}"
        self.uv_all = np.array(load_dict_from_yaml(dcn_can_space / "uv.yaml"))
        self.coordinates_all = np.array(load_dict_from_yaml(dcn_can_space / "coordinates_3d_fixed.yaml"))
        self.descriptors_all = np.array(load_dict_from_yaml(dcn_can_space / "descriptor.yaml"))
        self.can_inlier_idx_list = load_dict_from_yaml(dcn_can_space / "can_inlier.yaml")

    def _get_inlier(self):
        """
        --- Canonical Shape ---
        Compute inlier indices and outlier masks
        """
        console.rule(f"[green]{self.c.name}: inlier detection")
        _n_sample_pts = self.coordinates_all.shape[0]

        if self.c.kvil.remove_outlier and "hand" not in self.c.name:
            inlier_index = self.local_outlier_detector()
        else:
            inlier_index = np.arange(_n_sample_pts)

        if len(inlier_index) == 0:
            console.print("[bold red]All points are considered as outlier, please check your data")
            exit()

        self.inlier_index = inlier_index
        self.n_candidates_kvil = len(self.inlier_index)
        self.outlier_mask = np.ones(_n_sample_pts, dtype=bool)  # True for outliers
        self.outlier_mask[inlier_index] = False

    def _compute_spatial_scale(self):
        """
        --- Canonical Shape ---
        Compute the spatial scaling from the convex hull of the canonical shape
        """
        console.log(f"[cyan]-- {self.c.name}: [green]compute spatial scale")
        # self.coordinates = self.coordinates_all[self.inlier_index]
        # self.uv = self.uv_all[self.inlier_index]
        hull = ConvexHull(self.coordinates)
        self.hull_indices = hull.vertices
        hull_points = self.coordinates[hull.vertices, :]  # Extract the points forming the hull
        self.c.spatial_scale = cdist(hull_points, hull_points, metric='euclidean').max()
        if self.c.is_hand:
            self.c.spatial_scale *= 3.0

        self.current_scale = np.array([1] * 3, dtype=float)

    def local_outlier_detector(self):
        """
        -- Canonical Shape --
        outlier detector using sklearn
        """
        clf = LocalOutlierFactor(n_neighbors=30, contamination='auto')
        # clf = LocalOutlierFactor(n_neighbors=30, contamination=0.01)
        in_out_lier_indicator = clf.fit_predict(self.coordinates_all)
        inlier_idx = np.where(in_out_lier_indicator == 1)[0]
        return inlier_idx

    def _load_demo_traj(self):
        """
        --- kvil demo ---
        Load trajectories in the demo scene, np.ndarray (N, T, P, 3).
        N: num of demos, T: num of time steps, P: num of candidates
        """
        console.log(f"[cyan]-- {self.c.name}: [green]load demo trajectory")
        # ic(self.c.traj_file_list)
        # self.global_traj = load_and_resample_all_traj(self.c.traj_file_list)
        # TODO there is potential bug if the demo only contains two time steps.
        self.global_traj = np.array(
            [resample_traj(f"{self.kvil_root_path}/data/{file}") for file in self.c.traj_file_list])  # (N, T, P, 3)
        # self.global_traj = np.array([np.load(file) for file in self.c.traj_file_list])  # (N, T, P, 3)
        # ic(self.global_traj.shape)
        N, T, P, dim = self.global_traj.shape
        if self.c.is_virtual:
            virtual_traj = self.global_traj[:, 0, :, :]
            virtual_traj = np.repeat(np.expand_dims(virtual_traj, axis=1), T, axis=1)
            self.global_traj = virtual_traj
        self.kvil_time_idx = np.linspace(0, T - 1, self.c.kvil.n_time_frames).astype(int)

        # TODO fix this.
        _n_sample_pts = self.coordinates_all.shape[0]
        self.outlier_mask = np.zeros(_n_sample_pts, dtype=bool)
        # if not self.c.is_hand and self.c.kvil.remove_outlier and not self.c.is_virtual:
        if not self.c.is_hand and self.c.kvil.remove_outlier:

            console.log(f"[cyan]-- {self.c.name}: [green]removing outliers from traj of {self.c.name}")

            # # old outlier remover
            # def get_outlier_idx(data_points):
            #     clf = LocalOutlierFactor(n_neighbors=40, contamination="auto")
            #     return np.where(clf.fit_predict(data_points) == -1)[0]
            #
            # with Pool(os.cpu_count()) as p:
            #     outlier_idx_in_all_traj = np.concatenate(
            #         p.map(get_outlier_idx, self.global_traj[:, self.kvil_time_idx].reshape((-1, P, dim)))
            #     )
            #
            # outlier_idx_in_all_traj = np.unique(outlier_idx_in_all_traj)
            # self.outlier_mask[outlier_idx_in_all_traj] = True
            # # find points that are within 30cm to the camera, consider as outlier
            # close_to_cam_point_idx = np.where(np.linalg.norm(self.global_traj[:, self.kvil_time_idx], axis=-1) < 0.2)
            # outlier_idx = np.unique(close_to_cam_point_idx[-1])
            # self.outlier_mask[outlier_idx] = True
            #
            # # get inlier index and save to file
            # self.inlier_index = np.where(np.invert(self.outlier_mask))[0]

            # new outlier_remover
            inlier_lists = []
            for trail in self.c.traj_file_list:
                inlier_idx_file_this_trail = self.kvil_root_path / "data" / Path(trail).parent.parent / "obj/inlier_idx.yaml"
                inlier_idx_this_trail = load_dict_from_yaml(inlier_idx_file_this_trail)[self.c.original_obj]
                inlier_lists.append(inlier_idx_this_trail)
            common_items = set(inlier_lists[0])
            for lst in inlier_lists[1:]:
                common_items &= set(lst)

            common_items &= set(self.can_inlier_idx_list)

            self.inlier_index = np.array(list(common_items))

            if self.inlier_index.shape[0] > self.c.kvil.n_candidates:
                idx = np.random.choice(np.arange(self.inlier_index.shape[0]), self.c.kvil.n_candidates, replace=False)
                self.inlier_index = self.inlier_index[idx]
        else:
            self.inlier_index = np.arange(_n_sample_pts)
        self.n_candidates_kvil = len(self.inlier_index)

    def _find_neighbors(self):
        """
        --- canonical shape ---
        If this object is not a slave node, then initialize local frames
        """
        # if self.c.is_virtual:
        #     console.rule(f"[green]virtual_{self.c.name}: copying local frame from real object")
        #     self.neighbor_idx = copy.deepcopy(self.v_object_data["neighbor_idx"])  # (P, num_neighbor_pts=Q)
        # else:
        console.log(f"[cyan]-- {self.c.name}: [green]initialize local frame")
        self.neighbor_idx = find_neighbors(self.coordinates, self.c.kvil.n_neighbors)  # (P, num_neighbor_pts=Q)
        console.log(f"[cyan]-- {self.c.name}: [yellow]Neighbor index in total {len(self.neighbor_idx)}: max {self.neighbor_idx.max()}")

    # def plot_coords(self, ax, color='cyan', zorder=1, alpha=1):
    #     raise NotImplementedError
    #
    # def get_shape(self):
    #     """
    #     return the pts' coordinates after scaling
    #     """
    #     return self.hull_indices

    def get_sample_point_coordinates(self):
        return self.coordinates

    def get_scaled_sample_point_coordinates(self, n_sample_pts_max: int = 1000):
        # ic(self.c.name, self.sample_pts.shape)
        if self.n_candidates_kvil > n_sample_pts_max:
            down_sample_idx = np.linspace(0, self.n_candidates_kvil - 1, n_sample_pts_max).astype(int)
            return self.coordinates[down_sample_idx] * self.current_scale
        return self.coordinates * self.current_scale

    def get_global_traj_of_keypoint(self, idx_kpt: int):
        return self.global_traj.take(self.inlier_index, 2).take(idx_kpt, 2)  # (N, T, d)

    def get_vmp_train_traj_of_slave(self, idx_frame: int, keypoint_traj: np.ndarray) -> np.ndarray:
        """
        Args:
            idx_frame: the selected local frame
            keypoint_traj: (N, T, d)

        Returns: projected trajectory of that keypoint in the given local frame
        """
        T = keypoint_traj.shape[1]
        local_frame = self.__detect_local_frame__(np.array([idx_frame]), idx_time=np.arange(T))  # (N, T, 1, d, d+1)
        local_frame = local_frame.squeeze(2)  # (N, T, d, d+1)
        return np.einsum("ntji,ntj->nti", local_frame[..., :3], keypoint_traj - local_frame[..., -1])  # (N, T, d)

    def get_all_frame_mat_for_tvmp(self, idx_frame: int, idx_time: np.ndarray) -> np.ndarray:
        local_frame = self.__detect_local_frame__(np.array([idx_frame]), idx_time=idx_time)  # (N, T, 1, d, d+1)
        return local_frame[..., :3]

    def get_tvmp_train_traj_of_slave(self, idx_frame: int) -> np.ndarray:
        """
        get the trajectory of the idx_frame-th frame in all N demos
        Args:
            idx_frame: the selected local frame

        Returns: the task space trajectory in (N, T, 8)

        """
        idx_time = np.arange(self.global_traj.shape[1])  # T = self.global_traj.shape[1]
        local_frame_traj = self.__detect_local_frame__(np.array([idx_frame]), idx_time=idx_time)  # (N, T, P, d, d+1)

        local_frame_traj[..., -1] = np.einsum(
            "ntpij,ntpi->ntpj",
            local_frame_traj[:, [0]][..., :3],                            # (N, 1, P, 3, 3)
            local_frame_traj[..., -1] - local_frame_traj[:, [0]][..., -1])         # (N, T, P, 3) - (N, 1, P, 3)

        local_frame_traj[..., :3] = np.einsum(
            "ntpij,ntpik->ntpjk",
            local_frame_traj[:, [0]][..., :3],                            # (N, 1, P, 3, 3)
            local_frame_traj[..., :3]                                               # (N, T, P, 3, 3)
        )

        return np.concatenate((
            local_frame_traj[..., -1].squeeze(),                                    # (N, T, 3)
            matrix_to_quaternion(                                                   # (N, T, 4)
                torch.from_numpy(local_frame_traj[..., :3]).to("cuda:0").float()    # (N, T, 3, 3)
            ).detach().cpu().squeeze().numpy()
        ), axis=-1)                                                                 # (N, T, 8)

    def get_global_motion_saliency(self):
        """
        --- kvil demo ---
        compute the average total relative motion of each point in this object (only inliers)
        """
        # self.global_traj (N, T_all, P_inlier, 3)
        delta_motion = self.global_traj[:, 1:, self.inlier_index] - self.global_traj[:, :-1, self.inlier_index]
        delta_motion = delta_motion.reshape(-1, delta_motion.shape[-1])  # (N*T*P, 3)
        return np.linalg.norm(delta_motion, axis=-1).sum() / len(self.inlier_index)

    def _detect_local_frame(self):
        """
        Detect local frames for all demos, all kvil time frames, and all points on the master object
        """
        # if self.c.is_virtual:
        #     console.rule(f"[green]virtual_{self.c.name} copy local frame from real object")
        #     frame_mat_all_demo = self.v_object_data["frame_mat_all_demo"]
        #     ic(frame_mat_all_demo.shape)
        #     T = frame_mat_all_demo.shape[1]
        #     v_frame_mat_all_demo = copy.deepcopy(frame_mat_all_demo[:,0,:,:,:])
        #     v_frame_mat_all_demo = np.repeat(np.expand_dims(v_frame_mat_all_demo,axis=1),T,axis = 1)
        #     ic(v_frame_mat_all_demo.shape)
        #     self.frame_mat_all_demo = v_frame_mat_all_demo
        # else:
        console.log(f"[cyan]-- {self.c.name}: [green]local frame detection")
        idx_frame = self.kvil_traj.shape[2]  # select all frames on this object
        # (N, T, P_all, 3, 4)
        self.frame_mat_all_demo = self.__detect_local_frame__(np.arange(idx_frame), self.kvil_time_idx)

    def __detect_local_frame__(self, idx_frame: np.ndarray, idx_time: np.ndarray) -> np.ndarray:
        """
        given the global trajectories of all the points (N, T, P, d), compute the local frames of each point
        in each demos. return all the local frame matrices as (N, T, P, d, d+1)
        """
        T, P, d = np.size(idx_time), np.size(idx_frame), self.dim
        _, Q = self.neighbor_idx.shape  # self.neighbor_idx (P, Q)
        frame_mat_all_demo = []
        for n in range(self.n_demo):
            # console.print(f"demo {n}")
            # global_traj (N, T_raw, P_raw, d) -[n, idx_time]-> (T, P_all, d) -[inlier]-> (T, P_all, d) => (P_all, T, d)
            new_coordinates = self.global_traj[n, idx_time].take(self.inlier_index, 1).transpose(1, 0, 2)
            # new_point_coordinates (P_all, Q, T, d) -(idx_frame)-> (P, Q, T, d) -> (T, P, Q, d) -> (T*P, Q, d)
            new_coordinates = new_coordinates[self.neighbor_idx[idx_frame]].transpose(2, 0, 1, 3).reshape(T * P, Q, d)
            this_point = new_coordinates[:, 0]
            # init_point_coordinates (P, d) -> (P, Q, d) --(tile)--> (T*P, Q, d)
            can_shape_coordinates = np.tile(self.coordinates[self.neighbor_idx[idx_frame]], (T, 1, 1))
            pool = Pool(os.cpu_count())
            frame_config = np.array(pool.map(
                partial(
                    find_local_frame,
                    init_config=np.zeros(6, dtype=float),
                    n_neighbor_pts_for_origin=self.c.kvil.n_neighbors_for_origin,
                    return_matrix=False
                ),
                new_coordinates,
                can_shape_coordinates
            ))  # (T*P, 6)
            # Note: in pytorch3d, they only have intrinsic Euler angles, so need to first convert to rzyx
            frame_mat = euler_angles_to_matrix(torch.fliplr(torch.tensor(frame_config[:, 3:])), "ZYX")  # (T*P, d, d)
            frame_mat_all_demo.append(np.concatenate(
                (frame_mat.numpy(), (frame_config[:, :3] + this_point).reshape(T * P, d, 1)),
                axis=-1
            ))  # (TP, d, d+1)

        return np.array(frame_mat_all_demo).reshape((self.n_demo, T, P, d, d + 1))

    def set_as_master(self, remove_slaves: bool = True):
        console.log(f"-- object {self.c.name} is set as master")
        self.c.is_master = True
        if remove_slaves:
            self.c.slave_obj.clear()
            self.c.slave_idx.clear()

    def is_master(self):
        return self.c.is_master

    def save_obj_config(self):
        dump_data_to_yaml(ObjectConfig, self.c, self.obj_cfg_file)

    def load_object_config(self):
        load_dataclass(ObjectConfig, self.obj_cfg_file)

    def append_object(
            self,
            obj: 'RealObject',
            is_slave: bool = False
    ):
        """
        Append a slave object
        Args:
            obj: an RealObject instance
            # obj_name:
            # obj_id: index of the slave object in all object list
            # obj_traj: the kvil trajectory of that slave object (N, T, P, d)
            # obj_scale: spatial scale of a slave object
            is_slave: whether the appended object is a slave
        """
        console.log(
            f"-- [yellow]master [cyan]{self.c.name} [yellow]add slave object"
            f": [cyan]{obj.c.name} ({obj.c.index})[yellow] with traj {obj.kvil_traj.shape}")
        self.c.appended_idx.append(obj.c.index)
        self.c.appended_obj.append(obj.c.name)
        if is_slave:
            self.set_as_slave(obj)

        self.d.appended_obj_traj[obj.c.name] = np.einsum(
            "ntpji,ntpqj->ntpqi",
            self.frame_mat_all_demo[..., :3],  # (N, T, P_master, 1, d, d)
            # (N, T, 1,  P_slave, d) - (N, T, P_master, 1, d)                    (N, T, P_master, P_slave, d)
            np.expand_dims(obj.kvil_traj, 2) - np.expand_dims(self.frame_mat_all_demo[..., -1], -2)
        )  # (N, T, P_master, P_slave, d)
        self.d.updated_version += 1

    def set_as_non_slave(self, slave_obj: 'RealObject'):
        console.log(f"[red]object {slave_obj.c.index}: {slave_obj.c.name} is no longer a slave of master {self.c.name}")
        self.c.slave_idx.remove(slave_obj.c.index)
        self.c.slave_obj.remove(slave_obj.c.name)
        self.c.slave_scale[slave_obj.c.name] = None
        slave_obj.c.unset_master(self.c.index)

    def set_as_slave(self, slave_obj: 'RealObject'):
        console.log(f"[green]object {slave_obj.c.index}: {slave_obj.c.name} is now a slave of master {self.c.name}")
        self.c.slave_idx.append(slave_obj.c.index)
        self.c.slave_obj.append(slave_obj.c.name)
        self.c.slave_scale[slave_obj.c.name] = slave_obj.c.spatial_scale
        slave_obj.c.set_master(self.c.index)

    def init_kvil(self):
        if self.is_master():
            if not self.c.force_redo:
                self.d.load()

            for i, obj_slave in enumerate(self.c.slave_obj):
                # (N, T, P_master, P_slave, d) -> (T, P_master, P_slave)
                self.d.mask[obj_slave] = np.zeros(self.d.appended_obj_traj[obj_slave].shape[1:4], dtype=bool)

    def pca(self):
        disable_logging()
        if self.c.force_redo:
            console.log(f"-- [bold green]{self.c.name}: Running PCA")
            # slave_name = self.c.slave_obj[0]
            # N, T, Pm, Ps, d = self.d.appended_obj_traj[slave_name].shape

            for i, obj_slave in enumerate(self.c.slave_obj):
                if obj_slave in self.d.pca_data:
                    console.log(f"   {self.c.name}: PCA for {obj_slave} was done, skip")
                    continue
                N, T, Pm, Ps, d = self.d.appended_obj_traj[obj_slave].shape
                console.log(f"-- -- {obj_slave}: {self.d.appended_obj_traj[obj_slave].shape}")
                slave_traj = self.d.appended_obj_traj[obj_slave].transpose((1, 2, 3, 0, 4)).reshape((-1, N, d))
                pool = Pool(os.cpu_count())
                self.d.pca_data[obj_slave] = np.array(
                    pool.map(pca, slave_traj)  # slave_traj (T * P_master * P_slave, N, d)
                ).reshape((T, Pm, Ps, d + 2, d))
                self.d.updated_version += 1
            # self.d.save("PCA")
        enable_logging()

    # def pca_only_last_t(self):
    #     if self.c.force_redo:
    #         console.rule(f"[bold cyan]{self.c.name}: Running PCA at last time step")
    #         # slave_name = self.c.slave_obj[0]
    #         # N, T, Pm, Ps, d = self.d.appended_obj_traj[slave_name].shape
    #
    #         for i, obj_slave in enumerate(self.c.slave_obj):
    #             N, T, Pm, Ps, d = self.d.appended_obj_traj[obj_slave].shape
    #             ic(N, T, Pm, Ps, d)
    #             slave_traj = self.d.appended_obj_traj[obj_slave][:, -1].transpose((1, 2, 0, 3)).reshape((-1, N, d))
    #             pool = Pool(os.cpu_count())
    #             self.d.last_pca_data[obj_slave] = np.array(
    #                 pool.map(pca, slave_traj)  # slave_traj (T = -1 , P_master * P_slave, N, d)
    #             ).reshape((Pm, Ps, d + 2, d))
    #         self.d.save("PCA Only last T")

    def pme(self):
        if self.c.force_redo:
            console.rule(f"[bold cyan]{self.c.name}: Running PCA")
            slave_name = self.c.slave_obj[0]
            N, T, Pm, Ps, d = self.d.appended_obj_traj[slave_name].shape
            for i, obj_slave in enumerate(self.c.slave_obj):
                slave_traj = self.d.appended_obj_traj[obj_slave].transpose((1, 2, 3, 0, 4)).reshape((-1, N, d))

            # self.d.save("PME")

    def draw_pcl(self, idx_time: int, alpha: float = 1.0, kvil: bool = True):
        """
        --- kvil ---
        Args:
            idx_time:
            alpha:
            kvil: if True, then use kvil_traj, else, use global_traj
        """
        # self.kvil_traj (N, T, P, 3)
        if kvil:
            actual_coordinates = self.kvil_traj[:, idx_time]
        else:
            actual_coordinates = self.global_traj[:, idx_time, self.inlier_index]  # (N, P, 3)
        if self.c.is_hand:
            if len(self.skeleton) == 0:
                for n in range(self.n_demo):
                    self.skeleton.append(ps.register_curve_network(
                        f"{self.c.name}_pcl_{n}", nodes=actual_coordinates[n], edges=self.hand_edges, enabled=True,
                        radius=self.c.point_radius, transparency=alpha, color=self.demo_colors[n]
                    ))
            else:
                for n in range(self.n_demo):
                    self.skeleton[n].update_node_positions(actual_coordinates[n])
        else:
            if len(self.pcl) == 0:
                for n in range(self.n_demo):
                    self.pcl.append(ps.register_point_cloud(
                        f"{self.c.name}_pcl_{n}", actual_coordinates[n], enabled=True,
                        radius=self.c.point_radius, point_render_mode="sphere", color=self.demo_colors[n],
                        transparency=alpha
                    ))
                    self.pcl[-1].set_color(self.demo_colors[n])
            else:
                for n in range(self.n_demo):
                    self.pcl[n].update_point_positions(actual_coordinates[n])

    def draw_pcl_local(
            self, ui: UIConfig,
            idx_time: int,
            idx_frame: int,
            alphas: Tuple[float, float] = (1.0, 1.0),
            virt_alpha_dict: Dict[str, float] = None,
            virt_enabled_dict: Dict[str, bool] = None
    ):
        """
        --- kvil ---
        Args:
            ui: the ui data structure.
            idx_time: index time
            idx_frame: index local frame on master (this obj)
            alphas: (alpha_for_master, alpha_for_slave)
            virt_alpha_dict: a dictionary containing alpha values for all virtual objects
            virt_enabled_dict: a dictionary containing enable flag of the objects
        """
        # the data struct contains all PointCloud / CurveNetwork object of all N x O objects in the demo
        obj_pcl = ui.obj_pcl
        # self.d.appended_obj_traj -- Dict[obj_name, array(N, T, P_master, P_slave, d)]
        for i, obj_name in enumerate(self.c.appended_obj):
            actual_coordinates = self.d.appended_obj_traj[obj_name][:, idx_time, idx_frame]  # (N, P_slave, d)
            alpha = alphas[0] if obj_name == self.c.name else alphas[1]
            enabled = virt_enabled_dict[obj_name]
            if virt_alpha_dict is not None:
                alpha *= virt_alpha_dict.get(obj_name, 1.0)
            if len(obj_pcl[obj_name]) == 0:  # create pcl
                if "_hand" in obj_name:
                    for n in range(self.n_demo):
                        obj_pcl[obj_name].append(ps.register_curve_network(
                            f"{obj_name}_pcl_{n}", nodes=actual_coordinates[n], edges=hand_edges, enabled=enabled,
                            transparency=alpha, color=self.demo_colors[n]
                        ))
                        obj_pcl[obj_name][-1].set_radius(ui.point_radius, relative=False)
                else:
                    for n in range(self.n_demo):
                        obj_pcl[obj_name].append(ps.register_point_cloud(
                            f"{obj_name}_pcl_{n}", actual_coordinates[n], enabled=enabled,
                            point_render_mode="sphere", color=self.demo_colors[n], transparency=alpha
                        ))
                        obj_pcl[obj_name][-1].set_radius(ui.point_radius, relative=False)
            else:  # update pcl
                if "_hand" in obj_name:
                    for n in range(self.n_demo):
                        obj_pcl[obj_name][n].update_node_positions(actual_coordinates[n])
                        obj_pcl[obj_name][n].set_transparency(alpha)
                        obj_pcl[obj_name][n].set_enabled(enabled)
                else:
                    for n in range(self.n_demo):
                        obj_pcl[obj_name][n].update_point_positions(actual_coordinates[n])
                        obj_pcl[obj_name][n].set_transparency(alpha)
                        obj_pcl[obj_name][n].set_enabled(enabled)

    def draw_frame(self, frame_idx: int, time_idx: int, enabled: bool = True):
        """
        --- kvil ---
        if this object is a master object, then draw frame around index `frame_idx`
        """
        for n in range(self.n_demo):
            frame_config = self.frame_mat_all_demo[n, time_idx, frame_idx]  # (d, d+1)
            self.frame_master_global[n] = draw_frame_3d(
                frame_config, label=f"{self.c.name}_frame_{n}", scale=self.c.frame_scale, radius=self.c.frame_radius,
                alpha=self.c.frame_alpha, collections=self.frame_master_global[n], enabled=enabled
            )

    def set_hierarchical_lvl(self, lvl: int):
        self.c.hierarchical_lvl = lvl

    def find_nearest_local_frame(self, slave_name, keypoint_idx, timestep):
        N_pcl = self.d.appended_obj_traj[slave_name][:, timestep, :, keypoint_idx]  # N * P_master * d
        mean_dist = np.linalg.norm(N_pcl, axis=-1)
        # ic(mean_dist.shape)
        mean_dist = np.average(mean_dist, axis=0)
        # ic(mean_dist.shape)
        frame_idx = np.argmin(mean_dist)
        # ic(frame_idx)
        return frame_idx
