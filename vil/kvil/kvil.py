import numpy as np
import polyscope as ps
import polyscope.imgui as psim
import matplotlib.pyplot as plt

from pathlib import Path

import scipy.stats as st
from typing import List, Union, Dict
from sklearn.cluster import AgglomerativeClustering

from robot_utils import console
import robot_utils.viz.latex_colors_rgba as colors
from robot_utils.py.filesystem import create_path, get_ordered_files, get_ordered_subdirs, validate_path, \
    get_validate_path
from robot_utils.py.interact import ask_checkbox, box_message, ask_binary, user_warning
from robot_utils.py.utils import dump_data_to_yaml, load_dataclass
from robot_utils.math.viz.polyscope import draw_frame_3d
from robot_utils.math.transformations import quaternion_average

from vil.mp import VMP
from vil.mp import TVMP
from vil.obj.real import RealObject
from vil.cfg.hand_group import HandGroup, HandGroupConfig
from vil.cfg.preprocessing import CanonicalSpace
from vil.cfg import KVILConfig, KVILSceneConfig, UIConfig, MasterSlaveHierarchy, ObjectConfig, Constraints, priorities
from vil.constraints.pme import mapping, hdmde, pme_func
from vil.constraints.curvature import get_curvature_of_curve
from vil.hierarchy.functions import symmetric_detection, grasp_detection, is_global_static


class KVIL:
    def __init__(
            self,
            path: Union[str, Path],
            n_demos: int = 3,
            append: str = "",
            force_redo: bool = False,
            reload: bool = False,
            viz_debug: bool = False,
            enable_hand_group: bool = False,
            delete_previous_data: bool = False,
    ):
        """
        Given the path, read scene configuration, config the analyzer properly according to dimension (dim) of the task

        Args:
            path: path to the demonstration root
            n_demos: number of demos to use
            force_redo: force the algorithm to recompute everything
            reload: to reload the constraints stored from previous try
            viz_debug: allow visualization of intermediate results
            enable_hand_group: generate hand group configs
            delete_previous_data: set to true to delete previous data
        """
        self.viz_debug = viz_debug
        self.force_redo = force_redo
        self.reload_flag = reload
        self.enable_hand_group = enable_hand_group
        self.delete_previous_data = delete_previous_data
        self.n_demos = n_demos
        self.multi_moving_master = False
        self.namespace = "kvil"

        self.path = get_validate_path(path) / self.namespace
        validate_path(self.path, throw_error=True, message="you have to guarantee the path include namespace kvil")

        self.proc_path_postfix = f"_{append}" if append else ""

        # Loading K-VIL config and canonical space config
        self.c = load_dataclass(KVILConfig, self.path / "config" / "_kvil_config.yaml")
        self.canonical_space = load_dataclass(CanonicalSpace, self.path.parent / "canonical/dcn/canonical_cfg.yaml")

        self.ui = UIConfig()

        # init constraints
        self.constraints: Dict[int, Constraints] = {}
        self.constraints_index: List[int] = []

        self._load_scene_config()
        self._load_objects()
        console.log("[bold]K-VIL Initialized")

    def _load_scene_config(self):
        """
        create or load scene config from 'demo_folder/process/demo_xx/_scene_config.yaml'.
        The scene config is de-serialized into KVILSceneConfig type, including
        - obj
        - demo_paths
        - scale
        """
        self.proc_path = create_path(self.path / "process" / f"demo_{self.n_demos:>02d}{self.proc_path_postfix}")

        flag_remove, flag_auto = False, False
        if not self.reload_flag and self.delete_previous_data:
            if ask_binary("delete the following folders, 'constraint, obj_config, obj_data, process'"):
                user_warning("Removing 'constraint, obj_config, obj_data, process'")
                flag_remove, flag_auto = True, True

        self.p_constraints = create_path(self.proc_path / "constraints", flag_remove, flag_auto)
        self.p_obj_config = create_path(self.proc_path / "obj_config", flag_remove, flag_auto)
        self.p_obj_data = create_path(self.proc_path / "obj_data", flag_remove, flag_auto)
        self.p_process = create_path(self.proc_path / "process", self.force_redo, flag_auto)

        scene_config_file = self.proc_path / "_scene_config.yaml"
        if self.force_redo or not scene_config_file.is_file():
            # create a scene config file
            recordings = [p.stem for p in get_ordered_subdirs(self.path / "data")]
            while True:
                selected_recordings = ask_checkbox("choose demos", recordings)
                if len(selected_recordings) == self.n_demos:
                    break
                else:
                    console.print(f"you expect {self.n_demos} demos, but select "
                                  f"{len(selected_recordings)} demos, tray again")
            self.scene_cfg = KVILSceneConfig()
            # self.scene_cfg.obj = [p.stem for p in get_ordered_subdirs(self.path / "ref_images")]
            self.scene_cfg.obj = [p.stem for p in get_ordered_files(self.path / "canonical")]
            self.scene_cfg.demo_paths = selected_recordings
            dump_data_to_yaml(KVILSceneConfig, self.scene_cfg, scene_config_file)
        else:
            self.scene_cfg = load_dataclass(KVILSceneConfig, scene_config_file)

        ms_file_path = self.proc_path / "master_slave.yaml"
        if not self.reload_flag:
            self.ms = MasterSlaveHierarchy()
        else:
            self.ms = load_dataclass(MasterSlaveHierarchy, ms_file_path, include_unknown=True)
        self.ms.filename = str(ms_file_path)

    def get_objects(self) -> List[RealObject]:
        return self._objects

    def get_object(self, name):
        for obj in self._objects:
            if obj.c.name == name:
                return obj

    def get_object_by_index(self, index: int):
        if index > len(self._objects):
            raise IndexError(f"the number of objects {len(self._objects)}, but got index {index}")
        return self._objects[index]

    def get_real_objects(self, include_hand: bool = False):
        if include_hand:
            return [obj for obj in self._objects if not obj.c.is_virtual]
        return [obj for obj in self._objects if obj.c.is_real_obj()]

    def get_virtual_objects(self):
        return [obj for obj in self._objects if obj.c.is_virtual]

    def get_moving_objects(self, include_hand: bool = False):
        if include_hand:
            return [obj for obj in self._objects if not obj.c.is_static]
        return [obj for obj in self._objects if obj.c.is_real_obj() and not obj.c.is_static]

    def get_static_objects(self):
        return [obj for obj in self._objects if obj.c.is_static]

    def get_hand_objects(self):
        return [obj for obj in self._objects if obj.c.is_hand]

    def _load_objects(self):
        """
        1. create a list of [RealObjects] as canonical shapes,
        2. load trajectory files for each object.
        3. create virtual object
        4. compute initial HMSR
        """
        box_message("[bold blue]Loading objects", "blue")

        # read a list of objects
        self.obj_name_list = self.scene_cfg.obj
        self._n_obj = len(self.obj_name_list)
        self._objects: List[RealObject] = []
        console.log(f"[bold]Real object involved in this task: {self.obj_name_list}")

        # Create objects and load trajectory data from all selected recordings
        for i, obj_name in enumerate(self.obj_name_list):
            obj_cfg_file = self.p_obj_config / f"{obj_name}.yaml"

            if self.force_redo or not obj_cfg_file.is_file():
                c = ObjectConfig().with_name(obj_name).with_index(i)
                c.traj_file_list = [f"{p}/results/{obj_name}.npy" for p in self.scene_cfg.demo_paths]
            else:
                c = load_dataclass(ObjectConfig, obj_cfg_file)

            c.kvil = self.c
            c.force_redo = self.force_redo
            c.viz_debug = self.viz_debug

            obj = RealObject(c.with_path(self.proc_path))

            # detect static object (MS level = 0, master)
            if self.force_redo and not c.is_hand:
                global_traj = obj.global_traj  # global_traj.shape = N, T_all, P, dim
                if is_global_static(global_traj):
                    console.log(f"-- {c.name} is global static (fixed)")
                    obj.c.as_master()
                    obj.set_as_master()  # TODO fix set as master

            self._objects.append(obj)

        self._create_virtual_object()

        self.obj_name_list = [obj.c.name for obj in self._objects]
        console.log(f"[bold]All objects considered by K-VIL {self.obj_name_list}")

        console.log('[bold green]Adding moving object as slaves to all static objects')
        if self.force_redo:
            # add slave objects (all motion salient objects) to each fixed object
            moving_obj = self.get_moving_objects(include_hand=False)
            for obj in self.get_static_objects():
                for _obj in moving_obj:
                    obj.append_object(_obj, is_slave=True)

        self._compute_master_slave_hierarchy()

        for obj in self._objects:
            obj.save_obj_config()

        if not self.reload_flag:
            for t in range(self.c.n_time_frames):
                self.ms.t_m_c_order[t] = {}
        console.log("[bold]Load object: Done")

    def _create_virtual_object(self):
        box_message("Creating virtual objects for each moving object", "blue")

        self._virtual_obj_list = []
        num_obj = len(self._objects)
        for obj in self.get_moving_objects():
            obj_name = obj.c.name
            virtual_obj_cfg_file = self.proc_path / f"obj_config/virtual_{obj_name}.yaml"

            # create a new virtual object if necessary
            if self.force_redo or not virtual_obj_cfg_file.is_file():
                c_virtual = ObjectConfig().with_name(obj_name).as_virtual().with_index(num_obj)
                c_virtual.traj_file_list = [f"{p}/results/{obj_name}.npy" for p in self.scene_cfg.demo_paths]
                num_obj += 1
            else:
                c_virtual = load_dataclass(ObjectConfig, virtual_obj_cfg_file)

            c_virtual.kvil = self.c
            c_virtual.force_redo = self.force_redo
            c_virtual.viz_debug = self.viz_debug

            self._virtual_obj_list.append(RealObject(c_virtual.with_path(self.proc_path)))

        self._objects.extend(self._virtual_obj_list)
        console.log(f"[bold]Done (virtual object)")

    def _compute_master_slave_hierarchy(self):
        box_message("computing hybrid master-slave relationship", "blue")
        if self.force_redo:
            # relative saliency -> grasping detection
            real_obj_list = self.get_real_objects(include_hand=False)
            hand_list = self.get_hand_objects()

            console.rule("[bold green]Grasp detection")
            for idx, hand in enumerate(hand_list):
                for obj in real_obj_list:
                    console.log(f"\n[bold green]Detect grasp between [cyan]{hand.c.name} -- {obj.c.name}")

                    if not grasp_detection(hand_traj=hand.global_traj, object_traj=obj.global_traj):
                        console.log('-- [bold]No grasp found')
                        continue

                    obj.set_as_master(remove_slaves=False)
                    obj.append_object(hand, is_slave=True)

                    obj.c.is_grasped = True
                    obj.c.grasping_hand_idx.append(hand.c.index)
                    obj.c.grasping_point_idx[hand.c.index] = obj.find_nearest_local_frame(
                        slave_name=hand.c.name, keypoint_idx=9, timestep=0)
                    console.log(f"-- found grasping point: {obj.c.grasping_point_idx[hand.c.index]}")
                    console.log(f"-- [bold]{hand.c.name} grasps {obj.c.name}")
                    console.log(f"-- [red]Discard other grasps relevant to {hand.c.name}")
                    break
            console.log("[bold]Done (grasp detection)")
            # exit()

            # symmetric detection:
            console.rule("[bold green]Symmetry detection")
            for obj in real_obj_list:
                if len(obj.c.grasping_hand_idx) <= 1:
                    continue

                console.log(f"[bold yellow]multiple hands grasp the object [cyan]{obj.c.name}")
                for i in range(len(obj.c.grasping_hand_idx)):
                    hand_a = self._objects[obj.c.grasping_hand_idx[i]]
                    for j in range(i + 1, len(obj.c.grasping_hand_idx)):
                        hand_b = self._objects[obj.c.grasping_hand_idx[j]]
                        console.log(
                            f"[bold green]Detection symmetric among "
                            f"[cyan]{hand_a.c.name} -- {hand_b.c.name} -- {obj.c.name}")
                        if symmetric_detection(hand_a.global_traj, hand_b.global_traj, obj.global_traj):
                            console.log(
                                f"[bold yellow]-- Found symmetric coordination among "
                                f"[cyan]{hand_a.c.name} -- {hand_b.c.name} -- {obj.c.name}")
                            obj.c.is_symmetric = True
                            obj.c.symmetric_hand_idx.extend([hand_a.c.index, hand_b.c.index])
                        else:
                            console.log(f"[bold yellow]-- This is not a symmetric coordination")
            console.log("[bold]Done (symmetry detection)")

            from vil.hierarchy.variance_criteria import VarianceCriteria
            v = VarianceCriteria(self.get_static_objects(), self.get_moving_objects(), self.force_redo)
            v.run()

            # Note: you need to save data whenever it's necessary to keep the intermediate results
            # Here we have finished all the master-slave detection, then append the non-slave objects to all the objects:

            console.rule("[bold green]Append slave objects")
            for master_obj in self._objects:  # TODO explain
                console.log(f"[bold cyan]\nFor {master_obj.c.name}")
                # if master_obj.c.is_master:
                for to_append_obj in self._objects:
                    if to_append_obj.c.name not in master_obj.c.slave_obj:
                        master_obj.append_object(to_append_obj, is_slave=False)
                # master_obj.d.save("append slaves")

        else:
            if not self.reload:
                ms = load_dataclass(MasterSlaveHierarchy, self.ms.filename, include_unknown=True)
                self.ms.master_obj = ms.master_obj
                self.ms.master_obj_idx = []
                for master in self.ms.master_obj:
                    self.ms.slave_obj_idx[master] = []
                    self.ms.slave_obj[master] = ms.slave_obj[master]
                    for obj in self._objects:
                        if obj.c.name == master:
                            self.ms.master_obj_idx.append(obj.c.index)
                        if obj.c.name in self.ms.slave_obj[master]:
                            self.ms.slave_obj_idx[master].append(obj.c.index)

                for m_obj in self._objects:
                    if m_obj.c.name in self.ms.master_obj:
                        m_obj.set_as_master()
                        for s_obj in self._objects:
                            m_obj.append_object(s_obj, is_slave=s_obj.c.name in self.ms.slave_obj[m_obj.c.name])

                        self.ms.constraints[m_obj.c.name] = {}
                        create_path(self.p_constraints / m_obj.c.name)
                        for slave in m_obj.c.slave_obj:
                            self.ms.constraints[m_obj.c.name][slave] = []

            for obj in self._objects:
                obj.d.load()

        console.rule("[bold green]set object hierarchical level")  # TODO explain the level number
        obj_master = {}
        for slave in self._objects:
            obj_master[slave] = []
            for master in self._objects:
                if slave.c.name in master.c.slave_obj:
                    obj_master[slave].append(master)

        # to fix: probabily while ture here

        exist_obj_without_hierachical_lvl = True
        while exist_obj_without_hierachical_lvl:
            for slave in self._objects:
                if len(obj_master[slave]) > 0:
                    master_hierachical_lvl = []
                    for master in obj_master[slave]:
                        master_hierachical_lvl.append(master.c.hierarchical_lvl)
                    if not (-1) in master_hierachical_lvl:
                        slave.set_hierarchical_lvl(max(master_hierachical_lvl) + 1)
                else:
                    if slave.c.hierarchical_lvl == -1:
                        slave.set_hierarchical_lvl(1)
            obj_lvl = []
            for obj in self._objects:
                obj_lvl.append(obj.c.hierarchical_lvl)
            if not (-1) in obj_lvl:
                exist_obj_without_hierachical_lvl = False

        console.log("[bold]Done (hierarchy)")

        # Note: below is general code
        if not self.reload_flag:
            for obj in self._objects:
                if not obj.is_master():
                    continue
                self.ms.master_obj.append(obj.c.name)
                self.ms.master_obj_idx.append(obj.c.index)
                self.ms.slave_obj[obj.c.name] = obj.c.slave_obj
                self.ms.slave_obj_idx[obj.c.name] = obj.c.slave_idx
                self.ms.constraints[obj.c.name] = {}
                create_path(self.p_constraints / obj.c.name)
                for slave in obj.c.slave_obj:
                    self.ms.constraints[obj.c.name][slave] = []
        # self.ms.show()
        self.ms.save()
        console.log("[bold]HMSR: done")

    def distance_criteria(self):
        console.rule(f"[bold cyan]Distance Criteria")
        for i in self.ms.master_obj_idx:
            master = self._objects[i]  # type: RealObject
            c, d = master.c, master.d
            for idx_slave in c.slave_idx:
                slave = self._objects[idx_slave]
                console.print(f"[blue]distance criteria between {c.name} -- {slave.c.name}")
                # not enough number of demos, force to use only last time step
                d.mask[slave.c.name][:-1] = True

                slave_traj = d.appended_obj_traj[slave.c.name].squeeze(0)                   # (T, Pm, Ps, d), N=1
                distance = np.linalg.norm(slave_traj, axis=-1)                              # (T, Pm, Ps)
                distance_masked = np.ma.masked_array(distance)
                distance_masked.mask = d.mask[slave.c.name]

                idx_min_dist = np.ma.where(distance_masked == distance_masked.min())
                idx_frame = idx_min_dist[1][0]
                idx_frame_to_mask_out = np.concatenate((
                    np.arange(0, idx_frame), np.arange(idx_frame + 1, master.n_candidates_kvil)
                ))
                distance_masked.mask[:, idx_frame_to_mask_out] = True

                # Note: simplified version. However, it might be useful for soft object to use clustering here as well
                idx_max_dist = np.ma.where(distance_masked == distance_masked.max())
                # find min and max distance
                self.create_constraint("p2p", master, slave, idx_min_dist[0][0], idx_min_dist[1][0], idx_min_dist[2][0])
                self.create_constraint("p2p", master, slave, idx_max_dist[0][0], idx_max_dist[1][0], idx_max_dist[2][0])
                # find the max distance in the remaining smallest 60%
                dist_compressed = distance_masked.compressed()
                idx_large_dist = np.ma.where(distance_masked > np.percentile(dist_compressed, 40))
                distance_masked.mask[idx_large_dist] = True
                idx_max_dist = np.ma.where(distance_masked == distance_masked.max())
                self.create_constraint("p2p", master, slave, idx_max_dist[0][0], idx_max_dist[1][0], idx_max_dist[2][0])
        console.log("[bold]Done (distance criteria)")

    def linear_constraints(self):
        console.rule(f"[bold cyan]Linear Constraints")
        for i in self.ms.master_obj_idx:
            master = self._objects[i]
            master.pca()
            c, d = master.c, master.d
            for idx_slave in c.slave_idx:
                slave = self._objects[idx_slave]
                if slave.c.is_hand:
                    continue

                console.rule(f"[bold blue]constraint between M: {master.c.name} -- S: {slave.c.name} -- {slave.c.spatial_scale}")
                variability = np.sqrt(d.pca_data[slave.c.name][..., -1, :]) / slave.c.spatial_scale  # (T, Pm, Ps, d)

                variability_norm = np.linalg.norm(variability, axis=-1)  # (T, Pm, Ps)
                if c.viz_debug:
                    console.print(f"variability max: {variability_norm.max()}, min: {variability_norm.min()}")
                    fig = plt.figure(figsize=(12, 6))
                    ax = fig.add_subplot(111)
                    ax.hist(variability_norm.flatten(), bins=100, density=True)
                    plt.title("before p2p: variability norm (determine th_vari_low)")
                    console.log("[bold yellow]before p2p: variability norm (determine th_vari_low)")
                    plt.show()

                idx_low_variability = np.where(variability_norm < c.kvil.th_vari_low)
                d.mask[slave.c.name][idx_low_variability] = True  # set to True to exclude them for later steps

                if len(idx_low_variability[0]) > 0:
                    console.rule(f"[bold green] extracting p2p constraint -- "
                                 f"{len(idx_low_variability[0])} candidates")

                    if c.viz_debug:
                        from robot_utils.py.visualize import set_3d_ax_label
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection="3d")
                        ax.scatter3D(idx_low_variability[0], idx_low_variability[1], idx_low_variability[2])
                        set_3d_ax_label(ax, ["T", "Pm", "Ps"])
                        plt.title("before p2p: idx of points with low variability norm")
                        console.log("before p2p: idx of points with low variability norm")
                        plt.show()

                    self.filter_candidate(
                        idx_low_variability, variability_norm[idx_low_variability], master, slave, "p2p"
                    )
                else:
                    console.log(f"[bold]No p2p constraint found")

                variability_masked = np.ma.array(variability)
                variability_masked.mask = np.zeros_like(variability, dtype=bool)  # (T, Pm, Ps, d)
                for idx_dim in range(1, master.dim):
                    # allow p2P when N >=4, and allow p2l when N >= 3
                    if (idx_dim == 2 and master.n_demo <= 4) or (idx_dim == 1 and master.n_demo <= 3):
                        break

                    for idx_dim_ in range(master.dim):
                        variability_masked.mask[..., idx_dim_] = d.mask[slave.c.name]  # (T, P_master, P_slave)

                    if idx_dim == 1:
                        constraint_type = "p2l"
                        pm_linear_idx = np.ma.where(  # variability_masked (T, Pm, Ps, d)
                            (variability_masked[..., idx_dim] < c.kvil.th_linear_lv) &
                            # (variability_masked[..., 0] > c.kvil.principal_var_ratio * variability_masked[..., idx_dim]) &
                            (variability_masked[..., 0] > c.kvil.th_linear_hv)
                        )
                        value_array = np.abs(variability[..., 0][pm_linear_idx])
                        # ic(c.kvil.th_linear_hv, c.kvil.th_linear_lv, value_array)

                    else:
                        constraint_type = "p2P"

                        pm_linear_idx = np.ma.where(
                            # (variability_masked[..., 0] > c.kvil.principal_var_ratio * variability_masked[..., idx_dim])
                            (variability_masked[..., 0] > c.kvil.th_linear_hv)
                            # & (variability_masked[..., 1] > c.kvil.principal_var_ratio * variability_masked[..., idx_dim])
                            & (variability_masked[..., 1] > c.kvil.th_linear_hv)
                            & (variability_masked[..., 2] < c.kvil.th_linear_lv)
                            & (variability_masked[..., 1] / variability_masked[..., 0] > 0.3)
                        )
                        value_array = np.linalg.norm(variability[..., :2][pm_linear_idx], axis=-1)

                    if self.viz_debug:
                        console.log(
                            f"variability max: {variability_masked[1].max()}, min: {variability_masked[1].min()}")
                        fig2 = plt.figure(figsize=(12, 6))
                        ax = plt.Axes(fig2, [0., 0., 1., 1.])
                        ax.set_axis_off()
                        fig2.add_axes(ax)

                        idx = np.where(np.invert(variability_masked.mask))
                        console.log(
                            f"[bold yellow]before {constraint_type}: {np.size(idx[0])} points left from previous step")
                        console.log(
                            f"[bold yellow]before {constraint_type}: variability_masked: {pm_linear_idx[0].shape} candidates")

                        ax0 = fig2.add_subplot(311)
                        plt.title(f"before {constraint_type}: variability_masked[..., 0/1/2], for th_linear_lv/hv")
                        ax0.hist(variability_masked[..., 0].flatten(), bins=100, density=True)
                        ax1 = fig2.add_subplot(312)
                        ax1.hist(variability_masked[..., 1].flatten(), bins=100, density=True)
                        ax2 = fig2.add_subplot(313)
                        ax2.hist(variability_masked[..., 2].flatten(), bins=100, density=True)

                        plt.show()

                        # from robot_utils.py.visualize import set_3d_equal_auto
                        # idx = np.where(variability_masked.mask[..., 0][1, 0] == 0)[0]
                        # idx_ = np.unique(pm_linear_idx[2])
                        # ic(idx_)
                        # sampled_kpts_coordinate = slave.coordinates
                        # selected_sampled_kpts_coordinate = slave.coordinates[idx]
                        # kpts_coordinates_in_this_cluster = slave.coordinates[idx_]
                        # ax = plt.figure().add_subplot(1, 1, 1, projection='3d')
                        # ax.scatter3D(sampled_kpts_coordinate[:, 0], sampled_kpts_coordinate[:, 1],
                        #              sampled_kpts_coordinate[:, 2],
                        #              alpha=1, c='k', label="sampled points")
                        # ax.scatter3D(selected_sampled_kpts_coordinate[:, 0], selected_sampled_kpts_coordinate[:, 1],
                        #              selected_sampled_kpts_coordinate[:, 2],
                        #              marker='x', s=100, c='b', label=f"points on {slave.c.name}")
                        # ax.scatter3D(kpts_coordinates_in_this_cluster[:, 0], kpts_coordinates_in_this_cluster[:, 1],
                        #              kpts_coordinates_in_this_cluster[:, 2],
                        #              s=80, color="yellow", alpha=0.8, label="points in this cluster")
                        #
                        # set_3d_equal_auto(ax)
                        # ax.set_axis_off()
                        # plt.legend()
                        # plt.show()

                    if len(pm_linear_idx[0]) == 0:
                        console.log(f"[bold]No {constraint_type} constraint found")
                        continue
                    console.rule(f"[bold green]extracting {constraint_type} constraint -- "
                                 f"{slave.c.name}: {len(pm_linear_idx[0])} candidates")
                    self.filter_candidate(pm_linear_idx, value_array, master, slave, constraint_type)

                    if c.viz_debug:
                        from robot_utils.py.visualize import set_3d_ax_label
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection="3d")
                        ax.scatter3D(pm_linear_idx[0], pm_linear_idx[1], pm_linear_idx[2])
                        set_3d_ax_label(ax, ["T", "Pm", "Ps"])
                        plt.show()
                        # exit()

                # idx = np.where(np.invert(d.mask[slave.c.name]))
                idx_to_be_excluded = np.ma.where(np.abs(variability[..., 1]) < c.kvil.th_linear_lv)
                d.mask[slave.c.name][idx_to_be_excluded] = True
                # idx = np.where(np.invert(d.mask[slave.c.name]))
        console.log("[bold]Done (linear constraints)")

    def nonlinear_constraint(self):
        """ extract nonlinear constraints, e.g. p2c, pcS using PME """
        console.rule(f"[bold cyan]Non-Linear Constraints")
        for i in self.ms.master_obj_idx:
            master = self._objects[i]
            if master.c.is_virtual:
                continue
            c, d = master.c, master.d
            for idx_slave in c.slave_idx:
                slave = self._objects[idx_slave]
                if slave.c.is_hand:
                    continue

                # Note: Only allow last time step, as the PME algorithm is not as efficient as PCA
                # ic(master.d.mask[slave.c.name].shape)
                idx = np.where(np.invert(master.d.mask[slave.c.name]))
                # ic(idx[0].shape)
                master.d.mask[slave.c.name][:-1] = True
                idx_to_be_handled = np.where(np.invert(master.d.mask[slave.c.name]))
                if len(idx_to_be_handled[0]) == 0:
                    console.log("[red]No candidate anymore, continue")
                    continue

                variability = np.sqrt(master.d.pca_data[slave.c.name][..., -1, :]) / slave.c.spatial_scale
                variability = variability[idx_to_be_handled] / slave.c.spatial_scale
                # ic(variability.shape)
                self.filter_candidate(idx_to_be_handled, variability, master, slave, constraint_type='p2c')

                to_be_removed = []
                for c_candidate in self.ms.constraints[master.c.name][slave.c.name]:
                    if c_candidate.type == "p2c":
                        pm_intrinsic_dim = 1
                        # ic(master.d.appended_obj_traj[slave.c.name].shape)
                        neighbor_idx = self._objects[c_candidate.idx_slave].neighbor_idx[c_candidate.idx_keypoint, :5]
                        points = master.d.appended_obj_traj[slave.c.name][:, c_candidate.idx_time,
                                 c_candidate.idx_frame, neighbor_idx]  # (N, d)  (N, 5, d)
                        # points = points.reshape(-1, points.shape[-1])
                        points = points.mean(axis=1)

                        sol_opt_x, sol_opt_t, t_opt, std_stress, std_projection, embedding, scale, w, mu, sig, proj, stress_vectors = \
                            pme_func(points.copy(), intrinsic_dim=1)

                        samples = 500
                        obj_scale = self._objects[c_candidate.idx_slave].c.spatial_scale
                        t_test = np.linspace(-0.8 * np.abs(np.min(embedding)), 0.8 * np.abs(np.max(embedding)),
                                             samples) / obj_scale
                        x_test = mapping(t_test, d=pm_intrinsic_dim, sol_opt_x=sol_opt_x, sol_opt_t=sol_opt_t,
                                         t_opt=t_opt)
                        curvature = get_curvature_of_curve(x_test)
                        console.log(f"[bold green]the curvature of {c_candidate.idx_c}-th constraints is [bold cyan]{curvature.max():>3.2f}")
                        if curvature.max() > self.c.th_curvature_hv or curvature.max() < self.c.th_curvature_lv:
                            console.log(f"[bold red]curvature too large [bold cyan]{curvature.max():>3.2f} "
                                        f"[bold red]removing {c_candidate.idx_c}-th constraint")
                            to_be_removed.append(c_candidate)
                            continue

                        c_candidate.pm_sol_opt_x = sol_opt_x.tolist()
                        c_candidate.pm_sol_opt_t = sol_opt_t.tolist()
                        c_candidate.pm_embedding = embedding.tolist()
                        c_candidate.pm_t_opt = t_opt.tolist()
                        c_candidate.pm_std_stress = std_stress
                        c_candidate.pm_std_projection = std_projection
                        c_candidate.pm_projection = proj.tolist()
                        c_candidate.pm_stress_vectors = stress_vectors.tolist()
                        c_candidate.pm_intrinsic_dim = pm_intrinsic_dim
                        c_candidate.pm_scaling = scale
                        c_candidate.density_mu = mu.tolist()
                        c_candidate.density_weights = w.tolist()
                        c_candidate.density_std_dev = sig
                        center_point = mapping(mu, 1, sol_opt_x, sol_opt_t, t_opt) * scale
                        c_candidate.pm_mean = center_point.flatten().tolist()

                for c_ in to_be_removed:
                    self.ms.constraints[master.c.name][slave.c.name].remove(c_)
        console.log("[bold]Done (non-linear constraints)")

    def filter_candidate(
            self,
            idx_tuple,
            value_array: np.ndarray,
            master: RealObject,
            slave: RealObject,
            constraint_type: str,
    ):
        """
        Filtering out candidates based on time and position clustering

        Args:
            idx_tuple: Tuple of indices for (T, P_master, P_slave), each is a np.ndarray
            value_array: for p2p: this is the variance norm of all points corresponds to idx_tuple
                for p2l/P/c/S this is the absolute variance in certain direction
            master: the master object
            slave: the slave object
            constraint_type:
        """
        if slave.c.is_hand:
            return

        if constraint_type == "p2d":
            # TODO add GMM based method
            #  considering all candidate pairs (n_pair, [time, normalized_can_coord_master, normalized_can_coord_slave])
            #  compute the likelihood. maybe can also build the likelihood of each can_shape using marginal likelihood
            idx_t = idx_tuple[0] / (self.c.n_time_frames - 1)
            can_coord_master = master.coordinates / master.c.spatial_scale
            can_coord_slave = slave.coordinates / slave.c.spatial_scale
            can_coord_master -= can_coord_master.mean(0)
            can_coord_slave -= can_coord_slave.mean(0)

            offset = np.array([1.0, 0.0, 0.0])
            ps.register_point_cloud("can shape_master", can_coord_master, enabled=True, radius=0.01,
                                    color=colors.smoky_black, transparency=0.4, point_render_mode="quad")
            ps.register_point_cloud("candidates_master", can_coord_master[idx_tuple[1]], enabled=True, radius=0.01,
                                    color=colors.green_ryb, transparency=0.8)
            ps.register_point_cloud("can shape_slave", can_coord_slave + offset, enabled=True, radius=0.01,
                                    color=colors.smoky_black, transparency=0.4, point_render_mode="quad")
            ps.register_point_cloud("candidates_slave", can_coord_slave[idx_tuple[2]] + offset, enabled=True,
                                    radius=0.01, color=colors.green_ryb, transparency=0.8)

            data = np.concatenate((
                idx_t.reshape(-1, 1), can_coord_master[idx_tuple[1]], can_coord_slave[idx_tuple[2]]
            ), axis=-1)
            ic(data.shape)
            ic(data.min(axis=0), data.max(axis=0))
            model = AgglomerativeClustering(distance_threshold=slave.c.kvil.th_dist_ratio,
                                            linkage='average', n_clusters=None)
            kpts_clusters = model.fit_predict(data)
            n_clusters = model.n_clusters_
            ic(kpts_clusters, n_clusters)

            demo_colors = plt.get_cmap("plasma")(np.linspace(0, 1, n_clusters))
            for n in range(n_clusters):
                cluster_n_idx = np.where(kpts_clusters == n)[0]
                kpts_idx_of_cluster_n_master = idx_tuple[1][cluster_n_idx]
                kpts_idx_of_cluster_n_slave = idx_tuple[2][cluster_n_idx]
                kpts_idx_of_cluster_n_master = np.unique(kpts_idx_of_cluster_n_master)
                kpts_idx_of_cluster_n_slave = np.unique(kpts_idx_of_cluster_n_slave)
                kpts_master = can_coord_master[kpts_idx_of_cluster_n_master]
                kpts_slave = can_coord_slave[kpts_idx_of_cluster_n_slave]
                # ps.register_point_cloud(f"cluster-{n}_master", kpts_master, enabled=True, radius=0.015, color=demo_colors[n], transparency=0.8)
                # ps.register_point_cloud(f"cluster-{n}_slave", kpts_slave + offset, enabled=True, radius=0.015, color=demo_colors[n], transparency=0.8)
                ps.register_point_cloud(f"cluster-{n}", np.concatenate((kpts_master, kpts_slave + offset)),
                                        enabled=True, radius=0.015, color=demo_colors[n], transparency=0.8)

            ps.show()
            ps.remove_all_structures()

            # exit()

        console.log(f"[purple]1) time clustering")

        unique_t = (np.unique(idx_tuple[0]) / (self.c.n_time_frames - 1))
        # ic(unique_t)
        split_idx = np.where((unique_t[1:] - unique_t[:-1]) >= master.c.kvil.th_time_cluster)[0] + 1
        split_idx = [0, ] + split_idx.tolist() + [unique_t.shape[0], ]
        time_clusters = [unique_t[np.arange(split_idx[j], split_idx[j + 1])] for j in range(len(split_idx) - 1)]

        for i_tc, tc in enumerate(time_clusters):
            tc = (tc * (self.c.n_time_frames - 1)).astype(int)
            console.log(f"[yellow]-- in the {i_tc + 1}-th of {len(time_clusters)} time clusters: time idx: {tc}")

            # Note that element_idx is the index of the index of the key points, it is used to mask the idx_tuple
            #  we then get the actual kpt index and cluster them according to their coordinates on each primitive object
            element_idx_in_this_t_cluster = np.concatenate([np.where(idx_tuple[0] == t)[0] for t in tc])
            idx_tuple_at_tc = (
                idx_tuple[0][element_idx_in_this_t_cluster],
                idx_tuple[1][element_idx_in_this_t_cluster],
                idx_tuple[2][element_idx_in_this_t_cluster]
            )
            unique_kpts_idx_at_tc = np.unique(idx_tuple[2][element_idx_in_this_t_cluster])
            if master.slave_variation.get(slave.c.name, None):
                if not master.slave_variation[slave.c.name].get(max(tc), True):
                    console.log(f"[yellow]-- -- object {slave.c.name} no variation at time {max(tc)}, skip ...")
                    continue
            console.log(f"[purple]-- 2) coordinate clustering for {slave.c.name}")

            ratio = len(unique_kpts_idx_at_tc) / slave.n_candidates_kvil
            console.log(
                f"[yellow]-- -- the ratio = num(unique kpts) / num(total kpts on {slave.c.name}) = {ratio * 100:>2.2f}%")
            if constraint_type == "p2p" and ratio > 0.9:
                if master.slave_variation.get(slave.c.name, None) is None:
                    master.slave_variation[slave.c.name] = {}
                master.slave_variation[slave.c.name][max(tc)] = False
                console.log(
                    f"[yellow]-- -- ratio = {ratio:>.2f}. object {slave.c.name} doesn't have any variation, skip")
                continue

            # Note: fetch coordinates on the canonical shape of slave and fit the clustering model
            selected_sampled_kpts_coordinate = slave.coordinates[unique_kpts_idx_at_tc]
            if unique_kpts_idx_at_tc.shape[0] == 1:
                n_clusters = 1
                kpts_clusters = np.array([0])
            else:
                # Note: initialize the clustering algo based on the spatial scale of the slave object
                model = AgglomerativeClustering(distance_threshold=slave.c.spatial_scale * slave.c.kvil.th_dist_ratio,
                                                linkage='average', n_clusters=None)
                kpts_clusters = model.fit_predict(selected_sampled_kpts_coordinate)
                n_clusters = model.n_clusters_

            console.log(f"[yellow]-- -- num of clusters on canonical shape of {slave.c.name}: {n_clusters}, "
                        f"from {unique_kpts_idx_at_tc.shape[0]} unique kpts idx and "
                        f"in totol {selected_sampled_kpts_coordinate.shape[0]} candidates")

            for c in range(n_clusters):
                console.log(f"[yellow]-- -- {c + 1}-th of {n_clusters} coordinate clusters on {slave.c.name}")
                unique_kpts_idx_in_this_shape_cluster = unique_kpts_idx_at_tc[np.where(kpts_clusters == c)[0]]

                # Note new
                unique_kpts_min_dist_to_their_origin = np.zeros(len(unique_kpts_idx_in_this_shape_cluster))
                value_array_min = np.zeros_like(unique_kpts_min_dist_to_their_origin)
                value_array_max = np.zeros_like(unique_kpts_min_dist_to_their_origin)
                value_array_argmin = []
                value_array_argmax = []
                unique_kpts_all_dist_to_their_origin_list = []
                list_of_element_idx_of_all_kpt_in_this_cluster = []  # type: List[np.ndarray]

                # ic(value_array.shape)  # for p2p (n, ),    for p2l, p2P (T, Pm, Ps, d)
                total = 0
                for i_, t_ in enumerate(unique_kpts_idx_in_this_shape_cluster):
                    list_of_element_idx_of_one_kpt_in_this_cluster = np.where(idx_tuple_at_tc[2] == t_)[0]
                    n_kpts = len(list_of_element_idx_of_one_kpt_in_this_cluster)
                    total += n_kpts
                    # console.log(f"{n_kpts} candidates has index {t_} accumulated {total} / {len(idx_tuple[2])}")
                    if n_kpts == 0:
                        continue

                    list_of_element_idx_of_all_kpt_in_this_cluster.append(
                        list_of_element_idx_of_one_kpt_in_this_cluster)
                    # time_idx_ = idx_tuple[0][list_of_element_idx_of_one_kpt_in_this_cluster] + (self.total_time_steps - 1 if assign_T else 0.0)  # TODO check this in new setup
                    time_idx_ = idx_tuple_at_tc[0][list_of_element_idx_of_one_kpt_in_this_cluster]
                    frame_idx_ = idx_tuple_at_tc[1][list_of_element_idx_of_one_kpt_in_this_cluster]
                    kpt_idx_ = idx_tuple_at_tc[2][list_of_element_idx_of_one_kpt_in_this_cluster]
                    kpt_coord_in_corresponding_local_frames = master.d.appended_obj_traj[slave.c.name][
                        # (N, T, Pm, Ps, d)
                        (slice(None), time_idx_, frame_idx_, kpt_idx_)
                    ].transpose(1, 0, 2)  # (n_kpts, N, d)

                    kpt_dist_to_their_origins = np.linalg.norm(kpt_coord_in_corresponding_local_frames, axis=-1).mean(
                        axis=-1)  # (n_kpts, )
                    unique_kpts_all_dist_to_their_origin_list.append(kpt_dist_to_their_origins)
                    unique_kpts_min_dist_to_their_origin[i_] = kpt_dist_to_their_origins.min()

                    value_array_chunk = value_array[list_of_element_idx_of_one_kpt_in_this_cluster]
                    value_array_min[i_] = value_array_chunk.min()
                    value_array_max[i_] = value_array_chunk.max()
                    value_array_argmin.append(value_array_chunk.argmin())
                    value_array_argmax.append(value_array_chunk.argmax())

                # Note: sort the unique kpts in this can-shape cluster according to distance to local frame and value
                sort_dist = unique_kpts_min_dist_to_their_origin.argsort()  # len = len(unique_kpts_idx_in_this_cluster)
                ranks_dist = np.empty_like(sort_dist)
                ranks_dist[sort_dist] = np.arange(len(sort_dist))
                temp_min, temp_max = unique_kpts_min_dist_to_their_origin.min(), unique_kpts_min_dist_to_their_origin.max()
                score_dist = unique_kpts_min_dist_to_their_origin - temp_min if temp_max == temp_min else \
                    (unique_kpts_min_dist_to_their_origin - temp_min) / (temp_max - temp_min)

                sort_value = value_array_min.argsort()  # len = len(unique_kpts_idx_in_this_cluster)
                # ic(sort_value.shape)
                ranks_value = np.empty_like(sort_value)
                ranks_value[sort_value] = np.arange(len(sort_value))
                temp_min, temp_max = value_array_min.min(), value_array_min.max()
                score_value = value_array_min - temp_min if temp_max == temp_min else \
                    (value_array_min - temp_min) / (temp_max - temp_min)

                temp_min, temp_max = value_array_max.min(), value_array_max.max()
                score_value_max = temp_max - value_array_max if temp_max == temp_min else \
                    (temp_max - value_array_max) / (temp_max - temp_min)

                # ic(score_dist, score_value, score_value + score_dist, ranks_dist, ranks_value, ranks_dist+ranks_value)
                # ic(list_of_element_idx_of_all_kpt_in_this_cluster)

                masked_time_step = tc[:, None]  # has to expand -1 dim in order to use it in multi-axis indexing

                # Note: find points on the slave can shape, that are close to the kpts_in_this_cluster, and exclude them for later steps
                candidate_coords = slave.coordinates[
                    unique_kpts_idx_in_this_shape_cluster]  # (n_kpts_in_this_cluster, 3)
                dist = np.linalg.norm(
                    np.expand_dims(candidate_coords, 1) - np.expand_dims(slave.coordinates, 0),
                    axis=-1)  # (Ps, n_kpts_in_this_cluster)
                excluded_idx = np.unique(
                    np.where(dist < 0.1 * slave.c.spatial_scale)[1]
                )
                master.d.mask[slave.c.name][masked_time_step, :, excluded_idx] = True

                if constraint_type == "p2p":
                    selected_kpts_group = (score_value + score_dist).argmin()
                    # ic(score_value.shape, selected_kpts_group, len(list_of_element_idx_of_all_kpt_in_this_cluster))
                    selected_ele_idx = list_of_element_idx_of_all_kpt_in_this_cluster[selected_kpts_group][
                        unique_kpts_all_dist_to_their_origin_list[selected_kpts_group].argmin()
                    ]
                else:
                    if constraint_type in ["p2l", "p2P"]:
                        selected_kpts_group = (score_value_max + score_dist).argmin()
                    elif constraint_type in ["p2c", "p2S"]:
                        # selected_kpts_group = (score_value_max + score_dist).argmin()
                        selected_kpts_group = score_value_max.argmin()
                    else:
                        raise NotImplementedError
                    ele_index_ = list_of_element_idx_of_all_kpt_in_this_cluster[selected_kpts_group]
                    available_frame_idx = idx_tuple_at_tc[1][ele_index_]
                    # loop over all constraints assigned to this master-slave pair
                    for c_ in self.ms.constraints[master.c.name][slave.c.name]:
                        if c_.idx_master != master.c.index or c_.idx_slave != slave.c.index:
                            console.log(f"[bold red]constraints and master/slave object mismatch")
                            exit(1)
                        if c_.type == "p2p":
                            index_ = np.where(available_frame_idx == c_.idx_frame)[0]
                            # console.log(f"[yellow]we have a p2p {c_.idx_frame}, {available_frame_idx}")
                            if len(index_) > 0:
                                selected_ele_idx = ele_index_[index_][0]
                                break
                    else:
                        selected_ele_idx = ele_index_[
                            unique_kpts_all_dist_to_their_origin_list[selected_kpts_group].argmin()
                        ]

                selected_time_index = idx_tuple_at_tc[0][selected_ele_idx]
                if self.c.last_timestep:
                    if not (selected_time_index == self.c.n_time_frames - 1):
                        continue
                selected_frame_index = idx_tuple_at_tc[1][selected_ele_idx]
                selected_kpt_index = idx_tuple_at_tc[2][selected_ele_idx]
                self.create_constraint(
                    constraint_type, master, slave, selected_time_index, selected_frame_index, selected_kpt_index
                )

                # Note: visualize
                if self.viz_debug:
                    console.rule("")
                    console.log(
                        f"[bold yellow]{constraint_type}: {i_tc}-th time at time {selected_time_index} | {c}/{n_clusters}-th shape cluster")
                    can_shape_slave = slave.coordinates
                    can_shape_master = master.coordinates
                    # viz_idx = excluded_idx
                    viz_idx_slave = unique_kpts_idx_in_this_shape_cluster
                    viz_idx_master = idx_tuple_at_tc[1][
                        np.unique(np.concatenate(list_of_element_idx_of_all_kpt_in_this_cluster))]
                    kpts_coordinates_in_this_cluster_slave = can_shape_slave[viz_idx_slave]
                    kpts_coordinates_in_this_cluster_master = can_shape_master[viz_idx_master]
                    # selected_sampled_kpts_coordinate_master = master.coordinates[unique_kpts_idx_at_tc]
                    if master.dim == 2:
                        from robot_utils.py.visualize import draw_frame_2d, set_2d_equal_auto
                        # ic(excluded_idx)
                        ax = plt.figure().add_subplot(1, 1, 1)
                        ax.scatter(can_shape_slave[:, 0], can_shape_slave[:, 1],
                                   alpha=0.5, c='k', label="sampled points")
                        ax.scatter(selected_sampled_kpts_coordinate[:, 0], selected_sampled_kpts_coordinate[:, 1],
                                   marker='x', s=100, c='b', label=f"points on {slave.c.name}")
                        ax.scatter(kpts_coordinates_in_this_cluster_slave[:, 0],
                                   kpts_coordinates_in_this_cluster_slave[:, 1],
                                   s=80, color="yellow", alpha=0.8, label="points in this cluster")
                        ax.set_title(f"{c + 1}-th result of {n_clusters} clusters for {constraint_type}")
                        set_2d_equal_auto(ax)
                        ax.set_axis_off()
                        plt.legend()
                        plt.show()
                    else:
                        console.log(f"{c + 1}-th result of {n_clusters} clusters for {constraint_type}")
                        offset = np.array([1.0, 0.0, 0.0]) * master.c.spatial_scale
                        ps.register_point_cloud("canonical shape slave", can_shape_slave + offset, enabled=True,
                                                radius=0.005, color=colors.smoky_black, transparency=0.4)
                        ps.register_point_cloud("canonical shape master", can_shape_master, enabled=True, radius=0.005,
                                                color=colors.smoky_black, transparency=0.4)
                        ps.register_point_cloud("selected kpts slave", selected_sampled_kpts_coordinate + offset,
                                                enabled=True, radius=0.01, color=colors.cyan_process, transparency=0.8)
                        # ps.register_point_cloud("selected kpts master", selected_sampled_kpts_coordinate_master, enabled=True, radius=0.01, color=colors.cyan_process, transparency=0.8)
                        ps.register_point_cloud("kpts this cluster slave",
                                                kpts_coordinates_in_this_cluster_slave + offset, enabled=True,
                                                radius=0.015, color=colors.green_yellow, transparency=0.8)
                        ps.register_point_cloud("kpts this cluster master", kpts_coordinates_in_this_cluster_master,
                                                enabled=True, radius=0.015, color=colors.green_yellow, transparency=0.8)
                        ps.show()
                        ps.remove_all_structures()

                console.log(f"[yellow]-- -- done analyzing {master.c.name}-{slave.c.name} pair for {constraint_type}")

    def create_constraint(
            self,
            constraint_type: str,
            master: RealObject,
            slave: RealObject,
            idx_time: int, idx_frame: int, idx_kpt: int):
        c = Constraints()
        c.idx_c = self.ms.n_constraint
        c.type = constraint_type
        c.priority = priorities[constraint_type]

        c.idx_master = master.c.index
        c.idx_slave = slave.c.index
        c.obj_master = master.c.name
        c.obj_slave = slave.c.name
        c.obj_slave_scale = slave.c.spatial_scale

        c.idx_time = idx_time
        c.idx_keypoint = idx_kpt
        if slave.c.is_hand:
            c.kpt_descriptor = slave.descriptors.tolist()
        else:
            c.kpt_descriptor = slave.descriptors[c.idx_keypoint].tolist()                   # (D, )
        c.kpt_uv = slave.uv[c.idx_keypoint].tolist()                                        # (2, )

        c.idx_frame = idx_frame
        frame_neighbor_index = master.neighbor_idx[idx_frame]                               # (Q, )
        c.frame_neighbor_index = frame_neighbor_index.astype(int).tolist()                  # (Q, )
        c.frame_neighbor_coordinates = master.coordinates[frame_neighbor_index].tolist()    # (Q, 3)
        c.frame_neighbor_descriptors = master.descriptors[frame_neighbor_index].tolist()    # (Q, D)
        c.frame_neighbor_uvs = master.uv[frame_neighbor_index].tolist()                     # (Q, 2)
        console.log(f"---- {c.obj_master}: neighbor index in total {len(frame_neighbor_index)}: max {frame_neighbor_index.max()}")

        if self.n_demos >= 3:
            if c.priority < 2:  # of type p2p/l/P linear case, save PCA data (T, Pm, Ps, d+2, d) -> (d+2, d)
                c.pca_data = master.d.pca_data[slave.c.name][idx_time, idx_frame, idx_kpt].tolist()
            else:  # of type p2c/S non-linear case, save PME data
                # TODO implement for PME cases
                pass

        self.ms.n_constraint += 1
        self.ms.constraints[master.c.name][slave.c.name].append(c)
        console.log(f"[purple]-- -- -- {c}")
        return c

    def reload(self):
        console.rule("[bold green]K-VIL reloading ...")
        self.init()
        # self.organize_constraints_in_list()
        self._post_processing()

        console.log("[bold]Done (K-VIL reload)")

    def init(self):
        console.log("[green]K-VIL: initialize all master objects")
        for i in self.ms.master_obj_idx:
            self._objects[i].init_kvil()

    def pce(self):
        box_message("PCE", "blue")
        self.init()
        if self.n_demos < 3:
            self.distance_criteria()
        else:
            self.linear_constraints()
            if self.n_demos >= 8:
                self.nonlinear_constraint()

        self._post_processing()
        for obj in self._objects:
            obj.d.save("PCE", force_save=True)
        console.log("[bold]Done (PCE)")

    def _post_processing(self):
        self.organize_constraints_in_list()
        self._create_grasp_constraint()
        self._truncate_master_slave_pairs()
        self._create_free_constraint()
        self._reset_hierarchy_level()
        self._determine_constraint_time()
        self._compute_density()
        self.get_hand_group(is_swap=True)
        self.ms.show()
        self.ms.save()

    def _determine_constraint_time(self):
        if self.reload_flag:
            return

        if len(self.constraints.items()) == 0:
            console.rule(f"[bold red]No constraint found in your data")
            return

        self.ms.constraint_max_t = 0
        for c_idx, c in self.constraints.items():
            self.ms.constraint_max_t = max(self.ms.constraint_max_t, c.idx_time)

    def get_grasped_obj(self):
        return [obj for obj in self._objects if obj.c.is_grasped]

    def get_master_in_free_motion(self):
        return [obj for obj in self._objects if
                obj.c.is_master and not obj.c.is_slave() and not obj.c.is_hand and obj.c.hierarchical_lvl > 0]

    def _create_grasp_constraint(self):
        if not self.c.enable_grasping or self.reload_flag:
            return

        for obj in self.get_grasped_obj():
            for hand_idx in obj.c.grasping_hand_idx:
                grasping_hand = self._objects[hand_idx]
                c = self.create_constraint(
                    constraint_type="grasping", master=obj, slave=grasping_hand,
                    idx_time=(self.c.n_time_frames - 1),
                    idx_frame=obj.c.grasping_point_idx[hand_idx], idx_kpt=9)
                # c = self.ms.constraints[obj.c.name][grasping_hand.c.name][-1]
                if len(self.constraints_index) > 0:
                    nc = self.constraints_index[-1]
                else:
                    nc = -1
                c_index = nc + 1
                c.idx_c = c_index
                self.constraints_index.append(c_index)
                self.constraints[c_index] = c

                # get the list of constraint on c.obj_master at time c.idx_time, if empty, then append it
                c_on_master = self.ms.t_m_c_order[c.idx_time].get(c.obj_master, None)
                if c_on_master is None:
                    self.ms.t_m_c_order[c.idx_time][c.obj_master] = [c.idx_c]
                else:
                    c_on_master.append(c.idx_c)

                c.idx_c_on_master = len(self.ms.constraints[c.obj_master][c.obj_slave])
                c.save(self.proc_path)
                # obj.d.save()

    def _create_free_constraint(self):
        if not self.c.last_timestep or self.reload_flag:
            return

        for free_obj in self.get_master_in_free_motion():
            console.log(f"[yellow] The object {free_obj.c.name} is doing free movement")
            # find the tsvmp for this object:
            # ic(self.ms.slave_obj[free_obj.c.name])

            console.log(
                f"[yellow] Finding the best local frame to describe the free movement of object {free_obj.c.name}")

            c: Constraints = None
            for slave_idx in free_obj.c.slave_idx:
                slave_obj_name = self._objects[slave_idx].c.name
                for _c in self.ms.constraints[free_obj.c.name][slave_obj_name]:
                    if c is None:
                        c = _c
                        continue
                    if _c.priority < c.priority:
                        c = _c
                    if c.priority == 0:
                        break

            v_obj = self.get_object(f"virtual_{c.obj_master}")
            v_obj.set_as_slave(free_obj)

            c = self.create_constraint(
                constraint_type="free", master=v_obj, slave=free_obj,
                idx_time=c.idx_time, idx_frame=c.idx_frame, idx_kpt=c.idx_frame)

            c.idx_c = self.constraints_index[-1] + 1
            self.constraints_index.append(c.idx_c)
            self.constraints[c.idx_c] = c
            c_on_master = self.ms.t_m_c_order[c.idx_time].get(c.obj_master, None)
            if c_on_master is None:
                self.ms.t_m_c_order[c.idx_time][c.obj_master] = [c.idx_c]
            else:
                c_on_master.append(c.idx_c)

            c.idx_c_on_master = 0
            c.save(self.proc_path)

    def _truncate_master_slave_pairs(self):
        """
        un-set a slave if there's no constraint between it and its master
        """
        # we only check in the last_timestep, because in the first time step, there are always constraints between
        # virtual_obj and obj, therefore, the MSR should only be determined at the last timestep
        if not self.c.last_timestep or self.reload_flag:
            return

        for master_idx in self.ms.master_obj_idx:
            master = self._objects[master_idx]

            for slave_idx in self.ms.slave_obj_idx[master.c.name]:
                slave = self._objects[slave_idx]
                if len(self.ms.constraints[master.c.name][slave.c.name]) == 0:
                    master.set_as_non_slave(slave)

        for obj in self.get_real_objects():
            if not obj.c.is_master and not obj.c.is_slave() and not obj.c.is_static:
                raise RuntimeError(
                    f"Object {obj.c.name} moves but is neither master nor slave, check dataset, "
                    f"especially grasp detection")

    def _reset_hierarchy_level(self):
        if not self.c.last_timestep or self.reload_flag:
            return

        for obj in self._objects:
            obj.set_hierarchical_lvl(-1)

        for obj in self.get_static_objects():
            obj.set_hierarchical_lvl(0)

        moving_obj_list = self.get_moving_objects()
        while True:
            for obj in moving_obj_list:
                if obj.c.is_slave():
                    # wait until all its masters' hierarchy level are set, take the largest level as reference
                    master_hierarchical_lvl = [self._objects[idx].c.hierarchical_lvl for idx in obj.c.associated_master_idx]
                    if (-1) in master_hierarchical_lvl:
                        continue
                    obj.set_hierarchical_lvl(max(master_hierarchical_lvl) + 1)
                else:
                    # if a moving object is not a slave, it can be either a free motion object
                    # or a free moving hand
                    # Note: either this hand is actually moving freely, or the grasp detection failed in the symmetric case
                    #  In other coordination case, if the grasp detection failed for one hand
                    obj.set_hierarchical_lvl(1)

            if (-1) not in [obj.c.hierarchical_lvl for obj in moving_obj_list]:
                break

    def organize_constraints_in_list(self):
        console.rule("[bold cyan]Organizing constraints in list")
        nc = 0
        for idx_master in self.ms.master_obj_idx:
            master = self._objects[idx_master]

            for slave in master.c.slave_obj:
                constraints = self.ms.constraints[master.c.name][slave]
                for idx_c_on_master, c in enumerate(constraints):
                    # c.idx_c = nc
                    # nc += 1
                    # self.constraints[c.idx_c] = c
                    # self.constraints_index.append(c.idx_c)
                    if not self.reload_flag:
                        c.idx_c = nc
                        nc += 1
                        self.constraints[c.idx_c] = c
                        self.constraints_index.append(c.idx_c)
                        c_on_master = self.ms.t_m_c_order[c.idx_time].get(c.obj_master, None)
                        if c_on_master is None:
                            self.ms.t_m_c_order[c.idx_time][c.obj_master] = [c.idx_c]
                        else:
                            c_on_master.append(c.idx_c)
                        c.idx_c_on_master = idx_c_on_master
                        c.save(self.proc_path)
                    else:
                        c = c.load(self.proc_path)
                        self.constraints[c.idx_c] = c
                        self.constraints_index.append(c.idx_c)
                        self.ms.constraints[master.c.name][slave][idx_c_on_master] = c

    def compute_mps(self, request_redo_from_gui: bool = False):
        """ compute the mps of all constraint """
        box_message("Training VMPs", "blue")
        if self.reload_flag and not request_redo_from_gui:
            console.log("skip")
            self.ui.c_traj_ready = True
            console.log("[bold]Done (VMPs)")
            return

        # for idx_c in track(self.constraints_index, console=console):
        for idx_c in self.constraints_index:
            console.rule()
            c = self.constraints[idx_c]
            # traj_global = self._objects[c.idx_slave].get_global_traj_of_keypoint(c.idx_keypoint)
            # traj = self._objects[c.idx_master].get_vmp_train_traj_of_slave(c.idx_frame, traj_global)  # (N, T, d)
            # N, T, d = traj.shape
            # self._objects[c.idx_master].d.c_traj_data[c.idx_c] = traj.copy()
            # console.log(f"{c.type}, {traj.shape}, {c.idx_c}, {c.obj_master}, "
            #             f"{list(self._objects[c.idx_master].d.c_traj_data.keys())}")

            # if c.type == "p2p":
            #     vmp_dim = 3
            #     c.vmp_goal = c.pca_data[3]  # mean (3, )
            # elif c.type == "p2l":
            #     vmp_dim = 1
            #     traj = np.linalg.norm(np.cross(
            #         (traj - np.array(c.pca_data[3])), np.array(c.pca_data[0]), axis=-1),
            #         axis=-1, keepdims=True
            #     )  # (N, T, 1)
            #     c.vmp_goal = [0]
            # elif c.type == "p2P":
            #     vmp_dim = 1
            #     traj = (traj - np.array(c.pca_data[3])).dot(np.array(c.pca_data[-1]))[..., np.newaxis]
            #     c.vmp_goal = [0]
            # elif c.type == "p2c":
            #     vmp_dim = 3
            #     c.vmp_goal = np.array(c.pm_mean).flatten().tolist()  # Note: check this, make sure array is flat
            # elif c.type == "p2c":
            #     sol_opt_x = np.array(c.pm_sol_opt_x)
            #     sol_opt_t = np.array(c.pm_sol_opt_t)
            #     embedding = np.array(c.pm_embedding)  # (1, N)
            #     t_opt = np.array(c.pm_t_opt)
            #
            #     # traj_shape = traj.shape
            #     traj = traj.reshape(-1, 3)
            #
            #     proj_intrinsic = projection_index(traj / c.pm_scaling,                          # (N*T, 3)
            #                                       embedding.mean(axis=0, keepdims=True),        # (1, 1)
            #                                       sol_opt_x, sol_opt_t, t_opt, c.pm_intrinsic_dim)
            #     # ic(proj_intrinsic)
            #     proj_point = mapping(proj_intrinsic, d=c.pm_intrinsic_dim, sol_opt_x=sol_opt_x, sol_opt_t=sol_opt_t,
            #                          t_opt=t_opt)
            #     proj_point *= c.pm_scaling
            #     stress = traj - proj_point
            #     traj = np.linalg.norm(stress, axis=-1)
            #     traj = traj.reshape((N, T, 1))
            #     vmp_dim = 1
            #     c.vmp_goal = [0]
            # if c.type == "free":
            #     idx_time = np.arange(T)
            #     keypoint_8d_traj = []  # [t,x,y,z,qw,qx,qy,qz] (N,T,8)
            #     # TODO, why twice, why not just use the c.idx_frame? it is the same object (e.g. vc <- cup).
            #     ori_global_all_demo = self._objects[c.idx_slave].get_all_frame_mat_for_tvmp(
            #         idx_frame=c.idx_keypoint, idx_time=idx_time)
            #     frame_mat_all_demo = self._objects[c.idx_slave].get_all_frame_mat_for_tvmp(idx_frame=c.idx_frame,
            #                                                                                idx_time=idx_time)
            #     N, T, d = traj.shape
            #     # ic(ori_global_all_demo.shape,frame_mat_all_demo.shape)
            #     # target = []
            #     for demo_idx in range(N):
            #         keypoint_8d_traj_demo = []  # T,8
            #         demo_pos = traj[demo_idx]  # T,3
            #         # ic(demo_pos.shape)
            #         demo_ori_global = ori_global_all_demo[demo_idx]
            #         demo_frame_mat = frame_mat_all_demo[demo_idx]
            #         # ic(demo_ori_global.shape,demo_frame_mat.shape)
            #         for t in range(T):
            #             keypoint_8d_traj_at_t = []
            #             keypoint_8d_traj_at_t.append(t)  # [t]
            #             # ic(keypoint_8d_traj_at_T,demo_pos[t])
            #             keypoint_8d_traj_at_t += demo_pos[t].tolist()  # [t,x,y,z]
            #             demo_ori_local_t = demo_frame_mat[t, 0].T.dot(demo_ori_global[t, 0])
            #             # ic(keypoint_8d_traj_at_T,demo_ori_local_t)
            #             keypoint_8d_traj_at_t += quaternion_from_matrix(
            #                 demo_ori_local_t).tolist()  # [t,x,y,z,qw,qx,qy,qz]
            #             keypoint_8d_traj_demo.append(keypoint_8d_traj_at_t)
            #         # target.append(keypoint_8d_traj_demo[-1])
            #         keypoint_8d_traj.append(keypoint_8d_traj_demo)
            #     keypoint_8d_traj = np.array(keypoint_8d_traj)
            #     # target = np.array(target)
            #     # target_pos = target[:,1:4]
            #     # target_ori = target[:,4:]
            #     # ic(target_pos,target_ori)
            #     # ic(keypoint_8d_traj.shape)
            #
            #     c.vmp_kernel = 20
            #     tvmp = TVMP(kernel_num=c.vmp_kernel)
            #     tvmp.train(keypoint_8d_traj)
            #     c.tvmp_demo_traj = keypoint_8d_traj.tolist()
            #     c.tvmp_weights = tvmp.get_weights().tolist()
            #     # ic(tvmp.goal)
            #     c.tvmp_goal = tvmp.goal.tolist()
            #     c.save(self.proc_path)
            #
            #     continue

            if c.type == "grasping":
                continue
            elif c.type == "free":
                traj_frame = self._objects[c.idx_slave].get_tvmp_train_traj_of_slave(c.idx_frame)  # (N, T, 8)
                N, T, d = traj_frame.shape
                self._objects[c.idx_master].d.c_traj_data[c.idx_c] = traj_frame.copy()

                time_stamps = np.tile(np.linspace(0, 1, T).reshape(-1, 1), (N, 1, 1))
                traj_frame = np.concatenate((time_stamps, traj_frame), axis=-1)

                c.vmp_kernel = 20
                c.vmp_dim = 8
                tvmp = TVMP(kernel_num=c.vmp_kernel)
                tvmp.train(traj_frame)
                c.vmp_demo_traj = traj_frame.tolist()
                c.vmp_weights = tvmp.get_weights().tolist()
                c.vmp_goal = traj_frame[:, -1, 1:4].mean(0).tolist() + quaternion_average(traj_frame[:, -1, -4:]).tolist()

            elif c.type in ["p2p", "p2l", "p2P", "p2c", "p2S"]:
                traj_global = self._objects[c.idx_slave].get_global_traj_of_keypoint(c.idx_keypoint)
                traj = self._objects[c.idx_master].get_vmp_train_traj_of_slave(c.idx_frame, traj_global)  # (N, T, d)
                N, T, d = traj.shape
                self._objects[c.idx_master].d.c_traj_data[c.idx_c] = traj.copy()
                console.log(f"{c.type}, {traj.shape}, {c.idx_c}, {c.obj_master}, "
                            f"{list(self._objects[c.idx_master].d.c_traj_data.keys())}")

                vmp_dim = 3
                if c.type in ["p2c", "p2S"]:
                    c.vmp_goal = c.pm_mean
                else:
                    c.vmp_goal = c.pca_data[3]  # mean (3, )
                c.vmp_dim = vmp_dim
                c.vmp_kernel = 20
                vmp = VMP(c.vmp_dim, kernel_num=c.vmp_kernel)

                # check for early stop
                seg_point_t = T
                thr = 2e-4
                console.log(traj.shape)
                traj_var = np.linalg.norm(np.var(traj, axis=0), axis=-1)
                console.log(traj_var)
                low_variance_interval = np.where(traj_var < thr)[0]
                console.log(low_variance_interval)
                if low_variance_interval.shape[0] > 3 and low_variance_interval[0] > 0.2 * T:
                    seg_point_t = low_variance_interval[0]

                console.log(f"[red]-- Using segmentation point {seg_point_t} for the {idx_c}-th {c.type} constraint")
                time_stamps = np.tile(np.linspace(0, 1, seg_point_t).reshape(-1, 1), (N, 1, 1))
                traj = np.concatenate((time_stamps, traj[:, :seg_point_t]), axis=-1)
                vmp.train(traj)

                c.vmp_demo_traj = traj.tolist()
                c.vmp_weights = vmp.get_weights().tolist()
            else:
                raise NotImplementedError

            c.save(self.proc_path)

        self.ui.c_traj_ready = True
        console.log("[bold]Done (VMPs)")

    def _compute_density(self):
        """ compute the density function of all constraints """
        if not self.ui.c_traj_ready:
            self.compute_mps()

        if not self.reload_flag:
            box_message("computing density", "blue")
            # for idx_c in track(self.constraints_index, console=console):
            for idx_c in self.constraints_index:
                c = self.constraints[idx_c]
                data_points = self._objects[c.idx_master].d.appended_obj_traj[c.obj_slave][:, c.idx_time, c.idx_frame,
                              c.idx_keypoint]
                if c.type == "p2l":
                    # todo projection on to the line
                    line_vec = np.array(c.pca_data[0])
                    line_mean = np.array(c.pca_data[3])
                    projection = (data_points - line_mean).dot(line_vec).reshape(-1, 1)
                    weights, mu, sig, label = hdmde(projection, n_cluster_min=1, alpha=0.1, max_comp=1)

                    c.density_weights = weights.tolist()
                    c.density_mu = mu.tolist()
                    c.density_std_dev = sig

                elif c.type == "p2P":
                    plane_xy = np.array(c.pca_data[:2])
                    plane_mean = np.array(c.pca_data[3])
                    projection = (data_points - plane_mean).dot(plane_xy.T)
                    weights, mu, sig, label = hdmde(projection, n_cluster_min=1, alpha=0.9, max_comp=1)
                    # ic(projection, weights, mu, sig, label)

                    c.density_weights = weights.flatten().tolist()
                    c.density_mu = mu.tolist()
                    c.density_std_dev = sig

                    lim_x = 84
                    lim_y = 40
                    n = 31
                    x, y = np.mgrid[-lim_x:lim_x:31j, -lim_y:lim_y:31j]
                    test_data = np.stack((x.ravel(), y.ravel())).T
                    prob = np.zeros(test_data.shape[0])
                    prob_derivative = np.zeros_like(test_data)
                    for j in range(mu.shape[0]):
                        prob_ = st.multivariate_normal.pdf(test_data, mu[j], sig ** 2) * weights[j]
                        prob += prob_
                        prob_derivative -= prob_.reshape(-1, 1) * (test_data - mu[j]) / (sig ** 2)

                    # fig = plt.figure()
                    # ax = fig.add_subplot(111)
                    # ax.scatter(0, 0, color="yellow", edgecolor="k", zorder=10)
                    # ax.scatter(projection[:, 0], projection[:, 1], zorder=10)
                    # ax.contourf(test_data[:, 0].reshape(n, n), test_data[:, 1].reshape(n, n), prob.reshape(n, n),
                    #              levels=14, cmap="GnBu", alpha=1.0, zorder=-1)
                    # ax.quiver(test_data[:, 0], test_data[:, 1], prob_derivative[:, 0], prob_derivative[:, 1])
                    # plt.show()
                    # exit()

                c.save(self.proc_path)

        self.ui.c_pdf_ready = True
        console.log("[bold]Done (density functions)")

    def get_neighbor_descriptors(self, idx_obj: int, idx_keypoint: int, num_neighbors: int = 1):
        neighbor_idx = self._objects[idx_obj].neighbor_idx[idx_keypoint][:num_neighbors]
        return self._objects[idx_obj].descriptors[neighbor_idx]

    def get_hand_group(self, is_swap: bool = False):
        if not self.enable_hand_group:
            return
        obj_name_list = [obj.c.name for obj in self._objects]
        hand_list = [obj_name for obj_name in obj_name_list if (obj_name == "left_hand" or obj_name == "right_hand")]
        if len(hand_list) == 0:
            raise ValueError("No hand found in the list")
        is_bimanual = len(hand_list) == 2

        console.rule(f"[bold blue]Constructing Hand Group from KVIL Objects")
        c = HandGroupConfig()
        for hand_name in hand_list:
            console.log(f"[green]Hand group: {hand_name}")
            # constructing hand group for each side
            idx = obj_name_list.index(hand_name)
            hand = self._objects[idx]

            hand_group = HandGroup()
            hand_group.hand_name, hand_group.handedness = hand_name, hand_name.split("_")[0]
            hand_group.hand_idx = hand.c.index

            for obj in self._objects:
                if hand.c.index not in obj.c.grasping_hand_idx:
                    # if the hand does not grasp this object 'obj', let's skip
                    continue

                # Now hand is grasping this object
                hand_group.is_grasping = True
                hand_group.object_name = obj.c.name
                hand_group.object_idx = obj.c.index

                # Now let's check whether this object has another slave which is not a hand, if so, this group is master
                non_hand_slave = [slave for slave in self.ms.slave_obj[obj.c.name]
                                  if not ("left_hand" == slave or "right_hand" == slave)]
                hand_group.is_master = bool(non_hand_slave)

            c.hand_group_dict[hand_name] = hand_group

        # If only one hand, replace it with the other hand, only the handedness matters
        if not is_bimanual and is_swap:
            console.log("swap for uni-manual")
            is_left = "right_hand" == hand_list[0]
            g = c.hand_group_dict[hand_list[0]]
            g.hand_name = "right_hand" if is_left else "left_hand"
            g.handedness = "right" if is_left else "left"

        # If bimanual, we swap the grasped object and the master/slave flag
        if is_bimanual and is_swap:
            console.log("swap for bimanual")
            g_l = c.hand_group_dict["left_hand"]
            g_r = c.hand_group_dict["right_hand"]
            g_l.object_idx, g_r.object_idx = g_r.object_idx, g_l.object_idx
            g_l.object_name, g_r.object_name = g_r.object_name, g_l.object_name
            g_l.is_master, g_r.is_master = g_r.is_master, g_l.is_master

        if is_bimanual:
            console.log(f"[bold cyan]Determine the coordination category")
            is_symmetric = False
            for obj in self._objects:
                if obj.c.is_symmetric:
                    is_symmetric = True

            if is_symmetric:
                c.is_symmetric = True

            else:
                if not c.hand_group_dict["left_hand"].is_master and not c.hand_group_dict["right_hand"].is_master:
                    c.is_uncoordinated = True
                    console.log("coordination: uncoordinated")

        hand_group_config_file = self.proc_path / "hand_group_config.yaml"
        dump_data_to_yaml(HandGroupConfig, c, hand_group_config_file)
        return c

    def viz_in_global_callback(self):
        psim.PushItemWidth(150)
        psim.TextUnformatted("KVIL GUI")
        psim.Separator()

        # select object names
        psim.PushItemWidth(200)
        master_obj_changed = psim.BeginCombo("Pick master object", self.ui.obj_name_master)
        if master_obj_changed:
            for i, val in enumerate(self.obj_name_list):
                _, selected = psim.Selectable(val, self.ui.obj_name_master == val)
                if selected:
                    self.ui.obj_name_master = val
                    self.ui.idx_master_obj = i
                    self.ui.max_pts_master = self._objects[i].n_candidates_kvil
                    self.ui.max_time_kvil = self._objects[i].n_time
            psim.EndCombo()
        psim.PopItemWidth()

        time_changed_kvil, self.ui.idx_time_kvil = psim.SliderInt("K-VIL time", v=self.ui.idx_time_kvil, v_min=0,
                                                                  v_max=self.ui.max_time_kvil - 1)
        time_changed_raw, self.ui.idx_time_raw = psim.SliderInt("Demo time", v=self.ui.idx_time_raw, v_min=0,
                                                                v_max=self.ui.max_time_raw - 1)
        frame_changed, self.ui.idx_frame = psim.SliderInt("Index of frame", v=self.ui.idx_frame, v_min=0,
                                                          v_max=self.ui.max_pts_master - 1)
        if time_changed_kvil or frame_changed or time_changed_raw or self.ui.first_run:
            for i in range(self._n_obj):
                if i == self.ui.idx_master_obj:
                    self._objects[i].draw_frame(self.ui.idx_frame, self.ui.idx_time_kvil)
                if time_changed_kvil or time_changed_raw or self.ui.first_run:
                    kvil_flag = time_changed_kvil or self.ui.first_run
                    idx_time = self.ui.idx_time_kvil if kvil_flag else self.ui.idx_time_raw
                    self._objects[i].draw_pcl(idx_time, kvil=kvil_flag)
            if self.ui.first_run:
                self.ui.first_run = not self.ui.first_run

    def show_global_scene(self):
        # ps.look_at_dir([0, 0, -1], [0, 0, 0], [0, -1, 0], fly_to=True)
        self.ui.first_run = True
        master_idx = self.ms.master_obj_idx[0]
        master_name = self.obj_name_list[master_idx]
        self.ui.obj_name_master = master_name
        self.ui.obj_name_slave = self.ms.slave_obj[master_name][0]
        self.ui.idx_master_obj = master_idx
        self.ui.idx_time_kvil = self._objects[master_idx].c.kvil.n_time_frames - 1
        self.ui.max_pts_master = self._objects[master_idx].n_candidates_kvil
        self.ui.max_pts_raw = self._objects[master_idx].n_candidates_raw
        self.ui.max_time_kvil = self._objects[master_idx].n_time
        self.ui.max_time_raw = self._objects[master_idx].n_time_raw
        # origin = draw_frame_3d(np.zeros(6, dtype=float), scale=0.1, radius=0.005, alpha=0.8, label="scene_origin")

        box_message(f"Start K-VIL GUI for demonstrations in aligned camera frame", "blue")
        ps.set_user_callback(self.viz_in_global_callback)
        ps.show()
        ps.remove_all_structures()

    def viz_in_local_callback(self):
        self.update_ui_appearance_local()
        self.update_ui_content_local()

    def update_ui_appearance_local(self):
        psim.PushItemWidth(180)
        psim.TextUnformatted("KVIL GUI")
        psim.Separator()

        # select object names
        psim.PushItemWidth(300)
        self.ui.obj_master_changed = psim.BeginCombo("Pick master object", self.ui.obj_name_master)
        if self.ui.obj_master_changed:
            for i, val in enumerate(self.ms.master_obj):
                _, selected = psim.Selectable(val, self.ui.obj_name_master == val)
                if selected:
                    self.ui.obj_name_master = val
                    self.ui.idx_master_obj = i
                    self.ui.max_pts_master = self._objects[i].n_candidates_kvil
                    self.ui.max_time_kvil = self._objects[i].n_time
            psim.EndCombo()
        psim.PopItemWidth()

        self.ui.frame_changed, self.ui.idx_frame = psim.SliderInt("Index of frame", v=self.ui.idx_frame, v_min=0,
                                                                  v_max=self.ui.max_pts_master - 1)
        psim.SameLine()
        psim.Indent(260)
        psim.PushItemWidth(100)
        self.ui.o_scale_changed, self.ui.o_scale = psim.SliderFloat("origin scale", v=self.ui.o_scale, v_min=0, v_max=5)
        psim.Unindent(260)
        psim.PopItemWidth()

        self.ui.time_changed_kvil, self.ui.idx_time_kvil = psim.SliderInt("K-VIL time", v=self.ui.idx_time_kvil,
                                                                          v_min=0, v_max=self.ui.max_time_kvil - 1)
        psim.SameLine()
        psim.Indent(260)
        psim.PushItemWidth(100)
        self.ui.o_alpha_changed, self.ui.o_alpha = psim.SliderFloat("origin alpha", v=self.ui.o_alpha, v_min=0, v_max=1)
        psim.Unindent(260)
        psim.PopItemWidth()

        time_changed_raw, self.ui.idx_time_raw = psim.SliderInt("Demo time", v=self.ui.idx_time_raw, v_min=0,
                                                                v_max=self.ui.max_time_raw - 1)
        psim.SameLine()
        psim.Indent(260)
        psim.PushItemWidth(100)
        self.ui.o_radius_changed, self.ui.o_radius = psim.SliderFloat("origin radius", v=self.ui.o_radius, v_min=0, v_max=5)
        psim.Unindent(260)
        psim.PopItemWidth()

        psim.Separator()

        self.ui.alpha_master_changed, self.ui.alpha_master = psim.SliderFloat("master alpha", v=self.ui.alpha_master,
                                                                              v_min=0, v_max=1)
        psim.SameLine()
        self.ui.alpha_slave_changed, self.ui.alpha_slave = psim.SliderFloat("slave alpha", v=self.ui.alpha_slave,
                                                                            v_min=0, v_max=1)
        psim.Separator()
        for _v_obj in self.ui.o_virt_alpha.keys():
            self.ui.o_virt_alpha_changed[_v_obj], self.ui.o_virt_alpha[_v_obj] = psim.SliderFloat(
                f"{_v_obj} alpha", v=self.ui.o_virt_alpha[_v_obj], v_min=0, v_max=1
            )
            self.ui.o_virt_enabled[_v_obj] = self.ui.o_virt_alpha[_v_obj] > 0.01

        psim.Separator()
        psim.TextUnformatted("Constraints")

        self.ui.c_force_redraw = False
        psim.PushItemWidth(300)
        if psim.BeginCombo("Pick Constraint", self.ui.c_select_text[self.ui.c_selected_idx]):
            for i, val in enumerate(self.ui.c_select_text):
                _, selected = psim.Selectable(val, self.ui.c_selected_idx == val)
                if selected:
                    idx_c = int(str(val).split(" ")[0])
                    c = self.constraints[idx_c]
                    self.ui.c_selected_idx = idx_c

                    self.ui.obj_name_master = c.obj_master
                    self.ui.obj_name_slave = c.obj_slave
                    self.ui.idx_master_obj = c.idx_master
                    self.ui.idx_frame = c.idx_frame
                    self.ui.idx_time_kvil = c.idx_time

                    self.ui.max_pts_master = self._objects[c.idx_master].n_candidates_kvil
                    self.ui.max_pts_raw = self._objects[c.idx_master].n_candidates_raw

                    self.ui.c_force_redraw = True
            psim.EndCombo()
        psim.PopItemWidth()
        psim.Separator()

        for t in range(self.c.n_time_frames):
            psim.TextColored(colors.yellow_green, text=f"time {t}")

            for master_name in self.ms.t_m_c_order[t].keys():
                psim.TextColored(colors.baby_blue, text=master_name)

                for idx_c in self.ms.t_m_c_order[t][master_name]:
                    if idx_c in self.ms.ignore_idx:
                        continue
                    c = self.constraints[idx_c]
                    psim.PushItemWidth(150)
                    c_type = "grasp" if c.type == 'grasping' else c.type
                    self.ui.c_enable_changed[idx_c], self.ui.c_enable[idx_c] = psim.Checkbox(
                        f"{c.idx_c:>02d} {c_type:<4}", self.ui.c_enable[idx_c])
                    psim.SameLine()
                    psim.Indent(100)
                    psim.TextColored(colors.vanilla, f"{c.obj_slave[:8]:<8}")
                    psim.SameLine()
                    psim.Indent(70)
                    self.ui.c_color_changed[idx_c], self.ui.c_color_value[idx_c] = psim.ColorEdit4(
                        f"{c.idx_c:>02d}", self.ui.c_color_value[idx_c])
                    psim.SameLine()
                    psim.PushItemWidth(100)
                    self.ui.c_scale_changed[idx_c], self.ui.c_scale[idx_c] = psim.SliderFloat(
                        f"{c.idx_c:>02d}", self.ui.c_scale[idx_c], v_min=0, v_max=10)
                    psim.Unindent(170)
                    psim.PushItemWidth(200)

            psim.Separator()

        self.ui.c_pca_scale_changed, self.ui.c_pca_scale = psim.SliderFloat(
            "PCA vector scale", v=self.ui.c_pca_scale, v_min=0.1, v_max=10)
        psim.SameLine()
        self.ui.c_pca_revert_changed, self.ui.c_pca_revert = psim.Checkbox(f"revert direction", self.ui.c_pca_revert)
        psim.Separator()

        # ================ VMP trajectory ================
        if psim.Button("Compute VMP" if not self.ui.c_traj_ready else "VMP Done", (120, 30)):
            self.compute_mps(request_redo_from_gui=True)
        psim.SameLine()
        self.ui.c_traj_flag_changed, self.ui.c_traj_flag = psim.Checkbox(f"traj", self.ui.c_traj_flag)
        psim.SameLine()
        psim.PushItemWidth(80)
        self.ui.c_traj_alpha_changed, self.ui.c_traj_alpha = psim.SliderFloat(
            "t_alpha", v=self.ui.c_traj_alpha, v_min=0, v_max=1)
        psim.SameLine()
        psim.PushItemWidth(80)
        self.ui.c_traj_scale_changed, self.ui.c_traj_scale = psim.SliderFloat(
            "t_scale", v=self.ui.c_traj_scale, v_min=0.1, v_max=2)

        # ================ density function ================
        if psim.Button("Estimate density" if not self.ui.c_pdf_ready else "Density Done", (120, 30)):
            self._compute_density()
        psim.SameLine()
        self.ui.c_pdf_flag_changed, self.ui.c_pdf_flag = psim.Checkbox(f"pdf", self.ui.c_pdf_flag)
        psim.SameLine()
        psim.PushItemWidth(80)
        self.ui.c_pdf_alpha_changed, self.ui.c_pdf_alpha = psim.SliderFloat(
            "p_alpha", v=self.ui.c_pdf_alpha, v_min=0, v_max=1)
        psim.SameLine()
        self.ui.c_pdf_radius_changed, self.ui.c_pdf_radius = psim.SliderFloat(
            "p_radius", v=self.ui.c_pdf_radius, v_min=0.1, v_max=5)

        psim.Indent(130)
        psim.PushItemWidth(120)
        self.ui.c_pdf_span_changed, self.ui.c_pdf_span = psim.SliderFloat(
            "p_span", v=self.ui.c_pdf_span, v_min=0.1, v_max=2)
        psim.SameLine()
        self.ui.c_pdf_scale_changed, self.ui.c_pdf_scale = psim.SliderFloat(
            "p_scale", v=self.ui.c_pdf_scale, v_min=0.1, v_max=20)
        psim.Unindent(130)

        psim.Separator()
        for idx_c in self.constraints_index:
            self.ui.c_frame_neighbors_changed[idx_c], self.ui.c_frame_neighbors_enabled[idx_c] = False, False
            # self.ui.c_frame_neighbors_changed[idx_c], self.ui.c_frame_neighbors_enabled[idx_c] = psim.Checkbox(
            #     f"cn{idx_c}", self.ui.c_frame_neighbors_enabled.get(idx_c, False)
            # )

    def update_ui_content_local(self):
        # update flag
        if self.ui.c_force_redraw:
            for id_c in range(self.ui.n_c):
                self.ui.c_enable[id_c] = False
                self.ui.c_enable_changed[id_c] = True
                self.ui.c_enable[self.ui.c_selected_idx] = True

        for i in range(self.ui.n_c):
            self.ui.c_update_geom[i] = self.ui.obj_master_changed or self.ui.frame_changed or self.ui.c_force_redraw

        # take actions to update content
        # draw global frame
        if self.ui.o_scale_changed or self.ui.o_alpha_changed or self.ui.first_run or self.ui.o_radius_changed:
            self.ui.origin = draw_frame_3d(
                np.zeros(6, dtype=float), scale=0.1 * self.ui.o_scale, alpha=self.ui.o_alpha, label="scene_origin",
                radius=self.ui.o_scale * self.ui.point_radius * 5 * self.ui.o_radius, collections=self.ui.origin
            )

        # draw object point clouds
        if self.ui.time_changed_kvil or self.ui.frame_changed or self.ui.first_run or \
                self.ui.c_force_redraw or self.ui.obj_master_changed or \
                self.ui.frame_changed or self.ui.alpha_master_changed or self.ui.alpha_slave_changed or \
                any(self.ui.o_virt_alpha_changed.values()):
            self._objects[self.ui.idx_master_obj].draw_pcl_local(
                self.ui, self.ui.idx_time_kvil, self.ui.idx_frame,
                (self.ui.alpha_master, self.ui.alpha_slave), self.ui.o_virt_alpha, self.ui.o_virt_enabled
            )
            if self.ui.first_run:
                self.ui.first_run = not self.ui.first_run

        self.draw_constraint()

    def draw_constraint(self):
        for idx_c in range(self.ui.n_c):
            c = self.constraints[idx_c]
            if "_hand" in c.obj_slave:
                continue
            global_master = self._objects[self.ui.idx_master_obj]
            c_master = self._objects[c.idx_master]

            same_master = c_master.c.index == global_master.c.index
            same_frame = self.ui.idx_frame == c.idx_frame

            def transform_to_global_master(data: np.ndarray, rot_only: bool = False):
                """
                Args:
                    data: (n, t, d), n >= 1
                    rot_only: if True, then the data will be considered as an vector, we only rotate it. If False,
                        then we consider data as points in 3D, we need to add translation.

                Returns: (n, t, d) transformed data points
                """
                if not (same_master and same_frame):
                    dim = data.shape[-1]
                    c_master_local_frame = c_master.frame_mat_all_demo[:, self.ui.idx_time_kvil, c.idx_frame]  # (N, d, d+1)
                    c_global_master_local_frame = global_master.frame_mat_all_demo[:, self.ui.idx_time_kvil, self.ui.idx_frame]
                    data = np.einsum("nid,ntd->nti", c_master_local_frame[..., :dim], data)
                    if not rot_only:
                        data += np.expand_dims(c_master_local_frame[..., -1], 1)
                        data -= np.expand_dims(c_global_master_local_frame[..., -1], 1)
                    data = np.einsum("nil,nti->ntl", c_global_master_local_frame[..., :dim], data)
                if data.shape[0] != self.n_demos:
                    data = np.repeat(data, self.n_demos, 0)
                return data

            def transform_from_cam_to_global_master(data: np.ndarray):
                """
                data: (t, dim), here t might be the number of neighbors for example.
                """
                dim = data.shape[-1]
                c_global_master_local_frame = global_master.frame_mat_all_demo[0, self.ui.idx_time_kvil, self.ui.idx_frame]  # (d, d+1)
                return np.einsum("ij,ti->tj",
                                 c_global_master_local_frame[..., :dim],
                                 data - c_global_master_local_frame[..., -1])

            if self.ui.c_frame_neighbors_changed[c.idx_c]:
                neighbor_points = c_master.d.appended_obj_traj[c_master.c.name][:, self.ui.idx_time_kvil, self.ui.idx_frame, c.frame_neighbor_index]  # (N, Q, d)
                neighbor_points = transform_to_global_master(neighbor_points)
                if self.ui.c_frame_neighbors.get(c.idx_c, None) is None:
                    self.ui.c_frame_neighbors[c.idx_c] = {}
                    for n in range(self.n_demos):
                        self.ui.c_frame_neighbors[c.idx_c][n] = ps.register_point_cloud(
                            f"c_neighbors_{c.idx_c:02d}_n{n:02d}", neighbor_points[n], enabled=True,
                            point_render_mode="sphere", color=c_master.demo_colors[c.idx_c],
                            transparency=self.ui.c_alpha[c.idx_c]
                        )
                else:
                    for n in range(self.n_demos):
                        self.ui.c_frame_neighbors[c.idx_c][n].update_point_positions(neighbor_points[n])
                        self.ui.c_frame_neighbors[c.idx_c][n].set_enabled(self.ui.c_frame_neighbors_enabled[c.idx_c])
                for n in range(self.n_demos):
                    self.ui.c_frame_neighbors[c.idx_c][n].set_radius(self.ui.point_radius * 2, relative=False)

            # draw keypoints in N demos. (N, T, Pm, Ps, d) -> (N, d)
            keypoints_n_demo = global_master.d.appended_obj_traj[c.obj_slave][:, c.idx_time, self.ui.idx_frame, c.idx_keypoint]
            if self.ui.c_keypoints.get(c.idx_c, None) is None:
                self.ui.c_keypoints[c.idx_c] = {}
                for n in range(self.n_demos):
                    self.ui.c_keypoints[c.idx_c][n] = ps.register_point_cloud(
                        f"kpts_c{c.idx_c:>02d}_d{n:>02d}", keypoints_n_demo[[n]], enabled=self.ui.c_enable[c.idx_c],
                        point_render_mode="sphere",
                        color=c_master.demo_colors[n],
                        transparency=self.ui.c_alpha[c.idx_c]
                    )
                    self.ui.c_keypoints[c.idx_c][n].set_radius(self.ui.point_radius * 3, relative=False)
            else:
                for n in range(self.n_demos):
                    geom = self.ui.c_keypoints[c.idx_c][n]
                    if self.ui.c_color_changed[c.idx_c]:
                        geom.set_color(self.ui.c_color_value[c.idx_c])
                        geom.set_transparency(self.ui.c_color_value[c.idx_c][3])
                    if self.ui.c_update_geom[c.idx_c]:
                        geom.update_point_positions(keypoints_n_demo[[n]])
                    if self.ui.c_enable_changed[c.idx_c]:
                        geom.set_enabled(self.ui.c_enable[c.idx_c])
                    if self.ui.c_scale_changed[c.idx_c]:
                        geom.set_radius(self.ui.point_radius * 3 * self.ui.c_scale[c.idx_c], relative=False)

            # draw PCA or PME
            if c.priority < 2:  # linear constraints: priority < 2, PCA data (T, Pm, Ps, d+2, d)
                draw_pca_flag = not self.ui.c_geom_shape.get(c.idx_c, None) or \
                                self.ui.c_scale_changed[c.idx_c] or \
                                self.ui.c_pca_scale_changed or self.ui.c_pca_revert_changed or \
                                self.ui.c_enable_changed[c.idx_c] or self.ui.c_update_geom[c.idx_c]

                if draw_pca_flag and self.n_demos >= 3:
                    if self.ui.c_geom_shape.get(c.idx_c, None) is None:
                        self.ui.c_geom_shape[c.idx_c] = {}

                    pca_data = np.array(c.pca_data, dtype=float)
                    pca_data[:3] = pca_data[:3].T
                    pca_data = np.tile(pca_data, (self.n_demos, 1, 1))  # (d+2, d) -> (n_demo, d+2, d)
                    dim = c_master.dim

                    if not (same_master and same_frame):
                        # constraint on different local frames, need to map it to this object
                        c_master_local_frame = c_master.frame_mat_all_demo[:, self.ui.idx_time_kvil, c.idx_frame]  # (N, d, d+1)
                        c_global_master_local_frame = global_master.frame_mat_all_demo[:, self.ui.idx_time_kvil, self.ui.idx_frame]
                        pca_data[:, :dim] = np.einsum("nil,nij,njk->nlk", c_global_master_local_frame[..., :dim], c_master_local_frame[..., :dim], pca_data[:, :dim])
                        pca_data[:, dim] = np.einsum("nij,nj->ni", c_master_local_frame[..., :dim], pca_data[:, dim]) + c_master_local_frame[..., -1]
                        pca_data[:, dim] = np.einsum("nil,ni->nl", c_global_master_local_frame[..., :dim], pca_data[:, dim] - c_global_master_local_frame[..., -1])

                    for n in range(self.n_demos):
                        dir = -1 if self.ui.c_pca_revert else 1
                        self.ui.c_geom_shape[c.idx_c][n] = draw_frame_3d(
                            np.concatenate((pca_data[n, :dim, :dim], pca_data[n, dim][..., np.newaxis]), axis=-1),
                            scale=pca_data[n, -1] * 20 * self.ui.c_scale[c.idx_c] * self.ui.c_pca_scale * dir,
                            alpha=1.0, collections=self.ui.c_geom_shape[c.idx_c].get(n, None),
                            label=f"pca_{c.idx_c:>02d}_d{n:>02d}", radius=self.ui.frame_radius * 5,
                            enabled=self.ui.c_enable[c.idx_c]
                        )
            else:  # nonlinear constraints, PME data
                draw_pme_flag = not self.ui.c_geom_shape.get(c.idx_c, None) == 0 or \
                                self.ui.c_scale_changed[c.idx_c] or \
                                self.ui.c_pme_scale_changed or \
                                self.ui.c_enable_changed[c.idx_c] or self.ui.c_update_geom[c.idx_c]

                if draw_pme_flag and self.n_demos > 10:
                    scale = c.pm_scaling
                    mean = np.array(c.pm_mean).reshape((1, 1, -1))          # (1, 1, 3)
                    mean = transform_to_global_master(mean)                 # (n_demo, 1, 3)
                    if self.ui.c_geom_shape.get(c.idx_c, None) is None:
                        self.ui.c_geom_shape[c.idx_c] = {}  # mean position for each demo not yet registered
                        for n in range(self.n_demos):
                            self.ui.c_geom_shape[c.idx_c][n] = ps.register_point_cloud(
                                f"mean_c{c.idx_c:>02d}_d{n:>02d}", mean[n], enabled=self.ui.c_enable[c.idx_c],
                                point_render_mode="sphere",
                                color=colors.white_smoke,
                                transparency=self.ui.c_alpha[c.idx_c]
                            )
                            self.ui.c_geom_shape[c.idx_c][n].set_radius(self.ui.point_radius * 2, relative=False)
                    else:
                        for n in range(self.n_demos):
                            self.ui.c_geom_shape[c.idx_c][n].update_point_positions(mean[n])
                            self.ui.c_geom_shape[c.idx_c][n].set_radius(self.ui.point_radius * 2, relative=False)
                            self.ui.c_geom_shape[c.idx_c][n].set_transparency(self.ui.c_alpha[c.idx_c])
                            self.ui.c_geom_shape[c.idx_c][n].set_enabled(self.ui.c_enable[c.idx_c])

                    projection = np.array(c.pm_projection)[:, np.newaxis] * scale           # (n_demo, 3) -> (n_demo, 1, 3)
                    projection = transform_to_global_master(projection)                     # (n_demo, 1, 3)
                    stress_vectors = np.array(c.pm_stress_vectors)[:, np.newaxis] * scale   # (n_demo, 3) -> (n_demo, 1, 3)
                    stress_vectors = transform_to_global_master(stress_vectors, True)       # (n_demo, 1, 3)

                    if self.ui.c_stress_vector.get(c.idx_c, None) is None:
                        self.ui.c_stress_vector[c.idx_c] = {}  # stress vector for each demo not yet registered
                        for n in range(self.n_demos):
                            self.ui.c_stress_vector[c.idx_c][n] = ps.register_point_cloud(
                                f"projection_c{c.idx_c:>02d}_d{n:>02d}", projection[n],
                                enabled=self.ui.c_enable[c.idx_c],
                                point_render_mode="sphere",
                                color=c_master.demo_colors[n],
                                transparency=self.ui.c_alpha[c.idx_c]
                            )
                            self.ui.c_stress_vector[c.idx_c][n].set_radius(self.ui.point_radius * 2, relative=False)
                    else:
                        for n in range(self.n_demos):
                            self.ui.c_stress_vector[c.idx_c][n].update_point_positions(projection[n])
                            self.ui.c_stress_vector[c.idx_c][n].set_radius(self.ui.point_radius * 2, relative=False)
                            self.ui.c_stress_vector[c.idx_c][n].set_transparency(self.ui.c_alpha[c.idx_c])
                            self.ui.c_stress_vector[c.idx_c][n].set_enabled(self.ui.c_enable[c.idx_c])
                            self.ui.c_stress_vector[c.idx_c][n].remove_all_quantities()

                    for n in range(self.n_demos):
                        self.ui.c_stress_vector[c.idx_c][n].add_vector_quantity(
                            f"stress_vector_c{c.idx_c:>02d}_d{n:>02d}", stress_vectors[n],
                            enabled=self.ui.c_enable[c.idx_c],
                            length=np.linalg.norm(stress_vectors[n]),
                            color=c_master.demo_colors[n],
                            radius=self.ui.point_radius * 0.5,
                            vectortype="ambient"
                        )

            # draw trajectory of keypoints
            if self.ui.c_traj_ready and (
                    self.ui.c_traj_flag_changed or
                    self.ui.c_traj_alpha_changed or
                    self.ui.c_traj_scale_changed or
                    self.ui.c_enable_changed[idx_c]
            ):
                # ic(c_master.c.name,c_master.d.c_traj_data.keys())
                if c.type == "free":
                    data = c_master.d.c_traj_data[c.idx_c][..., :3]  # (N, T, 8) -> (N, T, 3)
                    data_frame = c_master.d.c_traj_data[c.idx_c][:, -1]  # (N, 7)

                    if self.ui.c_traj.get(c.idx_c, None) is None:
                        self.ui.c_traj_frame[c.idx_c] = {}

                    for n in range(self.n_demos):
                        self.ui.c_traj_frame[c.idx_c][n] = draw_frame_3d(
                            data_frame[n], scale=0.1, alpha=self.ui.c_traj_alpha,
                            collections=self.ui.c_traj_frame[idx_c].get(n, None),
                            label=f"frame_traj_c{c.idx_c:>02d}_d{n:>02d}",
                            radius=self.ui.o_scale * self.ui.point_radius * 2 * self.ui.c_traj_scale,
                            enabled=self.ui.c_traj_flag and self.ui.c_enable[idx_c]
                        )
                else:
                    data = c_master.d.c_traj_data[c.idx_c]

                traj_data = transform_to_global_master(data)  # (N, T, d)

                if self.ui.c_traj.get(c.idx_c, None) is None:
                    self.ui.c_traj[c.idx_c] = {}
                    T = traj_data.shape[1]
                    edges = np.array([np.arange(0, T - 1), np.arange(1, T)]).T
                    for n in range(self.n_demos):
                        self.ui.c_traj[idx_c][n] = ps.register_curve_network(
                            f"kpts_traj_c{c.idx_c:>02d}_d{n:>02d}", nodes=traj_data[n], edges=edges,
                            enabled=self.ui.c_traj_flag and self.ui.c_enable[idx_c],
                            color=c_master.demo_colors[n], transparency=self.ui.c_traj_alpha
                        )
                        self.ui.c_traj[idx_c][n].set_radius(
                            self.ui.curve_radius * self.ui.c_traj_scale * 0.2, relative=False
                        )
                else:
                    for n in range(self.n_demos):
                        self.ui.c_traj[idx_c][n].update_node_positions(traj_data[n])
                        self.ui.c_traj[idx_c][n].set_enabled(self.ui.c_traj_flag and self.ui.c_enable[idx_c])
                        self.ui.c_traj[idx_c][n].set_transparency(self.ui.c_traj_alpha)
                        self.ui.c_traj[idx_c][n].set_radius(self.ui.curve_radius * self.ui.c_traj_scale, relative=False)

            # draw pdfs of a constraint
            if self.ui.c_pdf_ready and c.type != "p2p" and (
                    self.ui.c_pdf_flag_changed or
                    self.ui.c_pdf_alpha_changed or
                    self.ui.c_pdf_radius_changed or
                    self.ui.c_pdf_scale_changed or
                    self.ui.c_pdf_span_changed or
                    self.ui.c_enable_changed[idx_c]
            ):
                weights = np.array(c.density_weights)
                mu = np.array(c.density_mu)
                sig = np.array(c.density_std_dev) * self.ui.c_pdf_span
                if c.type == "p2l":
                    # samples = int(100 * self.ui.c_pdf_scale)
                    samples = 500
                    scale = self._objects[c.idx_slave].c.spatial_scale
                    x = np.linspace(-1, 1, samples) * scale * self.ui.c_pdf_scale
                    prob = np.zeros(samples)
                    # prob_derivative = np.zeros_like(x)
                    for j in range(np.size(mu)):
                        var = sig ** 2
                        prob_ = st.multivariate_normal.pdf(x.reshape(-1, 1), mu[j], var) * weights[j]
                        prob += prob_
                        # prob_derivative -= prob_ * (x - mu[j]) / var

                    viz_scale = c.obj_slave_scale * 0.5 / np.max(prob) / self.ui.c_pdf_span
                    prob = prob[..., np.newaxis] * viz_scale
                    edges = np.array([np.arange(0, samples - 1), np.arange(1, samples)]).T

                    mean = np.array(c.pca_data[3])
                    line = np.array(c.pca_data[0]) * x[..., np.newaxis] + mean                  # (samples, 3)
                    pdf_line = np.array(c.pca_data[2]) * prob + line                            # (samples, 3)
                    line = transform_to_global_master(line[np.newaxis, ...])                    # (n_demo, samples, 3)
                    pdf_line = transform_to_global_master(pdf_line[np.newaxis, ...])            # (n_demo, samples, 3)

                    if self.ui.c_pdf_func.get(c.idx_c, None) is None:
                        self.ui.c_pdf_func[c.idx_c] = {}

                    if self.ui.c_pdf_base.get(c.idx_c, None) is None:
                        self.ui.c_pdf_base[c.idx_c] = {}

                    if len(self.ui.c_pdf_func[idx_c]) == 0:
                        for n in range(self.n_demos):
                            self.ui.c_pdf_base[idx_c][n] = ps.register_curve_network(
                                f"kpts_pdf_base_c{c.idx_c:>02d}_d{n:>02d}", nodes=line[n], edges=edges,
                                enabled=self.ui.c_pdf_flag and self.ui.c_enable[idx_c],
                                radius=self.ui.curve_radius * self.ui.c_pdf_scale * self.ui.c_pdf_radius,
                                color=c_master.demo_colors[n], transparency=self.ui.c_pdf_alpha
                            )
                            self.ui.c_pdf_func[idx_c][n] = ps.register_curve_network(
                                f"kpts_pdf_func_c{c.idx_c:>02d}_d{n:>02d}", nodes=pdf_line[n], edges=edges,
                                enabled=self.ui.c_pdf_flag and self.ui.c_enable[idx_c],
                                radius=self.ui.curve_radius * self.ui.c_pdf_scale * self.ui.c_pdf_radius * 0.5,
                                color=c_master.demo_colors[n], transparency=self.ui.c_pdf_alpha
                            )
                    else:
                        for n in range(self.n_demos):
                            self.ui.c_pdf_base[idx_c][n].update_node_positions(line[n])
                            self.ui.c_pdf_func[idx_c][n].update_node_positions(pdf_line[n])
                            self.ui.c_pdf_func[idx_c][n].update_node_positions(pdf_line[n])

                            self.ui.c_pdf_base[idx_c][n].set_enabled(self.ui.c_pdf_flag and self.ui.c_enable[idx_c])
                            self.ui.c_pdf_func[idx_c][n].set_enabled(self.ui.c_pdf_flag and self.ui.c_enable[idx_c])
                            self.ui.c_pdf_base[idx_c][n].set_transparency(self.ui.c_pdf_alpha)
                            self.ui.c_pdf_func[idx_c][n].set_transparency(self.ui.c_pdf_alpha)
                            self.ui.c_pdf_func[idx_c][n].set_radius(
                                self.ui.curve_radius * self.ui.c_pdf_scale * self.ui.c_pdf_radius * 0.5)
                            self.ui.c_pdf_base[idx_c][n].set_radius(
                                self.ui.curve_radius * self.ui.c_pdf_scale * self.ui.c_pdf_radius)

                elif c.type == "p2P":
                    pass

                elif c.type == "p2c":
                    sol_opt_x = np.array(c.pm_sol_opt_x)
                    sol_opt_t = np.array(c.pm_sol_opt_t)
                    t_opt = np.array(c.pm_t_opt)
                    d = c.pm_intrinsic_dim
                    embedding = np.array(c.pm_embedding)
                    scale = c.pm_scaling

                    samples = 500
                    obj_scale = self._objects[c.idx_slave].c.spatial_scale
                    # TODO why scaling is not working
                    t_test = np.linspace(
                        -0.8 * np.abs(np.min(embedding)), 0.8 * np.abs(np.max(embedding)), samples
                    ) / obj_scale * self.ui.c_pdf_scale
                    x_test = mapping(t_test, d=d, sol_opt_x=sol_opt_x, sol_opt_t=sol_opt_t, t_opt=t_opt)

                    # Note: compute density on manifold
                    density_data = embedding.reshape((-1, d))

                    theta_hat, mu, sig, label = hdmde(density_data, n_cluster_min=1, alpha=0.1, max_comp=1)
                    if d == 1:
                        t_test.reshape(-1, 1)
                        p = np.zeros_like(t_test)
                        prob_derivative = np.zeros_like(t_test)
                        for i in range(mu.shape[0]):
                            prob_ = st.multivariate_normal.pdf(t_test.reshape(-1, 1), mu[i], sig) * theta_hat[i]
                            p += prob_
                            prob_derivative -= prob_ * (t_test - mu[i]) / (sig ** 2)
                    elif d == 2:
                        lim = 4
                        x, y = np.mgrid[-lim:lim:31j, -lim:lim:31j]
                        test_data = np.stack((x.ravel(), y.ravel())).T
                        p = np.zeros(test_data.shape[0])
                        prob_derivative = np.zeros_like(test_data)
                        for i in range(mu.shape[0]):
                            prob_ = st.multivariate_normal.pdf(test_data, mu[i], sig) * theta_hat[i]
                            p += prob_
                            prob_derivative -= prob_.reshape(-1, 1) * (test_data - mu[i]) / (sig ** 2)
                    else:
                        raise ValueError
                    # entropy = - (p * np.log(p)).sum()

                    density_func = x_test.copy()
                    density_func[:, 1] -= p

                    line = transform_to_global_master(x_test[np.newaxis, ...] * scale)
                    pdf_line = transform_to_global_master(density_func[np.newaxis, ...] * scale)
                    edges = np.array([np.arange(0, samples - 1), np.arange(1, samples)]).T

                    if self.ui.c_pdf_func.get(c.idx_c, None) is None:
                        self.ui.c_pdf_func[c.idx_c] = {}

                    if self.ui.c_pdf_base.get(c.idx_c, None) is None:
                        self.ui.c_pdf_base[c.idx_c] = {}

                    if len(self.ui.c_pdf_func[idx_c]) == 0:
                        for n in range(self.n_demos):
                            self.ui.c_pdf_base[idx_c][n] = ps.register_curve_network(
                                f"kpts_pdf_base_c{c.idx_c:>02d}_d{n:>02d}", nodes=line[n], edges=edges,
                                enabled=self.ui.c_pdf_flag and self.ui.c_enable[idx_c],
                                color=c_master.demo_colors[n], transparency=self.ui.c_pdf_alpha
                            )
                            self.ui.c_pdf_base[idx_c][n].set_radius(
                                self.ui.curve_radius * self.ui.c_pdf_scale * self.ui.c_pdf_radius, relative=False)
                            self.ui.c_pdf_func[idx_c][n] = ps.register_curve_network(
                                f"kpts_pdf_func_c{c.idx_c:>02d}_d{n:>02d}", nodes=pdf_line[n], edges=edges,
                                enabled=self.ui.c_pdf_flag and self.ui.c_enable[idx_c],
                                color=c_master.demo_colors[n], transparency=self.ui.c_pdf_alpha
                            )
                            self.ui.c_pdf_func[idx_c][n].set_radius(
                                self.ui.curve_radius * self.ui.c_pdf_scale * 0.5 * self.ui.c_pdf_radius, relative=False)
                    else:
                        for n in range(self.n_demos):
                            self.ui.c_pdf_base[idx_c][n].update_node_positions(line[n])
                            self.ui.c_pdf_func[idx_c][n].update_node_positions(pdf_line[n])
                            self.ui.c_pdf_func[idx_c][n].update_node_positions(pdf_line[n])

                            self.ui.c_pdf_base[idx_c][n].set_enabled(self.ui.c_pdf_flag and self.ui.c_enable[idx_c])
                            self.ui.c_pdf_func[idx_c][n].set_enabled(self.ui.c_pdf_flag and self.ui.c_enable[idx_c])
                            self.ui.c_pdf_base[idx_c][n].set_transparency(self.ui.c_pdf_alpha)
                            self.ui.c_pdf_func[idx_c][n].set_transparency(self.ui.c_pdf_alpha)
                            self.ui.c_pdf_func[idx_c][n].set_radius(
                                self.ui.curve_radius * self.ui.c_pdf_scale * 0.5 * self.ui.c_pdf_radius)
                            self.ui.c_pdf_base[idx_c][n].set_radius(
                                self.ui.curve_radius * self.ui.c_pdf_scale * self.ui.c_pdf_radius)

    def show_constraints(self):
        if len(self.constraints_index) == 0:
            console.rule("Exit, no constraints to visualize.")
            exit()
        self.init_ui_local()
        box_message(f"Start K-VIL GUI for geometric constraints in aligned local frame", "green")
        ps.set_user_callback(self.viz_in_local_callback)
        ps.show()
        ps.remove_all_structures()
        self.ms.save()

    def init_ui_local(self):
        ps.set_give_focus_on_show(True)
        ps.look_at_dir([0, 0, -0.4], [0, 0, 0], [0, -1, 0], fly_to=True)
        self.ui.first_run = True
        c_at_kvil_time_dict = self.ms.t_m_c_order[self.ms.constraint_max_t]
        # find the first constraint at kvil_time that can be accessed
        idx_c = list(c_at_kvil_time_dict.values())[0][0]
        c = self.constraints[idx_c]
        self.ui.c_selected = idx_c

        self.ui.obj_name_master = c.obj_master
        self.ui.obj_name_slave = c.obj_slave
        self.ui.idx_master_obj = c.idx_master
        self.ui.idx_frame = c.idx_frame
        self.ui.idx_time_kvil = c.idx_time

        self.ui.max_pts_master = self._objects[c.idx_master].n_candidates_kvil
        self.ui.max_pts_raw = self._objects[c.idx_master].n_candidates_raw
        self.ui.max_time_kvil = self._objects[c.idx_master].n_time
        self.ui.max_time_raw = self._objects[c.idx_master].n_time_raw

        n_c = self.ui.n_c = len(self.constraints_index)
        self.ui.c_scale = np.ones(n_c).astype(float).tolist()
        self.ui.c_scale_changed = np.zeros(n_c).astype(bool).tolist()
        self.ui.c_alpha = np.ones(n_c).astype(float).tolist()
        self.ui.c_update_geom = np.zeros(n_c).astype(bool).tolist()
        self.ui.c_enable = np.zeros(n_c).astype(bool).tolist()
        self.ui.c_enable[c.idx_c] = True
        self.ui.c_enable_changed = np.zeros(n_c).astype(bool).tolist()
        self.ui.c_color_changed = np.zeros(n_c).astype(bool).tolist()
        self.ui.c_color_value = [tuple(colors.yellow_process)] * n_c
        for id_c in range(n_c):
            c = self.constraints[id_c]
            self.ui.c_select_text.append(f"{id_c:>02d} {c.type}: {c.obj_master} -> {c.obj_slave} t{c.idx_time}")

        for obj_name in self.obj_name_list:
            self.ui.obj_pcl[obj_name] = []

        for _obj in self._objects:
            # if _obj.c.is_virtual:
            self.ui.o_virt_alpha[_obj.c.name] = 1.0
            self.ui.o_virt_alpha_changed[_obj.c.name] = False
