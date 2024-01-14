import polyscope as ps
import numpy as np
import rich.box
from pathlib import Path
from rich.table import Table
from typing import List, Dict, Union, Tuple, Set
from marshmallow_dataclass import dataclass
from robot_utils.serialize.dataclass import default_field, dump_data_to_yaml, load_dataclass
from robot_utils.serialize.path import PathT
from robot_utils.py.filesystem import get_validate_path
from robot_utils import console


@dataclass
class KVILConfig:
    # scene and object config
    n_neighbors:                        int = 80
    n_neighbors_for_origin:             int = 25
    n_candidates:                       int = 300
    n_time_steps_raw:                   int = 30
    n_time_frames:                      int = 2         # the total number of frames should be considered by K-VIL

    remove_outlier:                     bool = True
    last_timestep:                      bool = True
    enable_grasping:                    bool = True

    # PCE config
    th_vari_low:                        float = 0.1
    th_linear_lv:                       float = 0.02
    th_linear_hv:                       float = 0.2
    th_curvature_hv:                    float = 2.0
    th_curvature_lv:                    float = 0.3
    th_time_cluster:                    float = 0.1
    th_dist_ratio:                      float = 0.4


class KVILSceneConfig:
    obj:                                List[str] = default_field([])
    demo_paths:                         List[str] = default_field([])
    scale:                              List[float] = None
    mask_model:                         str = None


class DeployUIConfig:
    def __init__(self):
        self.all_constraint_found:      bool = False
        self.found_constraint_idx:      List[int] = []
        self.c_refresh_flag:            bool = False
        self.c_keypoints:               Dict[int, List[ps.PointCloud]] = {}
        self.c_keypoints_data:          Dict[int, np.ndarray] = {}          # each (3, ) in corresponding local frame

        self.obj_can_pcl:               Dict[int, ps.PointCloud] = {}

        self.c_mp_traj:                 Dict[int, List[ps.CurveNetwork]] = {}
        self.c_mp_traj_data:            Dict[int, np.ndarray] = {}          # each (100, 3) in corresponding local frame
        self.c_traj_flag:               bool = False
        self.c_traj_flag_changed:       bool = False
        self.c_traj_alpha:              float = 1.0
        self.c_traj_alpha_changed:      bool = False
        self.c_traj_scale:              float = 1.0
        self.c_traj_scale_changed:      bool = False

        self.c_enabled:                 Dict[int, bool] = {}
        self.c_enabled_changed:         Dict[int, bool] = {}
        self.c_scale_value:             Dict[int, float] = {}  # (n_c, )
        self.c_scale_changed:           Dict[int, bool] = {}   # (n_c, )
        self.c_color_changed:           Dict[int, bool] = {}   # (n_c, )
        self.c_color_value:             Dict[int, Tuple[float, float, float, float]] = {}

        self.c_frame_neighbors:         Dict[int, ps.PointCloud] = {}
        self.c_frame_neighbors_data:    Dict[int, np.ndarray] = {}  # in camera local frame
        self.c_frame_neighbors_changed: Dict[int, bool] = {}
        self.c_frame_neighbors_enabled: Dict[int, bool] = {}

        self.scene_seg_pcl:             Dict[str, ps.PointCloud] = {}
        self.scene_tcp_frame:           Dict[str, ps.PointCloud] = {}
        self.enable_tcp_frame:          bool = False
        self.enable_tcp_frame_changed:  bool = False

        self.ex_mp_traj:                Dict[int, ps.CurveNetwork] = {}
        self.ex_mp_target:              Dict[int, ps.PointCloud] = {}


class UIConfig:
    def __init__(self):
        # polyscope param
        self.frame_scale:               float = 0.1
        self.frame_radius:              float = 0.002
        self.frame_alpha:               float = 1.0

        self.curve_radius:              float = 0.002

        self.point_radius:              float = 0.003
        self.point_alpha:               float = 0.8

        # control
        self.first_run:                 bool = True
        self.obj_name_master:           str = ""
        self.obj_master_changed:        bool = False
        self.obj_name_slave:            str = ""

        self.idx_master_obj:            int = 0
        self.idx_slave_obj:             int = 0

        self.max_pts_raw:               int = 0
        self.max_pts_master:            int = 0
        self.max_pts_slave:             int = 0

        self.max_time_kvil:             int = 0
        self.idx_time_kvil:             int = 0
        self.idx_time_changed:          bool = False

        self.max_time_raw:              int = 0
        self.idx_time_raw:              int = 0

        self.idx_demo:                  int = 0

        self.idx_frame:                 int = 0
        self.frame_changed:             bool = False
        self.idx_kpts:                  int = 0

        self.alpha_master:              float = 1.0
        self.alpha_master_changed:      bool = False
        self.alpha_slave:               float = 1.0
        self.alpha_slave_changed:       bool = False

        # constraints
        self.c_force_redraw:            bool = False
        self.c_select_text:             List[str] = []
        self.c_selected_idx:            int = 0
        self.c_selected_changed:        bool = False
        self.n_c:                       int = 0
        self.c_scale:                   List[float] = []  # (n_c, )
        self.c_scale_changed:           List[bool] = []   # (n_c, )

        self.c_pca_scale:               float = 1.0
        self.c_pca_scale_changed:       bool = False
        self.c_pca_revert:              bool = False
        self.c_pca_revert_changed:      bool = False

        self.c_pme_scale:               float = 1.0
        self.c_pme_scale_changed:       bool = False

        self.c_alpha:                   List[float] = []  # (n_c, )
        self.c_alpha_changed:           List[bool] = []   # (n_c, )
        self.c_update_geom:             List[bool] = []   # (n_c, )
        self.c_enable:                  List[bool] = []   # (n_c, )
        self.c_enable_changed:          List[bool] = []   # (n_c, )
        self.c_color_changed:           List[bool] = []   # (n_c, )
        self.c_color_value:             List[Tuple[float, float, float, float]] = []

        # geometries
        self.origin:                    ps.PointCloud = None
        self.o_scale:                   float = 1.0
        self.o_scale_changed:           bool = False
        self.o_radius:                  float = 1.0
        self.o_radius_changed:          bool = False
        self.o_alpha:                   float = 0.7
        self.o_alpha_changed:           bool = False
        self.obj_pcl:                   Dict[str, List[Union[ps.PointCloud, ps.CurveNetwork]]] = {}
        self.o_virt_alpha:              Dict[str, float] = {}
        self.o_virt_alpha_changed:      Dict[str, bool] = {}
        self.o_virt_enabled:            Dict[str, bool] = {}

        self.c_keypoints:               Dict[int, Dict[int, ps.PointCloud]] = {}
        # (n_c, ) each element is a list (n_demo, )

        # self.c_geom_shape:              List[List[Union[None, ps.PointCloud, ps.CurveNetwork]]] = []
        self.c_geom_shape:              Dict[int, Dict[int, Union[None, ps.PointCloud, ps.CurveNetwork]]] = {}
        # (n_c, ) each element is a list of the shape of the constraint of size (n_demo, ).
        # -- for p2p: a point
        # -- for p2l/P: a local frame as the result of PCA, include the mean and the direction
        # -- for p2c: the mean
        self.c_stress_vector:           Dict[int, Dict[int, ps.PointCloud]] = {}
        # idx: index of constraint: -> dict
        # -- idx: index of demo -> point cloud with vector field for the stress vector

        self.c_traj_ready:              bool = False
        self.c_traj_flag:               bool = False
        self.c_traj_flag_changed:       bool = False
        self.c_traj_alpha:              float = 1.0
        self.c_traj_alpha_changed:      bool = False
        self.c_traj_scale:              float = 1.0
        self.c_traj_scale_changed:      bool = False
        self.c_traj:                    Dict[int, Dict[int, Union[None, ps.CurveNetwork]]] = {}
        self.c_traj_frame:              Dict[int, Dict[int, Union[None, ps.PointCloud]]] = {}

        # for density functions
        self.c_pdf_ready:               bool = False
        # c_pdf_vertices:               List[np.ndarray] = []  # mesh vertices: each (V, 3) in local frame
        # c_pdf_indices:                List[np.ndarray] = []  # mesh vertices: each (I, 2 or 3) in local frame
        # the indices and be (I, 2) for CurveNetwork (p2l, p2c) or (I, 3) for SurfaceMesh (p2P, p2S)
        self.c_pdf_flag:                bool = False
        self.c_pdf_flag_changed:        bool = False
        self.c_pdf_alpha:               float = 1.0
        self.c_pdf_alpha_changed:       bool = False
        self.c_pdf_radius:              float = 1.0
        self.c_pdf_radius_changed:      bool = False
        self.c_pdf_span:                float = 1.0
        self.c_pdf_span_changed:        bool = False
        self.c_pdf_scale:               float = 1.0
        self.c_pdf_scale_changed:       bool = False
        self.c_pdf_base:                Dict[int, Dict[int, Union[None, ps.CurveNetwork, ps.SurfaceMesh, ps.VolumeMesh]]] = {}
        self.c_pdf_func:                Dict[int, Dict[int, Union[None, ps.CurveNetwork, ps.SurfaceMesh, ps.VolumeMesh]]] = {}

        self.c_frame_neighbors:         Dict[int, Dict[int, ps.PointCloud]] = {}
        self.c_frame_neighbors_changed: Dict[int, bool] = {}
        self.c_frame_neighbors_enabled: Dict[int, bool] = {}


@dataclass
class ObjectConfig:
    name:                               str = None
    index:                              int = 0
    is_master:                          bool = False
    is_virtual:                         bool = False  # virtual object must also be static
    is_hand:                            bool = False
    is_static:                           bool = False  # if object is always static
    spatial_scale:                      float = 0
    hierarchical_lvl:                   int = -1
    original_obj:                       str = None

    appended_obj:                       List[str] = default_field([])
    appended_idx:                       List[int] = default_field([])
    slave_idx:                          List[int] = default_field([])
    slave_obj:                          List[str] = default_field([])
    slave_scale:                        Dict[str, float] = default_field({})
    # slave_variation:                    Dict[str, Dict[str, bool]] = default_field({})
    # idx-1: object name, idx-2: idx_time_cluster
    associated_master_idx:              Set[int] = default_field(set())

    constraint_path:                    str = ""
    constraint_files:                   List[str] = default_field([])
    kvil_process_data_file:             str = ""

    path:                               PathT = None
    force_redo:                         bool = False
    traj_file_list:                     List[str] = None
    viz_debug:                          bool = False

    kvil:                               KVILConfig = None

    # (bimanual) coordination
    is_grasped:                         bool = False 
    grasping_hand:                      str = None
    grasping_hand_idx:                  List[int] = default_field([])
    grasping_point_idx:                 Dict[int, int] = default_field({})
    is_symmetric:                       bool = False 
    symmetric_hand_idx:                 List[int] = default_field([])

    # UI
    frame_scale:                        float = 0.1
    frame_radius:                       float = 0.01
    frame_alpha:                        float = 1.0

    point_radius:                       float = 0.01
    point_alpha:                        float = 0.8

    def is_slave(self):
        return len(self.associated_master_idx) > 0

    def set_master(self, master_idx: int):
        """
        append a master's index
        """
        self.associated_master_idx.add(master_idx)

    def unset_master(self, master_idx: int):
        """
        if there is no constraint between a master and this object,
        then remove the master's index from self.associated_master_idx
        """
        if master_idx in self.associated_master_idx:
            self.associated_master_idx.remove(master_idx)

    def is_real_obj(self):
        return not self.is_hand and not self.is_virtual

    def with_path(self, path):
        self.path = get_validate_path(path)
        return self

    def as_virtual(self):
        if not self.name:
            raise ValueError("you have to first set name, then as_virtual")
        if self.name.split("_")[0] != "virtual":
            self.name = f"virtual_{self.name}"
        self.is_virtual = self.is_static = self.is_master = True
        self.is_hand = False
        self.hierarchical_lvl = 0
        return self

    def as_master(self):
        self.is_master = True
        self.is_static = True
        self.hierarchical_lvl = 0
        return self

    def with_name(self, name: str):
        self.name = self.original_obj = name
        self.is_hand = name in ["left_hand", "right_hand"]
        return self

    def with_index(self, idx: int):
        self.index = idx
        return self


@dataclass
class Constraints:
    idx_c:                              int = 0
    idx_c_on_master:                    int = 0
    type:                               str = "p2p"
    priority:                           int = 0

    idx_master:                         int = 0
    idx_slave:                          int = 0
    obj_master:                         str = ""
    obj_slave:                          str = ""
    obj_slave_scale:                    float = 1.0

    idx_time:                           int = 0
    idx_keypoint:                       int = 0
    kpt_descriptor:                     List[float] = default_field([])        # (D, )
    kpt_uv:                             List[float] = default_field([])

    idx_frame:                          int = 0
    frame_neighbor_index:               List[int] = default_field([])          # (Q, )
    frame_neighbor_coordinates:         List[List[float]] = default_field([])  # (Q, 3)
    frame_neighbor_descriptors:         List[List[float]] = default_field([])  # (Q, D)
    frame_neighbor_uvs:                 List[List[float]] = default_field([])  # (Q, 2)

    pca_data:                           list = None  # (d+2, d) in master local frame

    pm_sol_opt_x:                       list = None  # in master local frame
    pm_sol_opt_t:                       list = None
    pm_t_opt:                           list = None
    pm_embedding:                       list = None
    pm_projection:                      list = None
    pm_mean:                            list = None
    pm_stress_vectors:                  list = None
    pm_std_stress:                      float = 0.0
    pm_std_projection:                  float = 0.0
    pm_intrinsic_dim:                   int = 0
    pm_scaling:                         float = 1.0   # multiply this to the result of PME

    pm_imit_kpts_position_local:        list = None
    pm_init_proj_point:                 list = None
    pm_init_tangent_vec:                list = None
    pm_init_proj_value:                 float = 0.0

    # the following are for deployment
    vmp_kernel:                         int = 20    # in master local frame
    vmp_dim:                            int = 3
    vmp_weights:                        list = None
    vmp_traj_files:                     list = None
    vmp_goal:                           list = None
    vmp_demo_traj:                      list = default_field([])

    kpts_obs_local:                     list = None  # in master local frame
    vmp_start:                          list = None  # in master local frame
    local_frame_rot_mat_obs:            list = None  # in robot root frame
    local_frame_origin_obs:             list = None  # in robot root frame

    density_mu:                         list = None  # in master local frame
    density_weights:                    list = None
    density_std_dev:                    float = 0.0

    def __repr__(self):
        return f"Constraint {self.idx_c+1:>2d} type {self.type:>4}, " \
               f"{self.idx_frame+1:>3d}-th local frame defined on {self.obj_master:>10}, " \
               f"  with {self.idx_keypoint+1:>3d}-th keypoints on {self.obj_slave:>10}," \
               f"  restricted at time step {self.idx_time+1:>3d}."

    def save(self, proc_path: Path):
        dump_data_to_yaml(
            Constraints,
            self, proc_path / f"constraints/{self.obj_master}/c_{self.obj_slave}_{self.idx_c_on_master:>02d}.yaml"
        )

    def load(self, proc_path: Path):
        return load_dataclass(Constraints, proc_path / f"constraints/{self.obj_master}/c_{self.obj_slave}_{self.idx_c_on_master:>02d}.yaml")


@dataclass
class MasterSlaveHierarchy:
    master_obj:                         List[str] = default_field([])
    master_obj_idx:                     List[int] = default_field([])
    slave_obj:                          Dict[str, List[str]] = default_field({})
    slave_obj_idx:                      Dict[str, List[int]] = default_field({})
    constraints:                        Dict[str, Dict[str, List[Constraints]]] = default_field({})  # key master/slave name
    n_constraint:                       int = 0
    constraint_max_t:                   int = 0
    ignore_idx:                         List[int] = default_field([])
    filename:                           str = ""
    # TODO
    # constraint_list:                    List[Constraints] = default_field([])
    t_m_c_order:                        Dict[int, Dict[str, List[int]]] = default_field({})
    # time0:                            # in total kvil_time keys
    # -- master0:                       # all objs that happen to be master at time0
    # -- -- c0                          # all constraints on master0 at time0 (List[int])
    # ...

    def save(self):
        dump_data_to_yaml(MasterSlaveHierarchy, self, self.filename)

    def show(self):
        table = Table(title="[bold yellow]Hybrid Master-Slave Relationship (HMSR)", box=rich.box.HORIZONTALS)
        table.add_column("master object", justify="right", style="cyan")
        table.add_column("idx", justify="left", style="green")
        table.add_column("slave object", justify="right", style="yellow")
        table.add_column("idx", justify="left", style="green")

        for idx_master, master_name in zip(self.master_obj_idx, self.master_obj):
            for i, (idx_slave, slave_name) in enumerate(zip(self.slave_obj_idx[master_name], self.slave_obj[master_name])):
                if i == 0:
                    table.add_row(master_name, f"{idx_master}", slave_name, f"{idx_slave}")
                else:
                    table.add_row("", "", slave_name, f"{idx_slave}")
        console.log(table)


class KVILProcessData:
    def __init__(self):
        self.data_file:  str = ""
        self.appended_obj_traj:         Dict[str, np.ndarray] = {}  # array (N, T, Pm, Ps, d)
        self.pca_data:                  Dict[str, np.ndarray] = {}  # each array (T, Pm, Ps, d+2, d)
        self.last_pca_data:             Dict[str, np.ndarray] = {}  # each array (Pm, Ps, d+2, d)
        self.pme_data:                  Dict[str, np.ndarray] = {}
        self.mask:                      Dict[str, np.ndarray] = {}  # each array (T, Pm, Ps)
        self.c_traj_data:               Dict[int, np.ndarray] = {}  # index: idx_c

        self.reload_finished:           bool = False
        self.saved_version:             int = 0
        self.updated_version:           int = 0

    def save(self, message: str = "", force_save: bool = False):
        if self.saved_version >= self.updated_version and not force_save:
            console.log("-- saved version up-to-date, skip")
            return
        console.log(f"-- [blue]saving K-VIL processing data to [white]{self.data_file} {message}")
        np.savez(
            self.data_file,
            pca_data=self.pca_data,  # np convert dict to an array and store it, you need to parse it during load
            pme_data=self.pme_data,
            appended_obj_traj=self.appended_obj_traj,
            mask=self.mask,
            c_traj_data=self.c_traj_data,
        )
        self.saved_version = self.updated_version

    def load(self):
        if self.reload_finished:
            return
        console.log(f"[blue]loading K-VIL processing data from [white]{self.data_file}")
        data = dict(np.load(self.data_file, allow_pickle=True))
        data = {key: data[key].item() for key in data}  # restore the dict from the stored 0-dim array
        self.pca_data = data.get("pca_data", {})
        self.pme_data = data.get("pme_data", {})
        self.appended_obj_traj = data.get("appended_obj_traj", {})
        self.mask = data.get("mask", {})
        self.c_traj_data = data.get("c_traj_data", {})

        self.reload_finished = True
