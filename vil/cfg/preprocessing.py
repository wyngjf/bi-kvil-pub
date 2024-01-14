import copy
import numpy as np
from pathlib import Path
from typing import Union, List, Dict, Any

import torch
from marshmallow_dataclass import dataclass
from robot_utils import console
from robot_utils.py.interact import ask_checkbox_with_all
from robot_utils.py.filesystem import validate_path, create_path, get_ordered_subdirs, validate_file
from robot_utils.serialize.schema_numpy import NumpyArray, DictNumpyArray
from robot_utils.serialize.schema_torch import TorchTensor, DictTorchTensor
from robot_utils.serialize.dataclass import load_dataclass, load_dict_from_yaml, default_field
from robot_utils.pkg.pkg_dep_graph import get_install_order
from robot_vision.dataset.stereo.zed.type import ZEDCameraParam
from robot_vision.dataset.azure_kinect.type import AzureKinectCameraParam
from robot_vision.utils.utils import get_default_checkpoints_path
from vil.utils.utils import get_vil_path


@dataclass
class DepthParserConfig(ZEDCameraParam):
    mode:           str = "stereo"


@dataclass
class ClipCfg:
    s:              int = None
    e:              int = None


@dataclass
class VideoClipConfig:
    video:          ClipCfg = ClipCfg()
    occlude:        Dict[str, ClipCfg] = default_field({})


@dataclass
class SegmentationConfig:
    manual:         Dict[str, VideoClipConfig] = default_field({})


@dataclass
class DemoInfo:
    folder_name:    str = ""
    object_id:      str = ""
    record_info:    str = ""
    s:              float = None
    e:              float = None
    occluded_obj:   str = ""
    ot:             float = None


class DemoFileStructure:
    """
    - root: task path
        - canonical
            - dcn
                - obj1
                    - uv.yaml
                    - intrinsics.yaml
                    - rgb.png
                    - depth.png
                    - mask.png
                    - overlay.png
                    - mesh (mesh vertex, etc)
                    - sdf
                    - ...
                - obj2
                - ...
            - ndf
            - ...
        - recordings
            - trail1
                - images    [ (stereo: left_xxxxxx.png, right_xxxxxx.png) (mono: xxxxxx.png) ]
                - video
                - param.yaml
                - scene_graph.yaml
                - namespace1:
                    - rgb       [ down-sampled rgb images (left view if stereo): xxxxxx.png ]
                    - depth     [ corresponds to rgb, also visible depth images for human ]
                    - mask      [ all, obj1, obj2, ... ]
                    - human     [ some model may only have holistic body model ]
                        - identity.yaml  [ info about handedness, belonging of each hands, identity of person, etc ]
                        - hand_00
                            - mp_2d.csv
                            - mp_3d.csv
                            - patch     [ xxxxxx.png ]
                            - kpt.csv   [ (graphormer: the original est, without using base point) ]
                            - mesh      [ xxxxxx.pth ???? ]
                        - body_0x
                            - patch
                            - kpt.csv
                            - mesh
                        - ...
                    - obj
                        - detection
                            - grounded_sam.pth
                        - obj1
                            - uv.csv
                            - kpt.csv
                            - mesh      [ xxxxxx.pth ???? ] (later for BundleSDF)
                        - obj2
                        - ...
                - namespace2
                - ...
            - trial2
            - ...
        - viz: [ the same filestructure as above ]
            - namespace1
                - trial1
                    - rgb           [ prefix_rgb.mp4 ]
                    - depth         [ prefix_depth.mp4 ]
                    - mask          [ prefix_all.mp4 ]
                    - human
                        - hand_00
                            - patch         [ prefix_hand_00_patch.mp4 ]
                            - kpt
                                - overlay   [ xxxxxx.png ]
                                - video     [ prefix_hand_00_kpt.mp4 ]
                            - mesh
                                - overlay   [ xxxxxx.png ]
                                - video     [ prefix_hand_00_mesh.mp4 ]
                        - body_0x: similar to hand above
                        - ...
                        - all [ prefix_hand.mp4, prefix_body.mp4, etc (colorize identity) ]
                    - obj
                        - obj1
                            - overlay       [ xxxxxx.png ]
                            - video         [ prefix_obj1_uv.mp4, prefix_obj1_mesh.mp4 ]
                        - obj2
                        - ...
                        - all [ prefix_obj_uv.mp4, prefix_obj_mesh.mp4, etc (colorize identity) ]
                - trial2
                - ...
            - namespace2
            - ...
    """
    def __init__(self, task_path: Path, namespace: str):
        self.task_path = validate_path(task_path, throw_error=True)[0]
        self.rec: Path = validate_path(task_path / "recordings", throw_error=True)[0]
        self.can: Path = validate_path(task_path / "canonical", throw_error=True)[0]
        self.ns_path = create_path(task_path / namespace)
        self.namespace = namespace

        self._select_trails()
        self.n_trials: int = len(self.trial_dirs)
        self._trial_idx: int = 0
        self._reset()

    def _select_trails(self):
        trail_paths = get_ordered_subdirs(validate_path(self.rec, throw_error=True)[0])
        to_select_from = [p.stem for p in trail_paths]
        selected_trials = ask_checkbox_with_all("Select demos to proceed", to_select_from)

        data_path = create_path(self.ns_path / "data")
        viz_path = create_path(self.ns_path / "viz")
        vid_path = create_path(self.ns_path / "video")

        self.trial_dirs: List[Path] = [trail_paths[to_select_from.index(i)] for i in selected_trials]
        self.trial_dirs_data: List[Path] = [data_path / trial for trial in selected_trials]
        self.trial_dirs_viz: List[Path] = [viz_path / trial for trial in selected_trials]
        self.trial_dirs_vid: List[Path] = [vid_path / trial for trial in selected_trials]

    @property
    def trial_idx(self):
        return self._trial_idx

    @trial_idx.setter
    def trial_idx(self, value: int):
        if value < 0 or value >= self.n_trials:
            raise ValueError(f"the trial idx {value} should be in range [0, {self.n_trials}-1]")
        self._trial_idx = value
        self._reset()

    def _reset(self):
        self.t_dir = self.trial_dirs[self._trial_idx]
        self.t_dir_data = self.trial_dirs_data[self._trial_idx]
        self.t_dir_viz = self.trial_dirs_viz[self._trial_idx]
        self.t_dir_vid = create_path(self.trial_dirs_vid[self._trial_idx])

        # trial paths
        self.t_param:       Path = self.t_dir / "param.yaml"
        self.t_seg:         Path = self.t_dir / "seg.yaml"
        self.t_scene:       Path = self.t_dir / "scene_graph.yaml"
        self.t_images:      Path = create_path(self.t_dir / "images")
        self.t_rgb:         Path = create_path(self.t_dir_data / "rgb")
        self.t_flow:        Path = create_path(self.t_dir_data / "flow")
        self.t_dcn:         Path = create_path(self.t_dir_data / "dcn")
        self.t_depth:       Path = create_path(self.t_dir_data / "depth")
        self.t_depth_mono:  Path = create_path(self.t_dir / "depth")
        self.t_mask:        Path = create_path(self.t_dir_data / "mask")
        self.t_human:       Path = create_path(self.t_dir_data / "human")
        self.t_obj:         Path = create_path(self.t_dir_data / "obj")
        self.t_legacy_kvil_path = create_path(self.t_dir_data / "results")

        # viz path
        self.v_rgb:         Path = create_path(self.t_dir_viz / "rgb")
        self.v_depth:       Path = create_path(self.t_dir_viz / "depth")
        self.v_mask:        Path = create_path(self.t_dir_viz / "mask")
        self.v_human:       Path = create_path(self.t_dir_viz / "human")
        self.v_obj:         Path = create_path(self.t_dir_viz / "obj")
        self.v_flow:        Path = create_path(self.t_dir_viz / "flow")
        self.v_dcn:         Path = create_path(self.t_dir_viz / "dcn")


class KVILPerceptionConfig:
    def __init__(
            self,
            task_path: Union[str, Path],
            target_frames: int = 30,
            namespace: str = "default",
            viz: bool = False,
            viz_demo: bool = False,
            auto_create_canonical_space: bool = True,
            create_legacy_kvil_canonical: bool = False,
            to_video: bool = False,
            mono: bool = False,
    ):
        self.task_path = validate_path(task_path, throw_error=True)[0]
        if not namespace:
            raise ValueError("you have to specify a namespace '-n NAMESPACE'")
        self.namespace = namespace
        self.p = DemoFileStructure(self.task_path, namespace)
        self.target_frames = target_frames
        self.viz = viz
        self.viz_demo = viz_demo
        self.to_video = to_video
        self.auto_can_space = auto_create_canonical_space
        self.create_legacy_kvil_canonical = create_legacy_kvil_canonical
        self.mono = mono

        self.checkpoints_root_dir = get_default_checkpoints_path()
        self._determine_process_order()

    def _determine_process_order(self):
        self.step_dict = load_dict_from_yaml(get_vil_path() / "cfg/preprocess_step_deps.yaml")
        step_list = [(k, v) for (k, v) in self.step_dict.items()]
        self.steps_to_update = ask_checkbox_with_all("select steps that you want to update", get_install_order(step_list))
        dep_dict = {}

        def add_deps(key_list: List[str]):
            for key in key_list:
                if key not in dep_dict:
                    dep_dict[key] = self.step_dict[key]
                    add_deps(self.step_dict[key])

        add_deps(self.steps_to_update)
        console.rule()
        self.process_order = get_install_order([(k, v) for (k, v) in dep_dict.items()])
        console.log(f"[bold cyan]Processing according to order [bold green]{' -> '.join(self.process_order)}")
        console.log(f"[bold cyan]Force update steps:  [bold green]{' -> '.join([s for s in self.process_order if s in self.steps_to_update])}")
        console.rule()

    def reset_trial(self, idx: int):
        self.p.trial_idx = idx
    #     self._load_cam_intrinsics()
    #
    # def _load_cam_intrinsics(self):
    #     self.cam = load_dataclass(DepthParserConfig, self.p.t_param)
    #
    # def _load_seg_cfg(self):
    #     self.seg = load_dataclass(SegmentationConfig, self.p.t_seg).manual[self.namespace]
    #     if self.seg.s is None or self.seg.e is None:
    #         console.log(
    #             f"[bold red]Missing starting/ending frame/time in {self.p.t_seg}, "
    #             f"you should add e.g. s: 123, e: 456")
    #         raise RuntimeError


@dataclass
class SceneGraphConfig:
    obj_cfg:            Dict[str, str] = None
    erode:              Dict[str, int] = None
    action:             List[Dict[str, List[str]]] = None
    spatial_relation:   List[Any] = None


@dataclass
class CanonicalSpace:
    dcn_obj_cfg:        Dict[str, str] = default_field({})
    dcn_image_wh:       Dict[str, List[int]] = default_field({})
    num_candidates:     Dict[str, int] = default_field({})
    rgb:                Dict[str, NumpyArray] = default_field({})  # color image (h, w, 3)
    depth:              Dict[str, NumpyArray] = default_field({})  # depth map (h, w)
    mask:               Dict[str, NumpyArray] = default_field({})  # binary mask (h, w)
    intrinsics:         Dict[str, NumpyArray] = default_field({})  # intrinsic parameter of the corresponding images
    shape:              Any = None                    # some kind of shape, mesh, sdf or whatever
    uv:                 Dict[str, NumpyArray] = default_field({})  # uv coordinates in image frame of the candidate points (N, 2)
    uv_colors:          Dict[str, NumpyArray] = default_field({})  # uv coordinates in image frame of the candidate points (N, 2)
    cand_pt:            Dict[str, NumpyArray] = default_field({})  # Candidate points (N, 3)
    pcl:                Dict[str, NumpyArray] = default_field({})  # object point cloud recovered from the mask and depth map (n, 3)
    descriptor:         Dict[str, TorchTensor] = default_field({})  # the dcn descriptors (N, dim_dcn)

    def load_dcn_cfg(self, filename: Path):
        validate_file(filename, throw_error=True)
        data = load_dict_from_yaml(filename)
        self.dcn_obj_cfg = data.get("dcn_obj_cfg", None)
        self.num_candidates = data.get("num_candidates", None)
        if self.dcn_obj_cfg is None or self.num_candidates is None:
            raise RuntimeError("the dcn configuration for the canonical space is not correct, please check "
                               f"file {filename}")


@dataclass
class HumanProperty:
    left_hand_idx: int = None
    right_hand_idx: int = None
    height: float = None


@dataclass
class HumanID:
    humans:             List[HumanProperty] = default_field([])


@dataclass
class RTMPoseHumanData:
    keypoints:          List[NumpyArray] = default_field([])            # T elements: (113, 2)
    keypoint_scores:    List[NumpyArray] = default_field([])            # T elements: (113, )
    bboxs:              List[NumpyArray] = default_field([])            # T elements: (n_bbox, 4)
    hand_landmark_l:    List[NumpyArray] = default_field([])            # T elements: (21, 2)
    hand_landmark_r:    List[NumpyArray] = default_field([])            # T elements: (21, 2)
    hand_time_idx_l:    List[int] = default_field([])                   # (T, )
    hand_time_idx_r:    List[int] = default_field([])                   # (T, )


@dataclass
class HumanDataList:
    data:               List[RTMPoseHumanData] = default_field([])


@dataclass
class HumanData:
    n_hands:            int = 0
    hand_time_masks:    DictNumpyArray = default_field(DictNumpyArray())  # n_hands, len(array) = T_i
    hand_num_list:      List[int] = default_field([])
    handedness:         List[str] = default_field([])  # len() = n_hands, either 'left' or 'right'
    mp_hand_landmarks:  DictNumpyArray = default_field(DictNumpyArray())  # n_hands Dict[hand, (T_i, 21, 2)]
    mp_hand_crop_bbox:  DictNumpyArray = default_field(DictNumpyArray())  # n_hands Dict[hand, (T_i, 4)]
    # mp_hand_crop_img:   np.ndarray = None  # (n_hands, T, 224, 224, 3)
    graphormer_uv:      DictNumpyArray = default_field(DictNumpyArray())  # (n_hands, T, 21, 2)
    graphormer_xyz:     DictNumpyArray = default_field(DictNumpyArray())  # (n_hands, T, 21, 3)
    human_id:           HumanID = default_field(HumanID())
    rtm_humans:         HumanDataList = default_field(HumanDataList())


@dataclass
class KVILDemoData:
    target_frames:      int = -1
    rgb_files_l:        List[Path] = None
    rgb_files_r:        List[Path] = None
    rgb_files_mono:     List[Path] = None
    rgb_files:          List[Path] = None
    rgb_down_sample_idx: List[int] = None
    rgb_array:          List[np.ndarray] = None

    depth_files:        List[Path] = None
    depth_files_raw:    List[Path] = None  # Note: only needed if you use depth camera
    depth_array:        List[np.ndarray] = None

    scene_graph:        SceneGraphConfig = None

    cam:                Union[ZEDCameraParam, DepthParserConfig] = None
    vid_clip:           VideoClipConfig = None
    human:              HumanData = default_field(HumanData())

    grounded_sam:       Dict[str, Any] = None
    mask_dict:          Dict[str, List[np.ndarray]] = default_field({})

    flow:               DictNumpyArray = default_field(DictNumpyArray())    # key: (h, w, 2)
    dcn_uv:             DictNumpyArray = default_field(DictNumpyArray())    # key: (300, 2)
    uv:                 DictNumpyArray = default_field(DictNumpyArray())    # key: (T, 300, 2)
    xyz:                DictNumpyArray = default_field(DictNumpyArray())    # key: (T, 300, 3)
    filtered_xyz:       DictNumpyArray = default_field(DictNumpyArray())    # key: (T, 300, 3) trajctories filtered by a sg filter
    fixed_xyz:          DictNumpyArray = default_field(DictNumpyArray())    # key: (T, 300, 3) trajctories fixed by estimate the outlier's depth from inlier neighbors

    def load_scene_graph(self, filename: Path):
        validate_file(filename, throw_error=True)
        self.scene_graph = load_dataclass(SceneGraphConfig, filename)

    def load_cam_intrinsics(self, filename: Path, is_mono: bool):
        validate_file(filename, throw_error=True)
        DataClass = AzureKinectCameraParam if is_mono else DepthParserConfig
        self.cam = load_dataclass(DataClass, filename)

    def load_segmentation(self, filename: Path, namespace: str):
        validate_file(filename, throw_error=True)
        self.vid_clip = load_dataclass(SegmentationConfig, filename).manual[namespace]
        # if self.vid_clip.video.s is None or self.vid_clip.video.e is None:
        #     console.log(
        #         f"[bold red]Missing starting/ending frame/time in segmentation config, "
        #         f"you should add e.g. s: 123, e: 456")
        #     raise RuntimeError


@dataclass
class KVILPreProcessingData:
    can:        CanonicalSpace = default_field(CanonicalSpace())
    demos:      List[KVILDemoData] = default_field([])
