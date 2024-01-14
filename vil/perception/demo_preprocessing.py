import os
import copy
import click
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from functools import partial
from rich.progress import track
from typing import List, Dict, Optional, Literal
from pathos.multiprocessing import ProcessingPool as Pool
from scipy.signal import savgol_filter
from scipy.optimize import linear_sum_assignment

import robot_utils.log as log
import robot_utils.viz.latex_colors_rgba as lc
from robot_utils import console
from robot_utils.cv.image.random_torch import random_uv_on_mask
from robot_utils.cv.geom.projection import pinhole_projection_image_to_camera
from robot_utils.cv.correspondence.viz import viz_points_with_color_map
from robot_utils.cv.correspondence.similarity import cosine_similarity
from robot_utils.cv.io.io import image_to_gif, image_to_video
from robot_utils.cv.io.io_cv import write_depth, write_binary_mask, write_rgb, write_colorized_depth
from robot_utils.cv.io.io_cv import load_depth, load_rgb, load_mask
from robot_utils.cv.opencv import overlay_masks_on_image, overlay_mask_on_image
from robot_utils.cv.correspondence.finder_torch import find_best_match_for_descriptor_cosine
from robot_utils.serialize.schema_numpy import DictNumpyArray, NumpyArray
from robot_utils.serialize.dataclass import save_to_yaml, load_dict_from_yaml, load_dataclass, dump_data_to_dict, dump_data_to_yaml
from robot_utils.py.filesystem import get_ordered_files, get_ordered_subdirs, copy2, create_path
from robot_utils.py.filesystem import validate_path, validate_file, get_validate_file
from robot_utils.py.interact import ask_list, ask_text

from robot_vision.uni_model.wrapper import estimate_depth_stereo
# from robot_vision.human_pose.mediapipe.utils.id_assignment import compute_assignment

from vil.cfg.preprocessing import KVILPerceptionConfig, KVILPreProcessingData, KVILDemoData
from vil.cfg.preprocessing import HumanDataList
import vil.utils.outlier.outlier_detection as outlier_det
from vil.constraints import find_neighbors


class KVILPreprocessing:
    def __init__(self, c: KVILPerceptionConfig):
        self.c = c

        self.process_map = dict(
            auto=self.dummy_func,
            extract=self._extract,
            down_sample=self._down_sample,
            depth=self._depth,
            mediapipe_hand=self._mediapipe_hand,
            mmpose_wholebody=self._mmpose_wholebody,
            mediapipe_handedness=self._mediapipe_handedness,
            crop_hand=self._crop_hand,
            mask=self._mask,
            graphormer_hand=self._graphormer_hand,
            canonical=self._create_dcn_canonical_space,
            dcn=self._dcn,
            random_dcn=self._random_dcn,
            opt_flow=self._opt_flow,
            opt_flow_raft=self._opt_flow_raft,
            point_track=self._point_track,
            obj_kpt_traj=self._obj_kpt_traj,
            filter_and_inlier_detect=self._filter_and_inlier_detect,
            inlier_detect_and_outlier_fix = self._inlier_detect_and_outlier_fix,
            convert_to_kvil=self._convert_to_kvil,
            to_video=self._to_video,
        )

        self.results = {}
        self.kvil_data = KVILPreProcessingData()
        self.d: Optional[KVILDemoData] = None
        self.pool = Pool(os.cpu_count())

    def _load_unimatch_wrapper(self, mode: Literal["stereo", "flow", "depth"] = "stereo"):
        if hasattr(self, "uni_wrapper"):
            if self.uni_wrapper.c.mode != mode:
                self.uni_wrapper.change_mode(mode)
                console.log(f"[bold green]change uni match mode to {mode}")
            return
        from robot_vision.uni_model.wrapper import UnimatchWrapper, UnimatchWrapperConfig
        console.log(f"[bold green]create unimatch wrapper with mode: {mode}")
        cfg = UnimatchWrapperConfig()
        cfg.mode = mode
        self.uni_wrapper = UnimatchWrapper(cfg)
        console.log("[bold cyan]done")

    def _load_sam_tracker(self):
        if hasattr(self, "mask_tracker"):
            return
        from robot_vision.sam_track.wrapper import SAMTracker
        self.mask_tracker = SAMTracker()

    def _load_opt_flow(self):
        if hasattr(self, "opt_flow"):
            return
        from robot_vision.opt_flow.raft.raft_wrapper import RAFTWrapper
        self.opt_flow = RAFTWrapper()

    def _load_point_tracker(self):
        if hasattr(self, "point_tracker"):
            return
        from robot_vision.point_tracking.pips.pips_wrapper import PointTrackWrapper
        self.point_tracker = PointTrackWrapper()

    def dummy_func(self):
        pass

    def _load_mask_wrapper(self):
        if hasattr(self, "mask_wrapper"):
            return
        from robot_vision.grounded_sam.wrapper import MaskWrapper, MaskWrapperConfig
        cfg = MaskWrapperConfig()
        cfg.training_root = self.c.checkpoints_root_dir
        cfg.sam.flag_sam_hq = False
        cfg.flag_highest_score_only = True

        log.disable_logging()
        self.mask_wrapper = MaskWrapper(cfg)
        log.enable_logging()

    def _load_mesh_graphormer(self):
        if hasattr(self, "mg_model"):
            return
        from robot_vision.human_pose.graphormer.mesh_graphormer_wrapper import MeshGraphormerWrapper
        log.disable_logging()
        self.mg_model = MeshGraphormerWrapper()
        log.enable_logging()

    def _load_dcn_model(self):
        if hasattr(self, "dcn"):
            return
        console.log("[bold yellow]loading DCN models")
        from robot_utils.torch.torch_utils import get_device
        from robot_vision.dcn.wrapper import DCNWrapper
        self.dcn = DCNWrapper(
            self.kvil_data.can.dcn_obj_cfg,
            self.c.checkpoints_root_dir / "dcn",
            get_device(use_gpu=True)
        )
        for obj in self.kvil_data.can.dcn_obj_cfg.keys():
            self.kvil_data.can.dcn_image_wh[obj] = self.dcn.dcn[
                obj].image_wh  # the image_wh has to match the one for training the DCN
            self.descriptor_dim = self.dcn.dcn[obj].dim
        self.transform = self.dcn.transform
        console.log(f"[bold cyan]DCN model loaded")

    def _extract(self):
        if self.c.mono:
            self._extract_mono()
        else:
            self._extract_stereo()

    def _extract_mono(self):
        console.rule(f"extract monocular recordings (Azure Kinect)")
        self.d.rgb_files_mono = get_ordered_files(self.c.p.t_images, pattern=[])
        self.d.depth_files_raw = get_ordered_files(self.c.p.t_depth_mono, pattern=[], ex_pattern=["v_"])
        n_rgb = len(self.d.rgb_files_mono)
        n_d = len(self.d.depth_files_raw)
        if "extract" not in self.c.steps_to_update and n_rgb == n_d > 0:
            console.log(f"[bold cyan]done (load)")
            return

        from robot_utils.sensor.azure_kinect.mkv_reader import ReaderWithCallback
        reader = ReaderWithCallback(mkv_dir=self.c.p.t_dir, output_path=self.c.p.t_dir)
        reader.run()
        console.log(f"[bold cyan]done")

    def _extract_stereo(self):
        console.rule(f"extract stereo recordings")
        self.d.rgb_files_l = get_ordered_files(self.c.p.t_images, pattern=["left_"])
        self.d.rgb_files_r = get_ordered_files(self.c.p.t_images, pattern=["right_"])
        n_l = len(self.d.rgb_files_l)
        n_r = len(self.d.rgb_files_r)
        if "extract" not in self.c.steps_to_update and n_l == n_r > 0:
            console.log(f"[bold cyan]done (load)")
            return

        from robot_vision.dataset.stereo.zed.exporter import export
        validate_file(self.c.p.t_dir / "recording.svo", throw_error=True)
        export(self.c.p.t_dir / "recording.svo", mode=2, tn_frames=-1)
        console.log(f"[bold cyan]done")

    def _down_sample(self):
        console.rule(f"down sample")
        self.d.target_frames = min(self.c.target_frames, self.d.vid_clip.video.e - self.d.vid_clip.video.s)
        self.d.rgb_files = get_ordered_files(self.c.p.t_rgb, pattern=[".jpg"])
        _loaded: bool = False

        if len(self.d.rgb_files) == self.d.target_frames:
            _loaded = True
            idx_list = load_dict_from_yaml(self.c.p.t_rgb / "down_sample_index.yaml")
            self.d.rgb_down_sample_idx = idx_list
        else:
            idx_list = np.linspace(self.d.vid_clip.video.s, self.d.vid_clip.video.e, self.d.target_frames, dtype=int) - 1
            self.d.rgb_down_sample_idx = idx_list.tolist()
            save_to_yaml(self.d.rgb_down_sample_idx, self.c.p.t_rgb / "down_sample_index.yaml")
            self.d.rgb_files = [self.c.p.t_rgb / f"{i:>06d}.jpg" for i in idx_list]

        _loaded_depth: bool = False
        if self.c.mono:
            self.d.depth_files = get_ordered_files(self.c.p.t_depth, pattern=[".png"], ex_pattern=["v_"])
            if len(self.d.depth_files) == self.d.target_frames:
                _loaded_depth = True
            else:
                self.d.depth_files = [self.c.p.t_depth / f"{i:>06d}.png" for i in idx_list]

        stem = self.d.rgb_files_mono[0].parent if self.c.mono else self.d.rgb_files_l[0].parent
        prefix = "" if self.c.mono else "left_"
        self.d.rgb_files_l = [stem / f"{prefix}{idx:>06d}.jpg" for idx in idx_list]

        if not self.c.mono:
            self.d.rgb_files_r = [stem / f"right_{idx:>06d}.jpg" for idx in idx_list]

        pool = Pool(os.cpu_count())

        if not _loaded:
            pool.map(copy2, self.d.rgb_files_l, self.d.rgb_files)

        if self.c.mono and not _loaded_depth:
            depth_stem = self.d.depth_files_raw[0].parent
            self.d.depth_files_raw = [depth_stem / f"{idx:>06d}.png" for idx in idx_list]
            pool.map(copy2, self.d.depth_files_raw, self.d.depth_files)

        self.d.rgb_array = pool.map(partial(load_rgb, bgr2rgb=True), self.d.rgb_files)
        console.log(f"[bold cyan]done")

    def _depth(self):
        console.rule(f"depth")
        filenames = get_ordered_files(self.c.p.t_depth, pattern=[".png"], ex_pattern=["v_"])

        if self.c.mono or "depth" not in self.c.steps_to_update and len(filenames) == self.d.target_frames:
            self.d.depth_array = self.pool.map(partial(load_depth, to_meter=True), filenames)
            console.log(f"[bold cyan]done (load)")
            return

        self._load_unimatch_wrapper(mode="stereo")
        self.d.depth_array = estimate_depth_stereo(  # In milli-meter now, will be converted to meter below
            self.uni_wrapper, self.d.rgb_files_l, self.d.rgb_files_r,
            camera_base_line=np.linalg.norm(self.d.cam.translation),
            focal_length_w=self.d.cam.get_intrinsics(flag_left=True, scale=1.0)[0][0],
            to_meter=False
        )

        console.log(f"[bold cyan]depth estimation done, saving to {self.c.p.t_depth}")

        raw_depth_filenames = [self.c.p.t_depth / f"{i:>06d}.png" for i in range(self.d.target_frames)]
        vis_depth_filenames = [self.c.p.t_depth / f"v_{i:>06d}.jpg" for i in range(self.d.target_frames)]
        self.d.depth_files = raw_depth_filenames

        pool = Pool(os.cpu_count())
        pool.map(write_depth, raw_depth_filenames, self.d.depth_array)
        for depth in self.d.depth_array:
            depth *= 0.001
        pool.map(partial(write_colorized_depth, min_meter=0., max_meter=3.), vis_depth_filenames, self.d.depth_array)
        console.log(f"[bold cyan]done")

    def _load_mediapipe_hand_detector(self):
        if hasattr(self, "_mp_hand_detector"):
            self._mp_hand_detector.reset()
            return

        from robot_vision.human_pose.mediapipe.mp_hand_kalman import HandDetector
        self._mp_hand_detector = HandDetector()

    def _load_mmpose_wholebody(self):
        if hasattr(self, "_rtmpose"):
            return

        from robot_vision.human_pose.mmpose.rtmpose.rtmpose_wrapper import RTMPoseWrapper
        self._rtmpose = RTMPoseWrapper()

    def _mmpose_wholebody(self):
        console.rule(f"mmpose whole-body pose estimation")
        hand_file_path = create_path(self.c.p.t_human / "rtmpose") / "humans.yaml"

        # TODO remove the deps to the following files in the future
        hand_uv_path = create_path(self.c.p.t_human / "rtmpose") / "hands.yaml"
        handedness_path = create_path(self.c.p.t_human / "rtmpose") / "handedness.yaml"
        hand_time_idx_path = create_path(self.c.p.t_human / "rtmpose") / "hands_time_idx.yaml"

        if "mmpose_wholebody" not in self.c.steps_to_update \
                and hand_file_path.is_file():
            self.d.human.rtm_humans = load_dataclass(HumanDataList, hand_file_path)

            # TODO remove the deps to the following files in the future, and move data structure to human instances
            self.d.human.mp_hand_landmarks = DictNumpyArray.from_yaml(hand_uv_path)  # n_hands Dict[hand, (T_i, 21, 2)]
            self.d.human.hand_time_masks = DictNumpyArray.from_yaml(hand_time_idx_path)  # n_hands Dict[hand, (T_i, )]
            self.d.human.n_hands = len(self.d.human.mp_hand_landmarks.data.keys())
            self.d.human.handedness = load_dict_from_yaml(handedness_path)

            console.log(f"[bold cyan]done (load)")
            return

        console.log(f"running rtmpose (MMPose) whole-body pose estimation ...")
        self._load_mmpose_wholebody()

        from vil.cfg.preprocessing import RTMPoseHumanData
        d_human: List[RTMPoseHumanData] = self.d.human.rtm_humans.data
        landmarks, time_idx, images = {}, {}, []

        # for i in track(range(len(self.d.rgb_array)), description="[blue]RTMPose Detection", console=console):
        for t in range(len(self.d.rgb_array)):
            humans, image = self._rtmpose.detect_on_images(self.d.rgb_array[t][..., ::-1], True)
            # positions: List[str, Any]
            n_new_humans = len(humans) - len(d_human)
            if n_new_humans > 0:
                d_human.extend([RTMPoseHumanData() for _ in range(n_new_humans)])

            for human_idx, human in enumerate(humans):
                human = self._rtmpose.check_hand_ids(human)
                h = d_human[human_idx]
                h.keypoints.append(np.array(human["keypoints"]))
                h.keypoint_scores.append(np.array(human['keypoint_scores']))
                h.bboxs.append(np.array(human["bbox"]))
                if human["left_hand"] is not None:
                    h.hand_landmark_l.append(human["left_hand"])
                    h.hand_time_idx_l.append(t)
                if human["right_hand"] is not None:
                    h.hand_landmark_r.append(human["right_hand"])
                    h.hand_time_idx_r.append(t)

            images.append(image)

        dump_data_to_yaml(HumanDataList, self.d.human.rtm_humans, hand_file_path)

        hand_idx = 0
        for human in d_human:
            if len(human.hand_time_idx_l) > 0:
                hand_name = f"hand_{hand_idx:>02d}"
                self.d.human.handedness.append("left")
                self.d.human.mp_hand_landmarks.data[hand_name] = np.array(human.hand_landmark_l)
                self.d.human.hand_time_masks.data[hand_name] = np.array(human.hand_time_idx_l)
                hand_idx += 1
            if len(human.hand_time_idx_r) > 0:
                hand_name = f"hand_{hand_idx:>02d}"
                self.d.human.handedness.append("right")
                self.d.human.mp_hand_landmarks.data[hand_name] = np.array(human.hand_landmark_r)
                self.d.human.hand_time_masks.data[hand_name] = np.array(human.hand_time_idx_r)
                hand_idx += 1
        self.d.human.n_hands = hand_idx
        DictNumpyArray.to_yaml(self.d.human.mp_hand_landmarks, hand_uv_path)  # n_hands Dict[hand, (T_i, 21, 2)]
        DictNumpyArray.to_yaml(self.d.human.hand_time_masks, hand_time_idx_path)  # n_hands Dict[hand, (T_i, )]
        save_to_yaml(self.d.human.handedness, handedness_path)

        if self.c.viz:
            path = create_path(self.c.p.v_human / "rtmpose")
            img_filenames = [path / f"{i:>06d}.jpg" for i in range(len(images))]
            self.pool.map(partial(write_rgb, bgr2rgb=False), img_filenames, images)

        console.log(f"[bold cyan]done")

    def _mediapipe_hand(self):
        console.rule(f"mediapipe hand detection")
        hand_uv_path = create_path(self.c.p.t_human / "mediapipe") / "hands.yaml"
        hand_num_path = create_path(self.c.p.t_human / "mediapipe") / "hands_num_array.yaml"
        hand_time_idx_path = create_path(self.c.p.t_human / "mediapipe") / "hands_time_idx.yaml"

        if "mediapipe_hand" not in self.c.steps_to_update \
                and hand_uv_path.is_file() \
                and hand_time_idx_path.is_file()\
                and hand_num_path.is_file():
            self.d.human.mp_hand_landmarks = DictNumpyArray.from_yaml(hand_uv_path)  # n_hands Dict[hand, (T_i, 21, 2)]
            self.d.human.hand_time_masks = DictNumpyArray.from_yaml(hand_time_idx_path)  # n_hands Dict[hand, (T_i, )]
            self.d.human.hand_num_list = load_dict_from_yaml(hand_num_path)
            self.d.human.n_hands = len(self.d.human.mp_hand_landmarks.data.keys())
            console.log(f"[bold cyan]done (load)")
            return

        console.log(f"running mediapipe hand detection ...")
        self._load_mediapipe_hand_detector()

        landmarks, time_idx, images = {}, {}, []
        for t, image in enumerate(self.d.rgb_array):
            positions, image = self._mp_hand_detector.run(image, True)
            for hand_name, hand_position in positions.items():
                if landmarks.get(hand_name, None) is None:
                    landmarks[hand_name] = [hand_position]
                    time_idx[hand_name] = [t]
                else:
                    landmarks[hand_name].append(hand_position)
                    time_idx[hand_name].append(t)
            self.d.human.hand_num_list.append(len(positions.keys()))
            images.append(image)

        # TODO: if some hands may disappear from the demo for a while (on purpose), we need a better strategy to
        #  assign hand ids. Maybe the hand detector should handle ID assignment inside it.
        self.d.human.n_hands = np.max(self.d.human.hand_num_list)
        img_size_wh = self.d.rgb_array[0].shape[:2][::-1]
        for i in range(self.d.human.n_hands):
            name = f"hand_{i:>02d}"
            self.d.human.hand_time_masks.data[name] = np.array(time_idx[name], dtype=int)
            self.d.human.mp_hand_landmarks.data[name] = np.array(landmarks[name]) * img_size_wh

        save_to_yaml(self.d.human.hand_num_list, hand_num_path)
        DictNumpyArray.to_yaml(self.d.human.mp_hand_landmarks, hand_uv_path)  # n_hands Dict[hand, (T_i, 21, 2)]
        DictNumpyArray.to_yaml(self.d.human.hand_time_masks, hand_time_idx_path)  # n_hands Dict[hand, (T_i, )]

        if self.c.viz:
            path_flow = create_path(self.c.p.v_human / "mp_hands")
            img_filenames = [path_flow / f"{i:>06d}.jpg" for i in range(len(images))]
            self.pool.map(partial(write_rgb, bgr2rgb=True), img_filenames, images)

        console.log(f"[bold cyan]done")

    def _crop_hand(self):
        console.rule(f"crop hand images")
        data_path = create_path(self.c.p.t_human / "mediapipe") / "crop_bbox.yaml"

        if "crop_hand" not in self.c.steps_to_update and data_path.is_file():
            self.d.human.mp_hand_crop_bbox = DictNumpyArray.from_yaml(data_path)  # n_hands Dict[hand, (T_i, 4)]
            console.log(f"[bold cyan]done (load)")
            return

        console.log(f"running cropping hand ...")
        img_size_wh = self.d.rgb_array[0].shape[:2][::-1]
        for hand_idx in range(self.d.human.n_hands):
            name = f"hand_{hand_idx:>02d}"
            # hand_mean_uv = (self.d.human.mp_hand_landmarks.data[name].mean(axis=-2) * img_size_wh).astype(int)  # (T, 2)
            hand_mean_uv = (self.d.human.mp_hand_landmarks.data[name].mean(axis=-2)).astype(int)  # (T, 2)
            hand_bbox_top_left = np.clip(
                hand_mean_uv - 112, np.zeros(2, dtype=int),
                np.array([img_size_wh[0] - 224, img_size_wh[1] - 224], dtype=int)
            )
            hand_bbox_bottom_right = hand_bbox_top_left + 224
            self.d.human.mp_hand_crop_bbox.data[name] = np.concatenate(
                (hand_bbox_top_left, hand_bbox_bottom_right), axis=-1
            )
        DictNumpyArray.to_yaml(self.d.human.mp_hand_crop_bbox, data_path)

    def _graphormer_hand(self):
        console.rule(f"mesh graphormer hand detection")
        uv_path = create_path(self.c.p.t_human / "graphormer") / "uv.yaml"
        xyz_path = self.c.p.t_human / "graphormer" / "xyz.yaml"
        hand_viz_path = create_path(self.c.p.v_human / "graphormer")

        if "graphormer_hand" not in self.c.steps_to_update and uv_path.is_file() and xyz_path.is_file():
            self.d.human.graphormer_uv = DictNumpyArray.from_yaml(uv_path)
            self.d.human.graphormer_xyz = DictNumpyArray.from_yaml(xyz_path)
            console.log(f"[bold cyan]done (load)")
            return

        console.log(f"running mesh graphormer ...")
        self._load_mesh_graphormer()

        # create a fused mask
        fused_mask_list: List[np.ndarray] = [None] * self.d.target_frames
        all_obj_list = self.d.mask_dict.keys()
        obj_list = []
        for obj in all_obj_list:
            if not "person" in obj:
               obj_list.append(obj) 

        for t in range(self.d.target_frames):
            for obj in obj_list:
                if fused_mask_list[t] is None:
                    fused_mask_list[t] = self.d.mask_dict[obj][t]
                else:
                    fused_mask_list[t] = self.d.mask_dict[obj][t] | fused_mask_list[t]

        log.disable_logging()
        self.d.human.graphormer_uv.data, self.d.human.graphormer_xyz.data = \
            self.mg_model.detect_hand_on_image_batch(
                self.d.rgb_array,                           # (T_i, h, w, 3)
                self.d.depth_array,                         # (T_i, h, w)
                self.d.cam.get_intrinsics(flag_left=True),  # (3, 3)
                self.d.human.mp_hand_crop_bbox,             # (T, n_hands, 4)
                self.d.human.handedness,                    # (n_hands, )
                hand_viz_path,
                True,
                fused_mask_list,                            # (n_hands, )
                self.d.human.hand_time_masks,               # (n_hands, )
            )  # [hand_name, (T_i, 21, 2)], [hand_name, (T_i, 21, 3)]
        log.enable_logging()

        DictNumpyArray.to_yaml(self.d.human.graphormer_uv, uv_path)
        DictNumpyArray.to_yaml(self.d.human.graphormer_xyz, xyz_path)
        console.log(f"[bold cyan]done")

    def _mediapipe_handedness(self):
        console.rule(f"mediapipe_handedness")
        result_filename = self.c.p.t_human / "human_id.yaml"
        from vil.cfg.preprocessing import HumanProperty, HumanID

        if "mediapipe_handedness" not in self.c.steps_to_update and result_filename.is_file():
            data = load_dict_from_yaml(result_filename)
            self.d.human.human_id = load_dataclass(HumanID, data["mediapipe_humans"])
            self.d.human.handedness = data["mediapipe_handedness"]
            console.log(f"[bold cyan]done (load)")
            return

        from robot_vision.human_pose.mediapipe.holistic_handedness_detection import handedness_detection

        # Dict[str, Any], np.ndarray
        start_time_idx = np.argmax(self.d.human.hand_num_list).flatten()[0]
        humans, viz_img = handedness_detection(self.d.rgb_array, viz=True, start_idx=start_time_idx)

        if humans["time_idx"] >= 0:
            out_yaml_file = create_path(self.c.p.t_human / "mediapipe/holistic_handedness") / "wrist_uvs.yaml"
            img_file = create_path(self.c.p.v_human / "mediapipe/holistic_handedness") / f"{humans['time_idx']:>06d}.jpg"
            save_to_yaml(humans, out_yaml_file)
            write_rgb(img_file, viz_img, True)

        # TODO check n_hands == 2 * n_humans
        # TODO make sure the holistic and the hand detection has the same number of hands at this time point
        hand_com = []
        img_size_wh = self.d.rgb_array[0].shape[:2][::-1]
        for hand_idx in range(self.d.human.n_hands):
            name = f"hand_{hand_idx:>02d}"
            t = np.argwhere(self.d.human.hand_time_masks.data[name] == humans["time_idx"]).flatten()[0]
            hand_com.append(
                (self.d.human.mp_hand_landmarks.data[name][t].mean(axis=-2) * img_size_wh).astype(int)
            )
        hand_com = np.array(hand_com)  # (n_hand, 2)

        # TODO update the above function to output multiple humans
        n_hands = self.d.human.n_hands
        handedness_list = [None] * n_hands

        if humans["time_idx"] < 0:
            console.log("[red]Holistic human detection failed, try to use heuristic method")
            if n_hands > 2:
                raise ValueError("heuristic method can only deal with equal or less than 2 hands")
            h = HumanProperty()
            if n_hands == 1:
                h.right_hand_idx = 0
                handedness_list[h.right_hand_idx] = "right"
            elif n_hands == 2:
                if hand_com[0][0] < hand_com[1][0]:
                    h.right_hand_idx = 0
                    h.left_hand_idx = 1
                else:
                    h.left_hand_idx = 0
                    h.right_hand_idx = 1
                handedness_list[h.left_hand_idx] = "left"
                handedness_list[h.right_hand_idx] = "right"
            self.d.human.human_id.humans.append(h)
        else:
            included_idx = list(np.arange(n_hands))
            for idx, human in enumerate(humans["human_list"]):
                h = HumanProperty()

                dist_left = np.linalg.norm(hand_com[included_idx] - human["wrist_uv_left"], axis=-1)
                dist_right = np.linalg.norm(hand_com[included_idx] - human["wrist_uv_right"], axis=-1)
                h.left_hand_idx = included_idx[np.argmin(dist_left)]
                h.right_hand_idx = included_idx[np.argmin(dist_right)]

                handedness_list[h.left_hand_idx] = "left"
                handedness_list[h.right_hand_idx] = "right"
                included_idx.pop(included_idx.index(h.left_hand_idx))
                included_idx.pop(included_idx.index(h.right_hand_idx))

                self.d.human.human_id.humans.append(h)

        self.d.human.handedness = handedness_list
        save_to_yaml(dict(
            mediapipe_humans=dump_data_to_dict(HumanID, self.d.human.human_id),
            mediapipe_handedness=handedness_list
        ), result_filename)
        console.log(f"[bold cyan]done")

    def _mask(self):
        console.rule(f"mask")

        data_file = self.c.p.t_mask / "grounded_sam.pth"
        obj_list = list(self.d.scene_graph.obj_cfg.keys())

        def redo():
            if "mask" in self.c.steps_to_update or not data_file.is_file():
                return True

            for obj in obj_list:
                p = self.c.p.t_mask / f"{obj}"
                if not p.is_dir() or len(
                        get_ordered_files(p, pattern=[], ex_pattern=["v_"])) != self.d.target_frames:
                    return True

                mask_files = [p / f"{i:>06d}.png" for i in range(self.d.target_frames)]
                self.d.mask_dict[obj] = self.pool.map(partial(load_mask, as_binary=True), mask_files)

            self.d.grounded_sam = torch.load(data_file)
            console.log(f"[bold cyan]done (load)")
            return False

        if not redo():
            return

        self._load_mask_wrapper()
        self._load_sam_tracker()
        self.mask_tracker.reset()

        mask_overlays = []
        mask_list = []
        erode_radius = []
        for i in track(range(self.d.target_frames), description="[blue]Generating mask", console=console):
            # ========only use SAM

            # self.d.grounded_sam = self.mask_wrapper.get_masks(
            #     self.d.rgb_array[i], self.d.scene_graph.erode, self.d.scene_graph.obj_cfg, as_binary=True
            # )
            # torch.save(self.d.grounded_sam, data_file)
            # if i == 0:
            #     self.d.mask_dict = self.d.grounded_sam["masks_np"]
            # else:
            #     for o in obj_list:
            #         self.d.mask_dict[o].append(self.d.grounded_sam["masks_np"][o][0])

            # # ======segmentation using SAM on the first image frame
            if self.mask_tracker.step == 0:
                self.d.grounded_sam = self.mask_wrapper.get_masks(
                    self.d.rgb_array[i], self.d.scene_graph.erode, self.d.scene_graph.obj_cfg, as_binary=True
                )
                torch.save(self.d.grounded_sam, data_file)
                self.d.mask_dict = self.d.grounded_sam["masks_np"]
                for o in obj_list:
                    mask_list.append(self.d.mask_dict[o][0])
                    erode_radius.append(self.d.scene_graph.erode[o])

            # Later on, use track anything to track the masks
            images, masks = self.mask_tracker.track(self.d.rgb_array[i], mask_list, viz=True, erode_radius=erode_radius)
            mask_overlays.append(images)
            if i > 0:
                for idx, o in enumerate(obj_list):
                    self.d.mask_dict[o].append(masks[idx])

        for obj in obj_list:
            mask_path = create_path(self.c.p.t_mask / f"{obj}")
            mask_files = [mask_path / f"{i:>06d}.png" for i in range(self.d.target_frames)]
            self.pool.map(write_binary_mask, mask_files, self.d.mask_dict[obj])

        mask_path = create_path(self.c.p.t_mask / f"all")
        mask_files = [mask_path / f"{i:>06d}.jpg" for i in range(self.d.target_frames)]
        self.pool.map(partial(write_rgb, bgr2rgb=True), mask_files, mask_overlays)
        console.log(f"[bold cyan]done")

    def _create_dcn_canonical_space(self):
        console.rule(f"canonical")

        obj_list = self.kvil_data.can.dcn_obj_cfg.keys()
        skip_uv_flag = {o: False for o in obj_list}
        for obj in obj_list:
            skip_uv_flag[obj] = validate_file(self.c.p.can / "dcn" / obj / "uv.yaml")[1]

        # the following for loop checks for each object, if the folder canonical/dcn/obj_name exist,
        # if not, either
        # 1) create one by automatically creating one and select a time step in one of the demos as reference, or
        # 2) ask the user to manually select one on the fly
        console.log(f'[yellow]check existence of canonical space')
        for obj in obj_list:
            if not validate_path(self.c.p.can / "dcn" / obj)[1]:
                console.log(f"[bold cyan]creating canonical space for {obj}")
                if self.c.auto_can_space:
                    demo_idx = 0
                    time_idx = 0
                    console.log(f"[bold yellow]automatically select the canonical space "
                                f"from {demo_idx}-th demo and {time_idx}-th time step")
                else:
                    text = "please go to filesystem and look for which demo and timestep to select, now select demo"
                    trials = [p.stem for p in get_ordered_subdirs(self.c.p.rec)]
                    selected_trial = ask_list(text, trials)
                    demo_idx = trials.index(selected_trial)
                    time_idx = int(ask_text("now type time index"))

                    console.log(f"[bold yellow]User selected the canonical space "
                                f"from {demo_idx}-th demo and {time_idx}-th time step")

                if demo_idx > len(self.kvil_data.demos):
                    raise RuntimeError(f"the {demo_idx}-th demo has not been processed yet")

                d = self.kvil_data.demos[self.c.p.trial_idx]
                if time_idx > len(d.rgb_array):
                    raise RuntimeError(f"the time index {time_idx} exceed the number time steps in the demo")

                self.kvil_data.can.rgb[obj] = d.rgb_array[time_idx]
                self.kvil_data.can.depth[obj] = d.depth_array[time_idx]
                self.kvil_data.can.mask[obj] = d.mask_dict[obj][time_idx]
                self.kvil_data.can.intrinsics[obj] = d.cam.get_intrinsics(flag_left=True)
                write_rgb(create_path(self.c.p.can / f"dcn/{obj}") / "rgb.jpg", d.rgb_array[time_idx], bgr2rgb=True)
                write_depth(self.c.p.can / f"dcn/{obj}/depth.png", d.depth_array[time_idx] * 1000)
                write_binary_mask(self.c.p.can / f"dcn/{obj}/mask.jpg", d.mask_dict[obj][time_idx])
                # dump_data_to_yaml(NumpyArray, d.cam.get_intrinsics(flag_left=True), self.c.p.can / f"dcn/{obj}/intrinsics.yaml")
                save_to_yaml(d.cam.get_intrinsics(flag_left=True), self.c.p.can / f"dcn/{obj}/intrinsics.yaml")
            else:
                console.log(f"[bold cyan]loading rgb, depth, and mask from {self.c.p.can / f'dcn/{obj}'} folder")
                self.kvil_data.can.rgb[obj] = load_rgb(self.c.p.can / f"dcn/{obj}/rgb.jpg", bgr2rgb=True)
                self.kvil_data.can.depth[obj] = load_depth(self.c.p.can / f"dcn/{obj}/depth.png")
                self.kvil_data.can.mask[obj] = load_mask(self.c.p.can / f"dcn/{obj}/mask.jpg")
                self.kvil_data.can.intrinsics[obj] = np.array(
                    load_dict_from_yaml(self.c.p.can / f"dcn/{obj}/intrinsics.yaml"))

        # The following for loop checks
        # 1) if there's already uv.yaml computed in previous run,
        #    if so, load it from file, otherwise resample from the mask image
        # 2) then project the uv coordinates to 3D using intrinsics
        console.log(f"[yellow]check uv existence")
        for obj in obj_list:
            path_uv = create_path(self.c.p.can / f"dcn/{obj}") / "uv.yaml"
            path_coord_3d = self.c.p.can / f"dcn/{obj}/coordinates_3d.yaml"
            path_color = self.c.p.can / "dcn" / obj / "uv_colors.yaml"
            if skip_uv_flag[obj]:
                uv = np.array(load_dict_from_yaml(path_uv), dtype=int)
                coordinates_3d = np.array(load_dict_from_yaml(path_coord_3d), dtype=float)
                colors = np.array(load_dict_from_yaml(path_color), dtype=int)
            else:
                uv = random_uv_on_mask(
                    torch.from_numpy(self.kvil_data.can.mask[obj]),
                    self.kvil_data.can.num_candidates[obj]
                ).numpy().astype(int)
                save_to_yaml(uv.tolist(), self.c.p.can / f"dcn/{obj}/uv.yaml")

                coordinates_3d = pinhole_projection_image_to_camera(
                    uv, self.kvil_data.can.depth[obj],
                    intrinsic=self.kvil_data.can.intrinsics[obj]
                )
                save_to_yaml(coordinates_3d.tolist(), path_coord_3d)

                img = copy.deepcopy(self.kvil_data.can.rgb[obj])
                distances = np.linalg.norm(uv, axis=-1)  # Calculate distance from the origin
                distances = (distances - distances.min()) / (distances.max() - distances.min())
                colors = plt.colormaps.get_cmap("plasma")(distances)[:, :3] * 255
                save_to_yaml(colors.tolist(), path_color)
                if self.c.viz:
                    img = viz_points_with_color_map(img, uv, colors, copy_image=False)
                    write_rgb(self.c.p.can / "dcn" / obj / "overlay.jpg", img, bgr2rgb=True)

            self.kvil_data.can.uv[obj] = uv
            self.kvil_data.can.cand_pt[obj] = coordinates_3d
            self.kvil_data.can.uv_colors[obj] = colors

        for obj in obj_list:
            outlier_idx_file = self.c.p.can / "dcn" / obj / "can_outlier.yaml"
            inlier_idx_file = self.c.p.can / "dcn" / obj / "can_inlier.yaml"
            fixed_xyz_file = self.c.p.can / "dcn" / obj / "coordinates_3d_fixed.yaml"
            if "canonical" not in self.c.steps_to_update and \
                    outlier_idx_file.is_file() and inlier_idx_file.is_file() and fixed_xyz_file.is_file():
                continue

            raw_xyz = self.kvil_data.can.cand_pt[obj]
            inlier_idx, outlier_idx = outlier_det.outlier_detection(raw_xyz, exclude_idx=None)
            neighbors = find_neighbors(raw_xyz, num_neighbor_pts=30)  # P, n_neighbor

            if outlier_idx.shape[0] > 0:
                masked_state = outlier_det.get_neighbor_and_inlier_state_matrix(raw_xyz.shape[0], inlier_idx, neighbors)

                fixed_depth = outlier_det.fix_depth_for_outlier_in_one_frame(
                    masked_state, self.kvil_data.can.depth[obj], self.kvil_data.can.uv[obj]
                )

                fixed_xyz = pinhole_projection_image_to_camera(
                    self.kvil_data.can.uv[obj], np.array(fixed_depth),  # (T, 300, 2), [T x (h, w)]
                    intrinsic=self.kvil_data.can.intrinsics[obj]
                )
                # get the new outliers and filter them with a low order filter
                inlier_idx, outlier_idx = outlier_det.outlier_detection(fixed_xyz, exclude_idx=None)
            else:
                fixed_xyz = raw_xyz

            save_to_yaml(fixed_xyz.tolist(), fixed_xyz_file)
            save_to_yaml(inlier_idx.tolist(), inlier_idx_file)
            save_to_yaml(outlier_idx.tolist(), outlier_idx_file)

        # compute the dense correspondence descriptors for each object with each corresponding rgb image
        # and then read the descriptors of each candidate pixels
        path_descriptor = self.c.p.can / "dcn/descriptors.pth"
        if "canonical" in self.c.steps_to_update or not path_descriptor.is_file():
            console.log(f"[yellow]compute descriptors")
            self._load_dcn_model()
            descriptor_images: Dict[str, torch.Tensor] = self.dcn.compute_descriptor_images(
                [self.kvil_data.can.rgb[obj] for obj in self.dcn.obj_list]
            )
            for obj in self.kvil_data.can.dcn_obj_cfg.keys():
                self.kvil_data.can.descriptor[obj] = descriptor_images[obj][
                    self.kvil_data.can.uv[obj][:, 1],
                    self.kvil_data.can.uv[obj][:, 0]
                ]
                descriptor_file = self.c.p.can / "dcn" / obj / "descriptor.yaml"
                save_to_yaml(self.kvil_data.can.descriptor[obj].cpu().numpy().tolist(), descriptor_file)
            torch.save(self.kvil_data.can.descriptor, str(path_descriptor))
        else:
            self.kvil_data.can.descriptor = torch.load(path_descriptor)
        console.log(f"[bold cyan]done")

    def _random_dcn(self):
        console.rule(f"dcn")
        uv_file_path = self.c.p.t_dcn / "uv.yaml"

        if "random_dcn" not in self.c.steps_to_update and uv_file_path.is_file():
            self.d.dcn_uv = DictNumpyArray.from_yaml(uv_file_path)
            console.log(f"[bold cyan]done (load)")
            return

        console.log(f"running random DCN ...")
        self._load_dcn_model()

        obj_list = self.kvil_data.can.dcn_obj_cfg.keys()
        descriptor_images: Dict[str, torch.Tensor] = self.dcn.compute_descriptor_images(
            [self.d.rgb_array[0] for _ in obj_list]
        )
        device = self.dcn.device
        for obj in obj_list:
            uv = random_uv_on_mask(
                    torch.from_numpy(self.d.mask_dict[obj][0]),
                    self.kvil_data.can.num_candidates[obj]
                ).numpy().astype(int)
            descriptor = descriptor_images[obj][
                    uv[:, 1],
                    uv[:, 0]
                ]
            can_descriptor = self.kvil_data.can.descriptor[obj]
            similarity_matrix = cosine_similarity(can_descriptor, descriptor)
            row_indices, col_indices = linear_sum_assignment(
                similarity_matrix,
                maximize=dict(
                    cosine=True,
                    euc=False
                )["cosine"]
            )
            # row_indices, col_indices = compute_assignment(can_descriptor, descriptor, "cosine")
            # ic(row_indices, col_indices)
            new_uv = np.asarray(uv)
            new_uv[row_indices] = uv[col_indices]
            self.d.dcn_uv.data[obj] = new_uv

            if self.c.viz:
                img = overlay_masks_on_image(
                    copy.deepcopy(self.d.rgb_array[0]), [self.d.mask_dict[obj][0]],
                    colors=[(np.array(lc.cyan_process[:3]) * 255).astype(int)], rgb_weights=0.5
                )
                img = viz_points_with_color_map(
                    img, self.d.dcn_uv.data[obj], self.kvil_data.can.uv_colors[obj])

                write_rgb(create_path(self.c.p.v_dcn / obj) / f"{0:>06d}.jpg", img, bgr2rgb=True)
                write_rgb(create_path(self.c.p.v_flow / obj / "uv") / f"{0:>06d}.jpg", img, bgr2rgb=True)
        DictNumpyArray.to_yaml(self.d.dcn_uv, uv_file_path)
        console.log(f"[bold cyan]done")


    def _dcn(self):
        console.rule(f"dcn")
        uv_file_path = self.c.p.t_dcn / "uv.yaml"

        if "dcn" not in self.c.steps_to_update and uv_file_path.is_file():
            self.d.dcn_uv = DictNumpyArray.from_yaml(uv_file_path)
            console.log(f"[bold cyan]done (load)")
            return

        console.log(f"running DCN ...")
        self._load_dcn_model()

        obj_list = self.kvil_data.can.dcn_obj_cfg.keys()
        descriptor_images: Dict[str, torch.Tensor] = self.dcn.compute_descriptor_images(
            [self.d.rgb_array[0] for _ in obj_list]
        )
        device = self.dcn.device
        for obj in obj_list:
            best_match_uv = find_best_match_for_descriptor_cosine(
                self.kvil_data.can.descriptor[obj].to(device), descriptor_images[obj].to(device),
                torch.from_numpy(self.d.mask_dict[obj][0].astype(bool)).to(device)
            )
            self.d.dcn_uv.data[obj] = best_match_uv.detach().cpu().numpy()

            if self.c.viz:
                img = overlay_masks_on_image(
                    copy.deepcopy(self.d.rgb_array[0]), [self.d.mask_dict[obj][0]],
                    colors=[(np.array(lc.cyan_process[:3]) * 255).astype(int)], rgb_weights=0.5
                )
                img = viz_points_with_color_map(
                    img, self.d.dcn_uv.data[obj], self.kvil_data.can.uv_colors[obj])

                write_rgb(create_path(self.c.p.v_dcn / obj) / f"{0:>06d}.jpg", img, bgr2rgb=True)
                write_rgb(create_path(self.c.p.v_flow / obj / "uv") / f"{0:>06d}.jpg", img, bgr2rgb=True)

        DictNumpyArray.to_yaml(self.d.dcn_uv, uv_file_path)
        console.log(f"[bold cyan]done")

    def _point_track(self):
        console.rule(f"point tracking")
        uv_file_path = self.c.p.t_flow / "uv.yaml"

        if "point_track" not in self.c.steps_to_update and uv_file_path.is_file():
            self.d.uv = DictNumpyArray.from_yaml(uv_file_path)
            console.log(f"[bold cyan]done (load)")
            return

        console.log(f"running point tracking ...")

        self._load_point_tracker()

        obj_list = self.kvil_data.can.dcn_obj_cfg.keys()

        device = self.point_tracker.device
        points = torch.tensor(
            np.concatenate([self.d.dcn_uv.data[obj] for obj in obj_list], axis=0), dtype=torch.float32, device=device
        )  # (n_obj * 300, 2)
        images = torch.tensor(
            np.array(self.d.rgb_array), dtype=torch.float32, device=device
        ).permute(0, 3, 1, 2).unsqueeze(0)  # (1, T, C, H, W)

        fused_mask = None
        for obj in obj_list:
            if fused_mask is None:
                fused_mask = np.array(self.d.mask_dict[obj])
            else:
                fused_mask = np.array(self.d.mask_dict[obj]) | fused_mask
        masks = torch.tensor(fused_mask, device=device)
        ic(images.shape, points.shape, masks.shape)

        with torch.no_grad():
            traj = self.point_tracker.run(images, points, masks)
            traj = traj.squeeze().detach().cpu().numpy().astype(np.int32)
            idx_start = 0
            for i, obj in enumerate(obj_list):
                n_points = self.kvil_data.can.num_candidates[obj]
                self.d.uv.data[obj] = traj[:, idx_start:idx_start + n_points]
                idx_start += n_points

        if self.c.viz:
            def viz(t: int, uvs: np.ndarray, img: np.ndarray, mask: np.ndarray, colors: np.ndarray, path_uv: Path):
                img = overlay_masks_on_image(
                    img, [mask], colors=[(np.array(lc.orange_peel[:3]) * 255).astype(int)], rgb_weights=0.5
                )
                img = viz_points_with_color_map(img, uvs, colors)
                write_rgb(path_uv / f"{t:>06d}.jpg", img, bgr2rgb=True)

            for obj in obj_list:
                path_uv = create_path(self.c.p.v_flow / obj / "uv")
                self.pool.map(
                    partial(viz, colors=self.kvil_data.can.uv_colors[obj], path_uv=path_uv),
                    np.arange(1, len(self.d.rgb_array)), self.d.uv.data[obj][1:],
                    self.d.rgb_array[1:], self.d.mask_dict[obj][1:]
                )

        DictNumpyArray.to_yaml(self.d.uv, uv_file_path)
        console.log(f"[bold cyan]done")

    def _opt_flow_raft(self):
        console.rule(f"optical flow")
        uv_file_path = self.c.p.t_flow / "uv.yaml"

        if "opt_flow_raft" not in self.c.steps_to_update and uv_file_path.is_file():
            self.d.uv = DictNumpyArray.from_yaml(uv_file_path)
            console.log(f"[bold cyan]done (load)")
            return

        console.log(f"running optical flow ...")
        self._load_opt_flow()

        obj_list = self.kvil_data.can.dcn_obj_cfg.keys()
        time_steps = self.d.target_frames

        for obj in obj_list:
            uv = self.d.dcn_uv.data.get(obj)
            self.d.uv.data[obj] = np.zeros((self.d.target_frames, ) + uv.shape, dtype=np.int32)
            self.d.uv.data[obj][0] = uv

        flow_imgs = []
        device = self.opt_flow.device

        def to_tensor(image: np.ndarray):
            return torch.tensor(image, dtype=torch.float32, device=device).permute(2, 0, 1).unsqueeze(0)

        from scipy.spatial.distance import cdist
        occ_index = {}
        down_sample_idx = np.array(self.d.rgb_down_sample_idx, dtype=int)
        for obj in obj_list:
            occ_cfg = self.d.vid_clip.occlude.get(obj, None)
            if occ_cfg is None:
                occ_index[obj] = None
                continue

            idx = np.where(occ_cfg.s <= down_sample_idx)[0][0]
            occ_index[obj] = idx
            console.log(f"[bold green]obj {obj} is occluded from {idx}-th frame in the demo")

        prev_uv_dict = {k: None for k in obj_list}
        uv_max = np.array(self.d.rgb_array[0].shape[:2][::-1], dtype=int) - 1
        for i in track(range(1, time_steps), description="[blue]Estimating flow: ", console=console):
            with torch.no_grad():
                flow, flow_img = self.opt_flow.run(to_tensor(self.d.rgb_array[i-1]), to_tensor(self.d.rgb_array[i]))

            for obj in obj_list:
                # check if the obj is occluded at this time, if so, use the previous detection
                if occ_index[obj] is not None and i >= occ_index[obj]:
                    new_uv = prev_uv_dict.get(obj, None)
                    # if no previous detection, do it anyway
                    if new_uv is not None:
                        self.d.uv.data[obj][i] = new_uv
                        continue
                uvs = self.d.uv.data[obj][i-1]
                best_match_uv_delta = flow[uvs[:, 1], uvs[:, 0]]
                new_uvs = uvs + best_match_uv_delta
                new_uvs = np.clip(new_uvs, np.zeros(2, dtype=int), uv_max)
                prev_uv_dict[obj] = new_uvs

                mask = self.d.mask_dict[obj][i]
                pixel_out_of_mask_idx = np.where(mask[new_uvs[:, 1], new_uvs[:, 0]] == 0)[0]
                if len(pixel_out_of_mask_idx) > 0:
                    console.log(f"{obj} at {i} step: {len(pixel_out_of_mask_idx)} out pixels")
                    mask_uv = np.argwhere(mask)[:, ::-1]
                    if len(mask_uv) == 0:
                        raise ValueError(f"Please check your mask generation of obj {obj},"
                                         f"seems to be empty at frame {i}")
                    dist = cdist(new_uvs[pixel_out_of_mask_idx], mask_uv)
                    idx = dist.argmin(axis=-1)
                    new_uvs[pixel_out_of_mask_idx] = mask_uv[idx]

                self.d.uv.data[obj][i] = new_uvs

            torch.cuda.empty_cache()
            flow_imgs.append(flow_img)

        if self.c.viz:
            def viz(uvs: np.ndarray, img: np.ndarray, mask: np.ndarray, colors: np.ndarray):
                img = overlay_masks_on_image(
                    img, [mask], colors=[(np.array(lc.cyan_process[:3]) * 255).astype(int)], rgb_weights=0.8
                )
                return viz_points_with_color_map(img, uvs, colors)

            uv_images = [copy.deepcopy(img) for img in self.d.rgb_array]
            for obj in obj_list:
                uv_images = self.pool.map(
                    partial(viz, colors=self.kvil_data.can.uv_colors[obj]),
                    self.d.uv.data[obj], uv_images, self.d.mask_dict[obj]
                )

            path_uv = create_path(self.c.p.v_flow / "uv")
            uv_img_filenames = [path_uv / f"{i:>06d}.jpg" for i in range(time_steps)]
            self.pool.map(partial(write_rgb, bgr2rgb=True), uv_img_filenames, uv_images)

            # save the flow images
            path_flow = create_path(self.c.p.v_flow / "flow")
            flow_img_filenames = [path_flow / f"{i:>06d}.jpg" for i in range(time_steps)]
            self.pool.map(partial(write_rgb, bgr2rgb=True), flow_img_filenames, flow_imgs)

        DictNumpyArray.to_yaml(self.d.uv, uv_file_path)
        console.log(f"[bold cyan]done")

    def _opt_flow(self):
        console.rule(f"optical flow")
        uv_file_path = self.c.p.t_flow / "uv.yaml"

        if "opt_flow" not in self.c.steps_to_update and uv_file_path.is_file():
            self.d.uv = DictNumpyArray.from_yaml(uv_file_path)
            console.log(f"[bold cyan]done (load)")
            return

        console.log(f"running optical flow ...")

        self._load_unimatch_wrapper(mode="flow")
        self.uni_wrapper.initialize(self.d.rgb_array[0])

        obj_list = self.kvil_data.can.dcn_obj_cfg.keys()
        time_steps = self.d.target_frames

        # # Note Case 1: ===========================================================
        # flow_imgs, flow_list = [], []
        # fused_mask = None
        # for obj in obj_list:
        #     if fused_mask is None:
        #         fused_mask = self.d.mask_dict[obj]
        #     else:
        #         fused_mask = self.d.mask_dict[obj] | fused_mask
        #
        # rgb = self.d.rgb_array * fused_mask[..., np.newaxis]
        #
        # if self.c.viz:
        #     path_fused_mask = create_path(self.c.p.v_flow / "fused_mask")
        #     path_fused_masked_rgb = create_path(self.c.p.v_flow / "fused_rgb")
        #
        #     fused_mask_filenames = [path_fused_mask / f"{i:>06d}.jpg" for i in range(self.d.target_frames)]
        #     fused_masked_rgb_filenames = [path_fused_masked_rgb / f"{i:>06d}.jpg" for i in range(self.d.target_frames)]
        #
        #     self.pool.map(write_binary_mask, fused_mask_filenames, fused_mask)
        #     self.pool.map(partial(write_rgb, bgr2rgb=True), fused_masked_rgb_filenames, rgb)
        #
        # for i in track(range(time_steps - 1), description="[blue]Estimating flow: ", console=console):
        #     flow, flow_img = self.uni_wrapper.estimate_flow(rgb[i], rgb[i + 1])
        #     flow_list.append(flow)
        #     flow_imgs.append(flow_img)
        # # Note: end case 1 ===========================================================

        import robot_utils.viz.latex_colors_rgba as lc
        color = [0.26, 0.26, 0]

        from scipy.spatial.distance import cdist
        from sklearn.cluster import KMeans
        for obj in obj_list:
            # # case 1
            # uv_list = [self.d.dcn_uv.data.get(obj)]

            # case 2:
            rgb = [
                overlay_mask_on_image(self.d.rgb_array[i], self.d.mask_dict[obj][i], color=np.array(color) * 255, rgb_weights=1.0)
                for i in range(self.d.target_frames)
            ]

            # def complementary(rgb, mask):
            #     grayscale_image = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) * 0.5
            #     blurred_image = cv2.GaussianBlur(grayscale_image, (15, 15), 0)
            #     rgb[mask == 0] = np.ones((1, 3)) * blurred_image[mask == 0][:, None]
            #     return rgb

            def find_complementary_color(rgb, mask):
                idx = np.where(mask)
                pixels = rgb[idx[0], idx[1]]

                # Perform k-means clustering on the unique colors
                num_clusters = 10
                kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init="auto").fit(pixels)
                centroids = np.array([c for c in kmeans.cluster_centers_])
                complementary_colors = 255 - centroids

                # # find the complementary color that is most far away from all other colors
                # dist = np.linalg.norm(complementary_colors - centroids[None, ...], axis=-1)
                # complementary_color_idx = np.argmax(dist.mean(axis=-1))
                # # most_different_color = complementary_colors[complementary_color_idx]
                # most_different_color = np.clip(complementary_colors[complementary_color_idx] * 2, 0, 255)

                # find the complementary color of the most dominant color
                cluster_assignments = kmeans.predict(pixels)
                cluster_counts = np.bincount(cluster_assignments)
                cluster_with_most_elements = np.argmax(cluster_counts)
                # most_different_color = complementary_colors[cluster_with_most_elements]
                most_different_color = np.clip(complementary_colors[cluster_with_most_elements] * 2, 0, 255)

                return most_different_color

            def replace_background_color(rgb, mask, color):
                rgb[mask == 0] = color
                return rgb

            color = find_complementary_color(self.d.rgb_array[0], self.d.mask_dict[obj][0])

            rgb = [
                replace_background_color(copy.deepcopy(self.d.rgb_array[i]), self.d.mask_dict[obj][i], color)
                for i in range(self.d.target_frames)
            ]
            
            flow_imgs, uv_list, flow_list = [], [self.d.dcn_uv.data.get(obj)], []

            for i in track(range(time_steps - 1), description="[blue]Estimating flow: ", console=console):
                # # case 1:
                # flow = flow_list[i]
                # case 2:
                flow, flow_img = self.uni_wrapper.estimate_flow(rgb[i], rgb[i + 1])

                uvs = uv_list[i]
                best_match_uv_delta = flow[uvs[:, 1], uvs[:, 0]]
                new_uvs = uvs + best_match_uv_delta

                mask = self.d.mask_dict[obj][i+1]
                pixel_out_of_mask_idx = np.where(mask[new_uvs[:, 1], new_uvs[:, 0]] == 0)[0]
                mask_uv = np.argwhere(mask)[:, ::-1]
                dist = cdist(new_uvs[pixel_out_of_mask_idx], mask_uv)
                idx = dist.argmin(axis=-1)
                new_uvs[pixel_out_of_mask_idx] = mask_uv[idx]

                uv_list.append(new_uvs)
                # case 2:
                flow_list.append(flow)
                flow_imgs.append(flow_img)

            self.d.uv.data[obj] = np.array(uv_list)
            self.d.flow.data[obj] = torch.from_numpy(np.array(flow_list))

            if self.c.viz:
                def viz(t: int, uvs: np.ndarray, img: np.ndarray, mask: np.ndarray, colors: np.ndarray, path_uv: Path):
                    img = overlay_masks_on_image(
                        img, [mask], colors=[(np.array(lc.orange_peel[:3]) * 255).astype(int)], rgb_weights=0.5
                    )
                    img = viz_points_with_color_map(img, uvs, colors)
                    write_rgb(path_uv / f"{t:>06d}.jpg", img, bgr2rgb=True)

                path_uv = create_path(self.c.p.v_flow / obj / "uv")
                self.pool.map(
                    partial(viz, colors=self.kvil_data.can.uv_colors[obj], path_uv=path_uv),
                    np.arange(1, len(self.d.rgb_array)), self.d.uv.data[obj][1:],
                    self.d.rgb_array[1:], self.d.mask_dict[obj][1:]
                )

                # save the flow images
                path_flow = create_path(self.c.p.v_flow / obj / "flow")
                flow_img_filenames = [path_flow / f"{i:>06d}.jpg" for i in range(time_steps)]
                self.pool.map(partial(write_rgb, bgr2rgb=True), flow_img_filenames, flow_imgs)

                # save the masked rgb images
                path_mask_rgb = create_path(self.c.p.v_flow / obj / "masked_rgb")
                img_filenames = [path_mask_rgb / f"{i:>06d}.jpg" for i in range(time_steps)]
                self.pool.map(partial(write_rgb, bgr2rgb=True), img_filenames, rgb)

        DictNumpyArray.to_yaml(self.d.uv, uv_file_path)
        console.log(f"[bold cyan]done")

    def _obj_kpt_traj(self):
        console.rule(f"obj_kpt_traj")
        obj_xyz_file = self.c.p.t_obj / "xyz.yaml"

        if "obj_kpt_traj" not in self.c.steps_to_update and obj_xyz_file.is_file():
            self.d.xyz = DictNumpyArray.from_yaml(obj_xyz_file)
            console.log(f"[bold cyan]done (load)")
            return

        console.log(f"projecting from image to camera frame (3D) ...")

        for obj in self.kvil_data.can.dcn_obj_cfg.keys():
            self.d.xyz.data[obj] = np.array(self.pool.map(
                partial(pinhole_projection_image_to_camera, intrinsic=self.d.cam.get_intrinsics(flag_left=True)),
                self.d.uv.data[obj], self.d.depth_array  # (T, 300, 2), [T x (h, w)]
            ))
        DictNumpyArray.to_yaml(self.d.xyz, obj_xyz_file)

        console.log(f"[bold cyan]done")

    def _convert_to_kvil(self):
        console.rule(f"convert to K-VIL legacy data structure")
        # human_id = self.d.human.human_id # NOTE: currently just use handedness to create output file
        result_folder = self.c.p.t_legacy_kvil_path
        for i in range(self.d.human.n_hands):
            name = f"hand_{i:>02d}"
            side = self.d.human.handedness[i]
            # hand_traj = self.d.human.graphormer_xyz.data[name] # raw traj
            # hand_traj = self.d.filtered_xyz.data[name] # filtered traj
            hand_traj = self.d.fixed_xyz.data[name] # fixed traj
            # ic(hand_traj.shape)
            out_file = result_folder / f"{side}_hand.npy"
            np.save(str(out_file), hand_traj)

        # down_sample_idx_file = self.c.p.t_rgb / "down_sample_index.yaml"
        # down_sample_idx = np.array(load_dict_from_yaml(down_sample_idx_file))
        obj_list = self.kvil_data.can.dcn_obj_cfg.keys()
        occ_index = {}
        down_sample_idx = np.array(self.d.rgb_down_sample_idx, dtype=int)
        for obj in obj_list:
            occ_cfg = self.d.vid_clip.occlude.get(obj, None)
            if occ_cfg is None:
                occ_index[obj] = None
                continue

            idx = np.where(occ_cfg.s <= down_sample_idx)[0][0]
            occ_index[obj] = idx
        raw_obj_traj = self.d.xyz
        for obj in self.kvil_data.can.dcn_obj_cfg.keys():
             #obj_traj_raw = self.d.xyz.data[obj] # raw traj
            # obj_traj = self.d.filtered_xyz.data[obj] # filtered traj
            obj_traj = self.d.fixed_xyz.data[obj] # fixed traj
            # ic(obj, obj_traj.shape)
            if occ_index[obj]:
                # occ_idx = self.d.vid_clip.occlude_s
                # if occ_idx > down_sample_idx.max():
                #     console.log(f"[bold red]The specified occlusion index {occ_idx} "
                #                 f"is larger than the downsampled range")
                #     continue
                # occ_start_idx = np.where(down_sample_idx > occ_idx)[0][0]
                # obj_traj[occ_index[obj]:] = self.d.filtered_xyz.data[obj][occ_index[obj] - 1] # filtered traj
                obj_traj[occ_index[obj]:] = self.d.fixed_xyz.data[obj][occ_index[obj] - 1] # fixed traj
                raw_obj_traj.data[obj][occ_index[obj]:] = self.d.xyz.data[obj][occ_index[obj] - 1]

            out_file = result_folder / f"{obj}.npy"
            np.save(str(out_file), obj_traj)
            # raw_out_file = result_folder / f"raw_{obj}.npy"
            # np.save(str(raw_out_file), obj_traj_raw)
        raw_out_file = result_folder / "raw_xyz.yaml"
        DictNumpyArray.to_yaml(raw_obj_traj, raw_out_file)

        if self.c.viz_demo:
            from vil.perception.viz.viz_demo import VizDemoTraj
            VizDemoTraj(self.c.p.t_dir, auto_viz_all=True)
        console.log(f"[bold cyan]done")

    def _create_kvil_canonical(self):
        if not self.c.create_legacy_kvil_canonical:
            return
        console.rule("creating legacy kvil canonical space")
        console.log("canonical space for objects ...")
        kvil_canonical_path = create_path(self.c.p.ns_path / "canonical")
        for obj in self.kvil_data.can.dcn_obj_cfg.keys():
            uv = self.kvil_data.can.uv[obj].tolist()
            obj_descriptor = self.kvil_data.can.descriptor[obj].numpy().tolist()
            coordinates = self.kvil_data.can.cand_pt[obj].tolist()
            obj_canonical_dict = {
                "uv": uv,
                "descriptors": obj_descriptor,
                "coordinates": coordinates
            }
            obj_canonical_file = kvil_canonical_path / f"{obj}.yaml"
            save_to_yaml(obj_canonical_dict, obj_canonical_file)

        console.log("canonical space for hands ...")
        self.c.reset_trial(0)
        self.c.steps_to_update = []
        # self._mediapipe_handedness()
        self._mmpose_wholebody()
        self._graphormer_hand()
        for i in range(self.d.human.n_hands):
            name = f"hand_{i:>02d}"
            side = self.d.human.handedness[i]
            hand_xyz = self.d.human.graphormer_xyz.data[name]
            hand_uv = self.d.human.graphormer_uv.data[name]
            init_uv = hand_uv[0].tolist()
            init_xyz = hand_xyz[0].tolist()
            intersection = np.arange(hand_uv.shape[1]).tolist()
            canonical_data = {
                "uv": init_uv,
                "descriptors": intersection,
                "coordinates": init_xyz
            }
            hand_canonical_file = kvil_canonical_path / f"{side}_hand.yaml"
            save_to_yaml(canonical_data, hand_canonical_file)

        config_file = create_path(self.c.p.ns_path / "config") / "_kvil_config.yaml"
        if not config_file.is_file():
            console.log(f"-- [bold green]creating _kvil_config: at {config_file}")
            from vil.cfg import KVILConfig
            config = KVILConfig()
            dump_data_to_yaml(KVILConfig, config, config_file)

        console.log(f"[bold cyan]done")
    
    # old code: filter first and then outlier detection
    def _filter_and_inlier_detect(self): 
        console.rule(f"filter and inlier detect")
        obj_filtered_file = self.c.p.t_obj / "filtered_xyz.yaml"
        if "filter_and_inlier_detect" not in self.c.steps_to_update and obj_filtered_file.is_file():
            self.d.filtered_xyz = DictNumpyArray.from_yaml(obj_filtered_file)
            console.log(f"[bold cyan]done (load)")
            return
        
        console.log(f"filtering the obj 3D trajectories...")
        inlier_idx_file = self.c.p.t_obj / "inlier_idx.yaml"
        inlier_idx_dict = {}
        for obj in self.kvil_data.can.dcn_obj_cfg.keys():
            T, P, dim = self.d.xyz.data[obj].shape
            filtered_xyz = savgol_filter(x=self.d.xyz.data[obj], window_length=9, polyorder=min(5, T - 1), deriv=0, mode='nearest', axis=0)
            inlier_idx, _ = outlier_det.outlier_detection(filtered_xyz)
            inlier_idx_dict[obj] = inlier_idx.tolist()
            self.d.filtered_xyz.data[obj] = filtered_xyz
        for hand_idx in range(self.d.human.n_hands):
            name = f"hand_{hand_idx:>02d}"
            # side = self.d.human.handedness[i]
            hand_traj = self.d.human.graphormer_xyz.data[name]
            filtered_xyz = savgol_filter(x=hand_traj, window_length=9, polyorder=min(5, T - 1), deriv=0, mode='nearest', axis=0)
            self.d.filtered_xyz.data[name] = filtered_xyz
        DictNumpyArray.to_yaml(self.d.filtered_xyz, obj_filtered_file)
        save_to_yaml(inlier_idx_dict, inlier_idx_file)
        console.log(f"[bold cyan]done")

    def _inlier_detect_and_outlier_fix(self):
        """ outlier detect first and then fix the outliers with inlier neighbors """
        console.rule(f"inlier detect and outlier fix")
        obj_fixed_file = self.c.p.t_obj / "fixed_xyz.yaml"
        if "inlier_detect_and_outlier_fix" not in self.c.steps_to_update and obj_fixed_file.is_file():
            self.d.fixed_xyz = DictNumpyArray.from_yaml(obj_fixed_file)
            console.log(f"[bold cyan]done (load)")
            return
        
        console.log(f"start to detect and fix outliers...")
        inlier_idx_dict = {}

        raw_depth = self.d.depth_array
        n_neighbor = 30 #TODO: make the n_neighbor in the config file
        for obj in self.kvil_data.can.dcn_obj_cfg.keys():
            raw_xyz = self.d.xyz.data[obj] # T, P, 3
            raw_uv = self.d.uv.data[obj] # T, P, 2
            T, P, dim = raw_xyz.shape

            # find points that are within 30cm to the camera, consider as outlier
            close_to_cam_point_idx = np.where(np.linalg.norm(raw_xyz, axis=-1) < 0.2)
            exclude_idx = np.unique(close_to_cam_point_idx[-1])

            inlier_idx, outlier_idx = outlier_det.outlier_detection(raw_xyz, exclude_idx=exclude_idx)
            neighbors = find_neighbors(raw_xyz[0], n_neighbor) # P, n_neighbor

            if outlier_idx.shape[0] == 0:
                inlier_idx_dict[obj] = inlier_idx.tolist()
                self.d.fixed_xyz.data[obj] = self.d.xyz.data[obj]
                console.log(f"{obj} has no outliers in this trail, skipping")
                continue

            masked_state = outlier_det.get_neighbor_and_inlier_state_matrix(P, inlier_idx, neighbors)

            fixed_depth = []
            for t in range(T):
                fixed_depth.append(outlier_det.fix_depth_for_outlier_in_one_frame(
                    masked_state, raw_depth[t], raw_uv[t]
                ))
            
            fixed_xyz = np.array(self.pool.map(
                partial(pinhole_projection_image_to_camera, intrinsic=self.d.cam.get_intrinsics(flag_left=True)),
                self.d.uv.data[obj], np.array(fixed_depth)  # (T, 300, 2), [T x (h, w)]
            ))
            
            # get the new outliers and filter them with a low order filter
            inlier_idx, outlier_idx = outlier_det.outlier_detection(fixed_xyz, exclude_idx=None)
            filtered_xyz = np.asarray(fixed_xyz)
            filtered_xyz[:, inlier_idx] = savgol_filter(
                x=fixed_xyz[:, inlier_idx], window_length=9, polyorder=min(5, T - 1), deriv=0, mode='nearest', axis=0
            )
            filtered_xyz[:, outlier_idx] = savgol_filter(
                x=fixed_xyz[:, outlier_idx], window_length=7, polyorder=min(1, T - 1), deriv=0, mode='nearest', axis=0
            )
            # self.d.fixed_xyz.data[obj] = savgol_filter(x=fixed_xyz, window_length=9, polyorder=min(5, T - 1), deriv=0, mode='nearest', axis=0)
            self.d.fixed_xyz.data[obj] = filtered_xyz

            inlier_idx, outlier_idx = outlier_det.outlier_detection(filtered_xyz, exclude_idx=None)
            final_outlier = outlier_idx.shape[0]
            inlier_idx_dict[obj] = inlier_idx.tolist()

            console.log(f"done (outlier traj fix of object {obj})")
            console.log(f"     initial number of outlier: {outlier_idx.shape[0]}, remaining {final_outlier} after fix")

        for hand_idx in range(self.d.human.n_hands):
            name = f"hand_{hand_idx:>02d}"
            # side = self.d.human.handedness[i]
            hand_traj = self.d.human.graphormer_xyz.data[name]
            filtered_xyz = savgol_filter(
                x=hand_traj, window_length=9, polyorder=min(5, hand_traj.shape[0] - 1), deriv=0, mode='nearest', axis=0
            )
            self.d.fixed_xyz.data[name] = filtered_xyz
            console.log(f"done (traj filter of {name})")
            
        DictNumpyArray.to_yaml(self.d.fixed_xyz, obj_fixed_file)
        inlier_idx_file = self.c.p.t_obj / "inlier_idx.yaml"
        save_to_yaml(inlier_idx_dict, inlier_idx_file)
        console.log(f"[bold cyan]done")
 
    def _to_video(self):
        if not self.c.to_video:
            return
        console.rule("generate video of preprocessing results")
        force_redo = "to_video" in self.c.steps_to_update
        vid_map = dict(
            # rgb=self.c.p.t_rgb,
            # depth=self.c.p.t_depth,
            mask=self.c.p.t_mask / "all",
            mesh_graphormer_mesh=self.c.p.v_human / "graphormer/mesh",
            # mesh_graphormer_uv=self.c.p.v_human / "graphormer/uv",
            flow_uv=self.c.p.v_flow / "uv"
        )
        vid_pattern = dict(
            rgb=[".jpg"],
            depth=["v_", ".jpg"]
        )
        for file_stem, path in vid_map.items():
            vid_file = self.c.p.t_dir_vid / f"{file_stem}.mp4"
            pattern = vid_pattern.get(file_stem, None)
            if force_redo or not vid_file.is_file():
                image_to_video(path, vid_file, "rgb", pattern, fps=10, codec="mp4v")
            # gif_file = self.c.p.t_dir_vid / f"{file_stem}.gif"
            # if force_redo or not gif_file.is_file():
            #     image_to_gif(path, pattern, gif_file, 3, 0)

    def process(self):
        self.kvil_data = KVILPreProcessingData()
        self.kvil_data.can.load_dcn_cfg(self.c.p.can / "dcn/canonical_cfg.yaml")
        for trial_idx in range(self.c.p.n_trials):
            self.c.reset_trial(trial_idx)
            self.kvil_data.demos.append(KVILDemoData())
            self.d = self.kvil_data.demos[trial_idx]
            if "extract" not in self.c.steps_to_update:
                self.d.load_scene_graph(self.c.p.t_scene)
                self.d.load_cam_intrinsics(get_validate_file(self.c.p.t_param), self.c.mono)
                self.d.load_segmentation(get_validate_file(self.c.p.t_seg), self.c.namespace)

            console.rule(f"[bold yellow]Processing trial {self.c.p.t_dir}")

            for p in self.c.process_order:
                self.process_map[p]()

            self.results.clear()
        self._create_kvil_canonical()


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--task_path",        "-p", type=str, help="the absolute path to task demo")
@click.option("--namespace",        "-n", type=str, help="the namespace to work on")
@click.option("--target_frames",    "-t", type=int, default=30, help="the desired number of frames")
@click.option("--viz",              "-v", is_flag=True, help="visualize intermediate results")
@click.option("--viz_demo",         "-vd", is_flag=True, help="visualize points in polyscope for each trial")
@click.option("--to_kvil_can",      "-k", is_flag=True, help="generate legacy version of kvil canonical space")
@click.option("--to_video",         "-tv", is_flag=True, help="generate videos of visualization results")
@click.option("--mono", is_flag=True, help="whether to use mono camera")
def main(task_path, namespace, target_frames, viz, viz_demo, to_kvil_can, to_video, mono):
    os.environ["TOKENIZERS_PARALLELISM"] = 'false'
    if not namespace:
        console.log("[bold red]You have to specify namespace with '-n'")
        exit()
    cfg = KVILPerceptionConfig(task_path, target_frames, namespace, viz, viz_demo,
                               create_legacy_kvil_canonical=to_kvil_can, to_video=to_video, mono=mono)
    p = KVILPreprocessing(cfg)
    p.process()


if __name__ == "__main__":
    main()

# TODO
#   cd path_to_kvil/vil/perception/
#   python demo_preprocessing.py -p /home/gao/dataset/kvil/test/demo_pour_new_structure/ -t 30 -v -n kvil
#   python demo_preprocessing.py -p /home/gao/dataset/kvil/test/demo_pour_new_structure/ -t 30 -v -n motion_seg
