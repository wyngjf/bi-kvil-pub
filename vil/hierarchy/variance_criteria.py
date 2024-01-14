import os
import numpy as np
import rich.box
from rich.table import Table
from typing import List, Literal
from pathos.multiprocessing import ProcessingPool as Pool

from robot_utils import console
from robot_utils.math.transformations import quaternion_matrix, quaternion_from_matrix

from vil.hierarchy.gaussian_distribution_from_data import compute_frechet_mean, compute_manifold_covariance


class VarianceCriteria:
    def __init__(self, static_obj_list: List['RealObject'], moving_obj_list: List['RealObject'], force_redo: bool):
        self.force_redo = force_redo
        self.normalized_covar_tr_ori: np.ndarray = np.empty(0)
        self.normalized_covar_tr_pos: np.ndarray = np.empty(0)

        self.min_ratio_ori: float = 0.0
        self.min_ratio_pos: float = 0.0

        self.least_salient_obj_pos: 'RealObject' = None
        self.least_salient_obj_ori: 'RealObject' = None

        self.static_obj_list = static_obj_list
        self.moving_obj_list = moving_obj_list

    def _table_message(self, title: str, data: np.ndarray):
        table = Table(title=f"[bold yellow]Variance-based MS Detection ({title})",
                      box=rich.box.HORIZONTALS)
        table.add_column("static \ moving object", justify="right", style="cyan")
        for moving_obj in self.moving_obj_list:
            table.add_column(f"{moving_obj.c.name}", justify="center", style="green")

        for i, static_obj in enumerate(self.static_obj_list):
            table.add_row(f"{static_obj.c.name}", *[f"{data[i][j]:>.4f}" for j in range(len(self.moving_obj_list))])

        console.log(table)

    def run(self):
        console.rule("[bold green]Variance Criteria")
        if len(self.moving_obj_list) <= 1:
            console.log("[bold red]less than 2 objects are moving, not need to use variance criteria")
        else:
            self._compute_variance_ori()
            self._compute_variance_pos()

            if self.min_ratio_ori < self.min_ratio_pos:
                master = self.least_salient_obj_ori
                console.log(f"[bold cyan]{master.c.name} is master due to lower orientation variation")
            else:
                master = self.least_salient_obj_pos
                console.log(f"[bold cyan]{master.c.name} is master due to lower position variation")

            master.set_as_master(remove_slaves=False)
            for obj in self.moving_obj_list:
                if obj.c.index == master.c.index:
                    continue
                master.append_object(obj, is_slave=True)
            # master.d.save("variance criteria: append moving obj")
        console.log("[bold]Done (variance criteria)")

    def _compute_variance_pos(self):
        console.log("[bold green]Compute positional variances")
        normalized_pos_covar_tr = []
        for v_idx, virtual_obj in enumerate(self.static_obj_list):
            pos_covar_tr_list = []
            virtual_obj.init_kvil()
            if self.force_redo:
                virtual_obj.pca()
                # virtual_obj.d.save("variance criteria)
            else:
                virtual_obj.d.load()

            for obj in self.moving_obj_list:
                pca_res_var_sum = np.sum(virtual_obj.d.pca_data[obj.c.name][-1, :, :, -1, :], axis=-1)
                min_var_sum_kpt_idx = np.unravel_index(np.argmin(pca_res_var_sum), pca_res_var_sum.shape)
                pos_covar_tr_list.append(pca_res_var_sum[min_var_sum_kpt_idx])
                console.log(f"-- observe {obj.c.name}: {pca_res_var_sum[min_var_sum_kpt_idx]}")

            pos_covar_tr_list = np.array(pos_covar_tr_list)
            pos_covar_tr_list = pos_covar_tr_list / np.max(pos_covar_tr_list)
            normalized_pos_covar_tr.append(pos_covar_tr_list)

        normalized_covar_tr = np.array(normalized_pos_covar_tr)
        self.min_ratio_pos = normalized_covar_tr.min()
        min_ratio_idx = np.unravel_index(np.argmin(normalized_covar_tr), normalized_covar_tr.shape
        )
        self.least_salient_obj_pos = self.moving_obj_list[min_ratio_idx[1]]
        self._table_message("Position", normalized_covar_tr)

    def _compute_variance_ori(self):
        console.log("[bold green]Compute orientation variances")
        normalized_covar_tr = []

        for virtual_obj in self.static_obj_list:
            console.log(f"[bold]For object {virtual_obj.c.name}")
            v_obj_rot_mat_at_T = virtual_obj.frame_mat_all_demo[:, -1, :, :, :3]
            # N,dim,_ = v_obj_rot_mat_at_T.shape
            N = v_obj_rot_mat_at_T.shape[0]
            covar_tr_list = []
            for obj in self.moving_obj_list:
                object_orient_at_T = obj.frame_mat_all_demo[:, -1, :, :, :3]
                obj_local_orient_in_quat = []
                for demo_idx in range(N):
                    pool = Pool(os.cpu_count())
                    all_v_quat_this_demo = np.array(
                        pool.map(quaternion_from_matrix, v_obj_rot_mat_at_T[demo_idx]))
                    mean_v_ori_this_demo = compute_frechet_mean(all_v_quat_this_demo)

                    # mean_v_ori_this_demo = v_obj_rot_mat_at_T[demo_idx]
                    # mean_v_obj_rot_mat_at_T.append(mean_v_ori_this_demo)
                    all_r_quat_this_demo = np.array(
                        pool.map(quaternion_from_matrix, object_orient_at_T[demo_idx]))
                    mean_r_ori_this_demo = compute_frechet_mean(all_r_quat_this_demo)

                    # mean_r_ori_this_demo = object_orient_at_T[demo_idx]
                    # mean_object_orient_at_T.append(mean_r_ori_this_demo)
                    obj_local_orient_this_demo = np.einsum(
                        "ij,jk->ik",
                        np.linalg.inv(quaternion_matrix(mean_v_ori_this_demo, rot_only=True)),
                        quaternion_matrix(mean_r_ori_this_demo, rot_only=True)
                    )
                    obj_local_quat_this_demo = quaternion_from_matrix(obj_local_orient_this_demo)
                    obj_local_orient_in_quat.append(obj_local_quat_this_demo)
                    # ic(demo_idx,virtual_obj.c.name,real_obj.c.name,obj_local_quat_this_demo)
                    # ic(all_v_quat_this_demo.shape,mean_v_ori_this_demo)

                # obj_local_orient = np.einsum("nij,njk->nik",mean_v_obj_rot_mat_at_T,mean_object_orient_at_T)

                obj_local_orient_in_quat = np.array(obj_local_orient_in_quat)
                ori_mean = compute_frechet_mean(obj_local_orient_in_quat)
                ori_mean = ori_mean / np.linalg.norm(ori_mean)
                ori_covar = compute_manifold_covariance(obj_local_orient_in_quat, ori_mean)
                orient_covar_eigval = np.linalg.eigvals(ori_covar)
                covar_tr = np.sum(orient_covar_eigval)
                covar_tr_list.append(covar_tr)
                console.log(f"-- observe {obj.c.name}: {covar_tr}")

            covar_tr_list = np.array(covar_tr_list)
            covar_tr_list = covar_tr_list / np.max(covar_tr_list)
            normalized_covar_tr.append(covar_tr_list)

        normalized_covar_tr = np.array(normalized_covar_tr)
        self.min_ratio_ori = normalized_covar_tr.min()
        min_ratio_idx = np.unravel_index(np.argmin(normalized_covar_tr), normalized_covar_tr.shape
        )
        self.least_salient_obj_ori = self.moving_obj_list[min_ratio_idx[1]]
        self._table_message("orientation", normalized_covar_tr)

