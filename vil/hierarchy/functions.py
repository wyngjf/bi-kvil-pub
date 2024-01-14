import os
import numpy as np
from scipy.signal import savgol_filter
from pathos.multiprocessing import ProcessingPool as Pool
from robot_utils import console


def create_local_frame_on_hand(traj_hand:np.ndarray):
    '''
    traj_hand.shape = T, P, Dim = T, 21, 3
    output: hand_fram_mats.shape = T, Dim, Dim + 1 = T, 3, 4
    '''
    T, P, dim = traj_hand.shape
    frame_mat = []
    for t in range(T):
        traj = traj_hand[t]
        wrist = traj[0]
        index_mcp = traj[5]
        middle_mcp = traj[9]
        pinky_mcp = traj[17]
        local_z = middle_mcp - wrist
        local_z = local_z / np.linalg.norm(local_z)

        local_y = (pinky_mcp - index_mcp) / np.linalg.norm(pinky_mcp - index_mcp)
        local_y = local_y - np.dot(local_z, local_y) * local_z
        local_y /= np.linalg.norm(local_y)

        local_x = np.cross(local_y, local_z)

        rotation_matrix = np.column_stack((local_x, local_y, local_z))
        frame_mat.append(np.column_stack((rotation_matrix, middle_mcp)))

    return np.array(frame_mat)

def parallel_min_distances(hand_traj:np.ndarray, object_traj:np.ndarray):
    """
    obtain the min distance between each point in the obj pcl and the hand pcl at each time step
    - inputs:
        - object_traj.shape =  T, P_obj, Dim, e.g. P_obj = 300
        - hand_traj.shape = T, P_hand, Dim = T, 21, 3
    - output: 
        - min_distances.shape = T, P_obj
    """
    def calculate_min_distance_for_a_time_step(hand_cloud:np.ndarray, obj_cloud:np.ndarray): # obj_cloud.shape = P, Dim, hand_clpud.shape = P, Dim = 21, 3
        min_distances_at_t = []
        for point in obj_cloud:
            min_distances_at_t.append(np.min(np.linalg.norm(hand_cloud - point, axis=-1)))
        min_distances_at_t = np.array(min_distances_at_t)
        return min_distances_at_t # (P,)
    
    pool = Pool(os.cpu_count())
    min_distances = np.array(pool.map(
        calculate_min_distance_for_a_time_step,
        hand_traj,
        object_traj
    )) 
    return min_distances 

def get_obj_dist_weights_from_hand(hand_traj:np.ndarray, object_traj:np.ndarray):
    """
    hand_traj.shape = T, P, Dim = T, 21, 3
    object_traj.shape = T, P, Dim
    output: weights.shape = T, P weight is a float between [0,1], the closer the point to the hand, the higher the weight is
    """
    T, P, Dim = hand_traj.shape
    weights = []
    min_distances = parallel_min_distances(hand_traj, object_traj)
    # ic(min_distances.shape)
    for t in range(T):
        weights.append(1 - min_distances[t] / np.max(min_distances[t]) + np.min(min_distances[t]) / np.max(min_distances[t])) 
        """
        distance between [min, max] -> weights between [min/max, 1] with min distance = 1, max distance = min/max
        """
    weights = np.array(weights) 
    # ic(weights.shape, np.max(weights), np.min(weights))
    return weights.astype(float)


def find_n_nearest_points(base_object_traj:np.ndarray, target_object_traj:np.ndarray, num_points:int=30):
    """
    - find the nearest {num_points} points to the base_obj on the target_object
    - input:
        - base_obj_traj.shape = T, P, Dim (for hand: T, 21, 3 )
        - target_object_traj.shape = T, P, Dim
    - output: 
        - nearest_points.shape = T, num_points
    """
    T, P, Dim = base_object_traj.shape
    nearest_points = []
    min_distances = parallel_min_distances(base_object_traj, target_object_traj)
    for t in range(T):
        nearest_points.append(np.argsort(min_distances[t],axis=-1)[:num_points])
    nearest_points = np.array(nearest_points) # T, num_points
    return nearest_points

def detect_change_of_norm(orient_traj:np.ndarray, object_traj:np.ndarray):
    """
    - detect the change of the norm of the offset between several point on an object to a base_point(orient of the local frame)
    - input:
        - orient_traj.shape = T, Dim 
        - object_traj.shape = T, P, Dim
    - output: 
        - change_of_norm.shape = T, P
    """
    T, Dim = orient_traj.shape
    norm = []
    for t in range(T):
        norm.append(np.linalg.norm(object_traj[t] - orient_traj[t], axis=-1))
    norm = np.array(norm)
    padding = np.expand_dims(norm[0], axis=0)
    # ic(padding.shape, norm.shape)
    padded_norm = np.concatenate((padding, norm), axis=0)
    change_of_norm = []
    for t in range(T):
        change_of_norm.append(np.abs(padded_norm[t + 1] - padded_norm[t]))
    change_of_norm = np.array(change_of_norm)
    # ic(change_of_norm.shape)
    return change_of_norm

def get_weighted_norm_change(hand_traj:np.ndarray, object_traj:np.ndarray, num_points:int=30):
    """
    - detect the change of the norm of the offset between several point on an object to a base_point(orient of the local frame)
    - input:
        - hand_traj.shape = T, P, Dim 
        - object_traj.shape = T, P, Dim
    - output: 
        - weighted_norm_change.shape = T, -> the weighted sum of the norm change according to the weights based on the distance
    """
    hand_lf_traj = np.expand_dims(hand_traj[:,9], axis=1)
    nearest_points = find_n_nearest_points(hand_lf_traj, object_traj, num_points)
    norm_change = detect_change_of_norm(hand_traj[:,9], object_traj)
    weights = get_obj_dist_weights_from_hand(hand_traj, object_traj)  
    weighted_norm_change = []
    for t in range(hand_traj.shape[0]):
        weighted_norm_change.append(np.dot(norm_change[t,nearest_points[t]], weights[t, nearest_points[t]]) / num_points)
        # ic(np.min(weights[t, nearest_points[t]]), np.max(weights[t, nearest_points[t]]))
    weighted_norm_change = np.array(weighted_norm_change)
    return weighted_norm_change


def get_xyz_scaling(data):
    x_value = data[:,0]
    y_value = data[:,1]
    z_value = data[:,2]
    x_scaling = np.max(x_value) - np.min(x_value) 
    y_scaling = np.max(y_value) - np.min(y_value) 
    z_scaling = np.max(z_value) - np.min(z_value) 
    return x_scaling,y_scaling,z_scaling


# def global_saliency_detection(global_traj) -> bool:
def is_global_static(global_traj: np.ndarray) -> bool:
    """

    Args:
        global_traj:

    Returns: if the object does not move TODO

    """
    N, T, P, dim = global_traj.shape
    trail_total_fixed_flag = []
    for i in range(N):
        trail_traj = global_traj[i] * 1000
        pcl = trail_traj[0]
        sg_x = savgol_filter(x=trail_traj,window_length=15,polyorder=min(2, T-1),deriv=0,mode='nearest',axis=0)
        sg_v = savgol_filter(x=trail_traj,window_length=15,polyorder=min(2, T-1),deriv=1,mode='nearest',axis=0)
        sum_sg_v = np.sum(sg_v,axis=1) / P
        norm_sg_v = np.linalg.norm(sum_sg_v,axis=-1)
        delta_sg_x = sg_x - sg_x[-1]
        avg_delta_sg_x = np.sum(delta_sg_x,axis=1) / P     
        obj_size_in_this_trail = get_xyz_scaling(pcl)
        fix_threshold = np.array(obj_size_in_this_trail) * 0.1
        v_threshold = 5
        moving_flag = []
        for t in range(T):
            if (avg_delta_sg_x[t][0] < fix_threshold[0]) and (avg_delta_sg_x[t][1] < fix_threshold[1]) and (avg_delta_sg_x[t][2] < fix_threshold[2]) and (norm_sg_v[t] < v_threshold):
                moving_flag.append(0)
            else:
                moving_flag.append(1) 

        if sum(moving_flag) < T * 0.1:
            trail_total_fixed_flag.append(0)
        else:
            trail_total_fixed_flag.append(1)

    return sum(trail_total_fixed_flag) < N * 0.1


def get_grasping_flags(hand_traj:np.ndarray, object_traj:np.ndarray, fps_factor:float=1.0):
    """
    - detect if the object is grasped in a hand
    - input:
        - hand_traj.shape = N, T, P, Dim
        - object_traj.shape = N, T, P, Dim
        - fps_factor: float -> the factor to the speed threshold calculated based on the frame rate of the down sampled demo
    - output:
        - grasping_flags.shape = (N, T), dtype = int {0, 1} with 1: grasping at this time step, 0: not grasping
    """
    N, T, P, Dim = hand_traj.shape
    grasping_flags = []
    distance_threshold = 100
    trans_speed_threshold = 5 / fps_factor
    for i in range(N):
        trail_hand_traj = hand_traj[i] * 1000
        trail_obj_traj = object_traj[i] * 1000
        weighted_norm_change = get_weighted_norm_change(trail_hand_traj, trail_obj_traj)
        min_distances = parallel_min_distances(trail_hand_traj, trail_obj_traj)
        grasping_flags_this_trail = []
        for t in range(T):
            min_p2p_dist = np.min(min_distances[t])
            if min_p2p_dist < distance_threshold and weighted_norm_change[t] < trans_speed_threshold:
                grasping_flags_this_trail.append(1)
            else:
                grasping_flags_this_trail.append(0)
        grasping_flags.append(grasping_flags_this_trail)
    return np.array(grasping_flags, dtype=int)

def grasp_detection(hand_traj:np.ndarray, object_traj:np.ndarray, fps_factor:float=1.0):
    """
    - detect if the object is grasped in a hand
    - input:
        - hand_traj.shape = N, T, P, Dim
        - object_traj.shape = N, T, P, Dim
        - fps_factor: float -> the factor to the speed threshold calculated based on the frame rate of the down sampled demo
    - output:
        - True: the object is grasped, False: not grasped
    """
    grasping_flags = get_grasping_flags(hand_traj, object_traj, fps_factor)
    grasp_ratio = grasping_flags.mean(-1)
    grasp_ratio_text = "  ".join([f'{r:>1.2f}' for r in grasp_ratio])
    console.log(f"-- [grey]median grasp ratio = {np.median(grasp_ratio)} -- [{grasp_ratio_text} ]")
    return np.median(grasp_ratio) > 0.7


def symmetric_detection(hand_a_traj:np.ndarray, hand_b_traj:np.ndarray, object_traj:np.ndarray, fps_factor:float=1.0):
    """
    - only if two hands are detected to grasp the same object, detect whether the grasping is symmetric
    - inputs:
        - hand_a_traj.shape = N, T, P, Dim
        - hand_b_traj.shape = N, T, P, Dim
        - object_traj.shape = N, T, P, Dim
        - fps_factor: float -> the factor to the speed threshold calculated based on the frame rate of the down sampled demo
    - output:
        - True: symmetric, False: asymmetric
    """
    hand_a_grasping_flags = get_grasping_flags(hand_a_traj, object_traj, fps_factor)
    hand_b_grasping_flags = get_grasping_flags(hand_b_traj, object_traj, fps_factor)
    N, T = hand_a_grasping_flags.shape
    trans_speed_threshold = 5 / fps_factor
    symmetric_grasp_flags = []
    for n in range(N):
        hand_lf_relative_trans_speed = get_weighted_norm_change(hand_a_traj[n] * 1000, np.expand_dims(hand_b_traj[n,:,9] * 1000, axis=-2), num_points=1)
        # ic(hand_lf_relative_trans_speed)
        trail_symmetric_grasp_flags = []
        for t in range(T):
            if hand_a_grasping_flags[n,t] and hand_b_grasping_flags[n,t] and hand_lf_relative_trans_speed[t] < trans_speed_threshold:
                trail_symmetric_grasp_flags.append(1)
            else:
                trail_symmetric_grasp_flags.append(0)
        if np.sum(np.array(trail_symmetric_grasp_flags)) > 0.4 * T:
            symmetric_grasp_flags.append(1)
        else:
            symmetric_grasp_flags.append(0)
    # ic(symmetric_grasp_flags)
    return np.sum(np.array(trail_symmetric_grasp_flags)) > 0.9 * N
