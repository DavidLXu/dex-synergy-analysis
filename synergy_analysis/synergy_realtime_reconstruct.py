import multiprocessing
import pickle
import time
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import numpy as np
import sapien
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer
from sklearn.decomposition import PCA

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector


def load_pca_parameters(pkl_file: str):
    """Load PCA parameters from a previously saved pkl file"""
    print(f"Loading PCA parameters from {pkl_file}...")
    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    qpos_data = data['qpos']
    joint_names = data['joint_names']
    robot_name = data.get('robot_name', 'unknown')
    hand_type = data.get('hand_type', 'unknown')
    
    print(f"Loaded training data: {len(qpos_data)} recordings with {qpos_data.shape[1]} joints")
    print(f"Robot: {robot_name}, Hand: {hand_type}")
    print(f"Joint names: {joint_names}")
    
    return {
        'training_data': qpos_data,
        'joint_names': joint_names,
        'robot_name': robot_name,
        'hand_type': hand_type
    }


def start_realtime_synergy_reconstruction(
    queue: multiprocessing.Queue, 
    robot_dir: str, 
    config_path: str, 
    pkl_file: str, 
    n_components: int = 5
):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start realtime synergy reconstruction with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    
    # Load PCA parameters from previous data
    pca_data = load_pca_parameters(pkl_file)
    training_qpos = pca_data['training_data']
    
    # Fit PCA on training data
    print(f"Fitting PCA with {n_components} components on training data...")
    pca = PCA(n_components=n_components)
    pca.fit(training_qpos)
    
    # Print PCA analysis results
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print("\n" + "="*50)
    print("PCA Parameters Loaded:")
    print("="*50)
    for i, (var_ratio, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
        print(f"  PC{i+1}: {var_ratio:.3f} ({cum_var:.3f} cumulative)")
    print(f"Total variance explained: {cumulative_variance[-1]:.3f}")
    print("="*50)

    hand_type = "Right" if "right" in config_path.lower() else "Left"
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)

    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    config = RetargetingConfig.load_from_file(config_path)

    # Setup
    scene = sapien.Scene()
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # Lighting
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )
    scene.add_area_light_for_ray_tracing(
        sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
    )

    # Camera
    cam = scene.add_camera(
        name="Cheese!", width=800, height=600, fovy=1, near=0.1, far=10
    )
    cam.set_local_pose(sapien.Pose([0.75, 0, 0.1], [0, 0, 0, -1]))

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = False
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())

    # Load robots and set them to good poses to take picture
    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
    loader.load_multiple_collisions_from_file = True
    if "ability" in robot_name:
        loader.scale = 1.5
    elif "dclaw" in robot_name:
        loader.scale = 1.25
    elif "allegro" in robot_name:
        loader.scale = 1.4
    elif "shadow" in robot_name:
        loader.scale = 0.9
    elif "bhand" in robot_name:
        loader.scale = 1.5
    elif "leap" in robot_name:
        loader.scale = 1.4
    elif "svh" in robot_name:
        loader.scale = 1.5

    if "glb" not in robot_name:
        filepath = str(filepath).replace(".urdf", "_glb.urdf")
    else:
        filepath = str(filepath)

    # Load two identical robots
    robot1 = loader.load(filepath)
    robot2 = loader.load(filepath)

    # Base z position for different robots
    base_z = 0
    if "ability" in robot_name:
        base_z = -0.15
    elif "shadow" in robot_name:
        base_z = -0.2
    elif "dclaw" in robot_name:
        base_z = -0.15
    elif "allegro" in robot_name:
        base_z = -0.05
    elif "bhand" in robot_name:
        base_z = -0.2
    elif "leap" in robot_name:
        base_z = -0.15
    elif "svh" in robot_name:
        base_z = -0.13

    # Position robots side by side
    robot1.set_pose(sapien.Pose([0, 0.15, base_z]))  # Left robot - PCA reconstructed
    robot2.set_pose(sapien.Pose([0, -0.15, base_z]))   # Right robot - Original

    # Different robot loader may have different orders for joints
    sapien_joint_names = [joint.get_name() for joint in robot1.get_active_joints()]
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)

    print(f"\nStarting realtime synergy reconstruction...")
    print("Controls: 'q' to quit")
    print("Left robot: PCA reconstructed (realtime)")
    print("Right robot: Original retargeted poses")
    print(f"Using {n_components} principal components")
    
    # Statistics tracking
    frame_count = 0
    successful_detections = 0
    reconstruction_errors = []

    while True:
        try:
            bgr = queue.get(timeout=5)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Empty:
            logger.error(
                "Fail to fetch image from camera in 5 secs. Please check your web camera device."
            )
            return

        frame_count += 1
        
        _, joint_pos, keypoint_2d, _ = detector.detect(rgb)
        bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
        
        # Add overlay information
        info_text = f"Frame: {frame_count} | Detections: {successful_detections} | PCA: {n_components} components"
        cv2.putText(bgr, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        success_rate = (successful_detections / frame_count * 100) if frame_count > 0 else 0
        rate_text = f"Success rate: {success_rate:.1f}%"
        cv2.putText(bgr, rate_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        if len(reconstruction_errors) > 0:
            error_text = f"Avg reconstruction error: {np.mean(reconstruction_errors[-30:]):.4f}"
            cv2.putText(bgr, error_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        cv2.imshow("Realtime Synergy Reconstruction", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if joint_pos is None:
            logger.warning(f"{hand_type} hand is not detected.")
        else:
            successful_detections += 1
            
            # Get original retargeted pose
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                indices = indices
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            qpos_original = retargeting.retarget(ref_value)
            
            # Apply PCA decomposition and reconstruction in realtime
            qpos_original_reshaped = qpos_original.reshape(1, -1)  # Reshape for PCA
            qpos_pca_coefficients = pca.transform(qpos_original_reshaped)  # Decompose
            qpos_reconstructed = pca.inverse_transform(qpos_pca_coefficients)  # Reconstruct
            qpos_reconstructed = qpos_reconstructed.flatten()  # Flatten back
            
            # Calculate reconstruction error
            reconstruction_error = np.mean(np.abs(qpos_original - qpos_reconstructed))
            reconstruction_errors.append(reconstruction_error)
            
            # Print statistics every 100 successful detections
            if successful_detections % 100 == 0:
                avg_error = np.mean(reconstruction_errors[-100:])
                print(f"Detections: {successful_detections}, Avg reconstruction error (last 100): {avg_error:.4f}")
            
            # Apply joint positions to robots
            robot1.set_qpos(qpos_reconstructed[retargeting_to_sapien])  # Left - PCA reconstructed
            robot2.set_qpos(qpos_original[retargeting_to_sapien])       # Right - Original

        for _ in range(2):
            viewer.render()
    
    # Print final statistics
    print(f"\nRealtime synergy reconstruction completed!")
    print(f"Total frames: {frame_count}")
    print(f"Successful detections: {successful_detections}")
    print(f"Success rate: {successful_detections/frame_count*100:.1f}%")
    if reconstruction_errors:
        print(f"Average reconstruction error: {np.mean(reconstruction_errors):.4f}")
        print(f"Min reconstruction error: {np.min(reconstruction_errors):.4f}")
        print(f"Max reconstruction error: {np.max(reconstruction_errors):.4f}")


def produce_frame(queue: multiprocessing.Queue, camera_path: Optional[str] = None):
    if camera_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(camera_path)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        queue.put(image)


def main(
    pkl_file: str,
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    n_components: int = 5,
    camera_path: Optional[str] = None,
):
    """
    Performs realtime synergy reconstruction using PCA parameters from a previous pkl file.
    Shows original retargeted poses vs PCA-reconstructed poses in realtime.

    Args:
        pkl_file: Path to the pkl file containing PCA training data.
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
        n_components: Number of principal components to use for PCA reconstruction (default: 5).
        camera_path: the device path to feed to opencv to open the web camera. It will use 0 by default.
    """
    if not Path(pkl_file).exists():
        print(f"Error: pkl file not found: {pkl_file}")
        return
    
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent / "assets" / "robots" / "hands"
    )

    print(f"Loading configuration: {config_path}")
    print(f"Robot directory: {robot_dir}")
    print(f"PCA training file: {pkl_file}")
    print(f"PCA components: {n_components}")
    print(f"Camera path: {camera_path or 'default (0)'}")

    queue = multiprocessing.Queue(maxsize=1000)
    producer_process = multiprocessing.Process(
        target=produce_frame, args=(queue, camera_path)
    )
    consumer_process = multiprocessing.Process(
        target=start_realtime_synergy_reconstruction, 
        args=(queue, str(robot_dir), str(config_path), pkl_file, n_components)
    )

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()
    time.sleep(2)

    print("Realtime synergy reconstruction completed!")


if __name__ == "__main__":
    tyro.cli(main) 