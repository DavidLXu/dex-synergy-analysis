import pickle
import time
from pathlib import Path
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


def load_and_analyze_qpos_data(pkl_file: str, n_components: int = 5):
    """Load qpos data from pkl file and perform PCA analysis"""
    print(f"Loading data from {pkl_file}...")
    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    qpos_data = data['qpos']
    timestamps = data['timestamps']
    joint_names = data['joint_names']
    
    # Get additional metadata if available
    robot_name = data.get('robot_name', 'unknown')
    hand_type = data.get('hand_type', 'unknown')
    source_video = data.get('source_video', None)
    
    print(f"Loaded {len(qpos_data)} recordings with {qpos_data.shape[1]} joints")
    print(f"Robot: {robot_name}, Hand: {hand_type}")
    if source_video:
        print(f"Source video: {source_video}")
    print(f"Joint names: {joint_names}")
    
    # Perform PCA
    print(f"Performing PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    qpos_pca = pca.fit_transform(qpos_data)
    qpos_reconstructed = pca.inverse_transform(qpos_pca)
    
    # Print PCA analysis results
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print("\n" + "="*50)
    print("PCA Analysis Results:")
    print("="*50)
    for i, (var_ratio, cum_var) in enumerate(zip(explained_variance_ratio, cumulative_variance)):
        print(f"  PC{i+1}: {var_ratio:.3f} ({cum_var:.3f} cumulative)")
    
    print(f"\nTotal variance explained by {n_components} components: {cumulative_variance[-1]:.3f}")
    
    # Calculate reconstruction error
    mse = np.mean((qpos_data - qpos_reconstructed) ** 2)
    rmse = np.sqrt(mse)
    print(f"Mean Squared Error (reconstruction): {mse:.6f}")
    print(f"Root Mean Squared Error (reconstruction): {rmse:.6f}")
    
    # Show top contributing joints for first few PCs
    print(f"\nTop contributing joints for first 3 principal components:")
    for pc_idx in range(min(3, n_components)):
        pc_loadings = np.abs(pca.components_[pc_idx])
        top_joints_idx = np.argsort(pc_loadings)[-5:][::-1]  # Top 5 joints
        print(f"  PC{pc_idx+1} (explains {explained_variance_ratio[pc_idx]:.3f} variance):")
        for joint_idx in top_joints_idx:
            loading = pca.components_[pc_idx][joint_idx]
            joint_name = joint_names[joint_idx] if joint_idx < len(joint_names) else f"Joint_{joint_idx}"
            print(f"    {joint_name}: {loading:.3f}")
    
    print("="*50)
    
    return {
        'original': qpos_data,
        'reconstructed': qpos_reconstructed,
        'timestamps': timestamps,
        'joint_names': joint_names,
        'pca': pca,
        'explained_variance_ratio': explained_variance_ratio,
        'n_components': n_components,
        'robot_name': robot_name,
        'hand_type': hand_type,
        'source_video': source_video
    }


def start_synergy_analysis(robot_dir: str, config_path: str, pkl_file: str, n_components: int = 5):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start synergy analysis with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    
    # Load and analyze data with PCA
    analysis_data = load_and_analyze_qpos_data(pkl_file, n_components)
    original_qpos = analysis_data['original']
    reconstructed_qpos = analysis_data['reconstructed']
    joint_names = analysis_data['joint_names']

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
    robot1.set_pose(sapien.Pose([0, 0.1, base_z]))  # Left robot - PCA reconstructed
    robot2.set_pose(sapien.Pose([0, -0.1, base_z]))   # Right robot - Original data

    # Different robot loader may have different orders for joints
    sapien_joint_names = [joint.get_name() for joint in robot1.get_active_joints()]
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)


    print(f"\nStarting playback of {len(original_qpos)} recordings...")
    print("Controls: 'q' to quit, 'r' to restart, space to pause/resume")
    print("Left robot: PCA reconstructed data")
    print("Right robot: Original data")
    
    frame_idx = 0
    paused = False
    playback_speed = 1.0  # Playback speed multiplier
    
    while True:
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            frame_idx = 0
            print("Restarting playback...")
        elif key == ord(' '):
            paused = not paused
            print("Paused" if paused else "Resumed")
        elif key == ord('+') or key == ord('='):
            playback_speed = min(playback_speed * 1.2, 5.0)
            print(f"Playback speed: {playback_speed:.2f}x")
        elif key == ord('-'):
            playback_speed = max(playback_speed / 1.2, 0.1)
            print(f"Playback speed: {playback_speed:.2f}x")
        
        if not paused:
            if frame_idx < len(original_qpos):
                # Get current frame data
                qpos_original = original_qpos[frame_idx]
                qpos_reconstructed = reconstructed_qpos[frame_idx]
                
                # Apply joint positions to robots
                robot1.set_qpos(qpos_reconstructed[retargeting_to_sapien])  # Left - PCA reconstructed
                robot2.set_qpos(qpos_original[retargeting_to_sapien])       # Right - Original
                
                # Print progress every 100 frames
                if frame_idx % 100 == 0:
                    progress = (frame_idx / len(original_qpos)) * 100
                    print(f"Progress: {frame_idx}/{len(original_qpos)} ({progress:.1f}%)")
                    
                    # Show difference between original and reconstructed
                    diff = np.abs(qpos_original - qpos_reconstructed)
                    mean_diff = np.mean(diff)
                    max_diff = np.max(diff)
                    print(f"  Reconstruction diff - Mean: {mean_diff:.4f}, Max: {max_diff:.4f}")
                
                frame_idx += 1
            else:
                # Loop back to beginning
                frame_idx = 0
                print("Playback completed. Looping...")
        
        # Render scene
        for _ in range(2):
            viewer.render()
        
        # Control playback speed
        time.sleep(0.03 / playback_speed)  # Base delay of 30ms
    
    print("Synergy analysis session completed!")
    print(f"Analyzed {len(original_qpos)} recordings with {analysis_data['n_components']} PCA components")
    print(f"Variance explained: {analysis_data['explained_variance_ratio'].sum():.3f}")


def main(
    pkl_file: str,
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    n_components: int = 5,
):
    """
    Loads saved qpos data from a pkl file, performs PCA analysis, and displays the comparison between 
    original and PCA-reconstructed hand movements.

    Args:
        pkl_file: Path to the pkl file containing saved qpos data.
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
        n_components: Number of principal components to use for PCA reconstruction (default: 5).
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
    print(f"PKL file: {pkl_file}")
    print(f"PCA components: {n_components}")
    
    start_synergy_analysis(str(robot_dir), str(config_path), pkl_file, n_components)
    
    print("Analysis completed!")


if __name__ == "__main__":
    tyro.cli(main)
