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


def load_pca_parameters(pkl_file: str, n_components: int = 5):
    """Load PCA parameters from a previously saved pkl file and fit PCA"""
    print(f"Loading PCA parameters from {pkl_file}...")
    
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    qpos_data = data['qpos']
    joint_names = data['joint_names']
    robot_name = data.get('robot_name', 'unknown')
    hand_type = data.get('hand_type', 'unknown')
    source_video = data.get('source_video', None)
    
    print(f"Loaded training data: {len(qpos_data)} recordings with {qpos_data.shape[1]} joints")
    print(f"Robot: {robot_name}, Hand: {hand_type}")
    if source_video:
        print(f"Source video: {source_video}")
    print(f"Joint names: {joint_names}")
    
    # Fit PCA on training data
    print(f"Fitting PCA with {n_components} components...")
    pca = PCA(n_components=n_components)
    pca_coefficients = pca.fit_transform(qpos_data)
    
    # Calculate component ranges from training data
    component_mins = np.min(pca_coefficients, axis=0)
    component_maxs = np.max(pca_coefficients, axis=0)
    component_stds = np.std(pca_coefficients, axis=0)
    
    # Print PCA analysis
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    
    print("\n" + "="*50)
    print("PCA Component Analysis:")
    print("="*50)
    for i in range(n_components):
        print(f"  PC{i+1}:")
        print(f"    Variance explained: {explained_variance_ratio[i]:.3f} ({cumulative_variance[i]:.3f} cumulative)")
        print(f"    Range: [{component_mins[i]:.3f}, {component_maxs[i]:.3f}]")
        print(f"    Std dev: {component_stds[i]:.3f}")
    print("="*50)
    
    return {
        'pca': pca,
        'joint_names': joint_names,
        'robot_name': robot_name,
        'hand_type': hand_type,
        'component_mins': component_mins,
        'component_maxs': component_maxs,
        'component_stds': component_stds,
        'explained_variance_ratio': explained_variance_ratio,
        'training_data': qpos_data,
        'pca_coefficients': pca_coefficients
    }


def start_component_exploration(robot_dir: str, config_path: str, pkl_file: str, n_components: int = 5):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start component exploration with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    
    # Load PCA parameters
    pca_data = load_pca_parameters(pkl_file, n_components)
    pca = pca_data['pca']
    component_mins = pca_data['component_mins']
    component_maxs = pca_data['component_maxs']
    component_stds = pca_data['component_stds']
    
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
    cam.set_local_pose(sapien.Pose([0.6, 0, 0.1], [0, 0, 0, -1]))

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = False
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())

    # Load robot
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

    robot = loader.load(filepath)

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

    robot.set_pose(sapien.Pose([0, 0, base_z]))

    # Different robot loader may have different orders for joints
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)

    print(f"\nStarting component exploration...")
    print("Controls:")
    print("  '1'-'9': Select component to explore (1=PC1, 2=PC2, etc.)")
    print("  'a': Auto-sweep current component")
    print("  's': Stop auto-sweep")
    print("  'r': Reset all components to zero")
    print("  '+'/'-': Manual adjustment of current component")
    print("  'q': Quit")
    print("  'space': Toggle through components automatically")
    
    # State variables
    current_component = 0  # Currently selected component (0-indexed)
    auto_sweep = False
    sweep_direction = 1
    sweep_speed = 0.02
    component_values = np.zeros(n_components)
    auto_cycle = False
    cycle_timer = 0
    cycle_duration = 3.0  # seconds per component
    
    # Initialize with neutral pose
    neutral_qpos = pca.inverse_transform(np.zeros((1, n_components))).flatten()
    robot.set_qpos(neutral_qpos[retargeting_to_sapien])
    
    # Create a simple info display window
    info_window = np.zeros((400, 600, 3), dtype=np.uint8)
    
    start_time = time.time()
    
    while True:
        current_time = time.time()
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif ord('1') <= key <= ord('9'):
            new_component = key - ord('1')
            if new_component < n_components:
                current_component = new_component
                auto_sweep = False
                auto_cycle = False
                print(f"Selected component {current_component + 1} (PC{current_component + 1})")
        elif key == ord('a'):
            auto_sweep = True
            auto_cycle = False
            print(f"Auto-sweeping component {current_component + 1}")
        elif key == ord('s'):
            auto_sweep = False
            auto_cycle = False
            print("Stopped auto-sweep")
        elif key == ord('r'):
            component_values.fill(0)
            auto_sweep = False
            auto_cycle = False
            print("Reset all components to zero")
        elif key == ord('+') or key == ord('='):
            component_values[current_component] = min(
                component_values[current_component] + 0.1,
                component_maxs[current_component]
            )
            auto_sweep = False
            auto_cycle = False
        elif key == ord('-'):
            component_values[current_component] = max(
                component_values[current_component] - 0.1,
                component_mins[current_component]
            )
            auto_sweep = False
            auto_cycle = False
        elif key == ord(' '):
            auto_cycle = not auto_cycle
            auto_sweep = False
            cycle_timer = current_time
            if auto_cycle:
                component_values.fill(0)
                current_component = 0
                print("Started auto-cycling through components")
            else:
                print("Stopped auto-cycling")
        
        # Auto-sweep logic
        if auto_sweep:
            # Sweep current component
            current_val = component_values[current_component]
            min_val = component_mins[current_component]
            max_val = component_maxs[current_component]
            
            current_val += sweep_direction * sweep_speed * (max_val - min_val)
            
            if current_val >= max_val:
                current_val = max_val
                sweep_direction = -1
            elif current_val <= min_val:
                current_val = min_val
                sweep_direction = 1
            
            component_values[current_component] = current_val
        
        # Auto-cycle logic
        if auto_cycle:
            elapsed = current_time - cycle_timer
            if elapsed >= cycle_duration:
                # Move to next component
                current_component = (current_component + 1) % n_components
                component_values.fill(0)
                cycle_timer = current_time
                print(f"Auto-cycling to component {current_component + 1} (PC{current_component + 1})")
            else:
                # Sweep current component in cycle
                progress = elapsed / cycle_duration
                min_val = component_mins[current_component]
                max_val = component_maxs[current_component]
                
                # Create a smooth sine wave sweep
                sweep_progress = np.sin(progress * 4 * np.pi)  # 2 full cycles
                component_values[current_component] = min_val + (max_val - min_val) * (sweep_progress + 1) / 2
        
        # Reconstruct pose using current component values
        qpos_reconstructed = pca.inverse_transform(component_values.reshape(1, -1)).flatten()
        robot.set_qpos(qpos_reconstructed[retargeting_to_sapien])
        
        # Update info display
        info_window.fill(0)
        
        # Title
        cv2.putText(info_window, "PCA Component Explorer", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Current component info
        y_offset = 70
        cv2.putText(info_window, f"Current Component: PC{current_component + 1}", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        
        variance_pct = pca_data['explained_variance_ratio'][current_component] * 100
        cv2.putText(info_window, f"Explains: {variance_pct:.1f}% variance", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 35
        
        # Component values
        for i in range(n_components):
            color = (0, 255, 0) if i == current_component else (150, 150, 150)
            val_text = f"PC{i+1}: {component_values[i]:+6.3f}"
            min_val = component_mins[i]
            max_val = component_maxs[i]
            range_text = f"[{min_val:.2f}, {max_val:.2f}]"
            
            cv2.putText(info_window, val_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(info_window, range_text, (200, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
            y_offset += 20
        
        # Status
        y_offset += 20
        if auto_sweep:
            cv2.putText(info_window, "Status: Auto-sweeping", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        elif auto_cycle:
            elapsed = current_time - cycle_timer
            progress_pct = (elapsed / cycle_duration) * 100
            cv2.putText(info_window, f"Status: Auto-cycling ({progress_pct:.0f}%)", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        else:
            cv2.putText(info_window, "Status: Manual control", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls reminder
        y_offset += 40
        controls = [
            "Controls:",
            "1-9: Select component",
            "a: Auto-sweep", "s: Stop sweep",
            "r: Reset all", "+/-: Adjust",
            "space: Auto-cycle", "q: Quit"
        ]
        
        for i, control in enumerate(controls):
            color = (255, 200, 100) if i == 0 else (180, 180, 180)
            font_scale = 0.5 if i == 0 else 0.4
            cv2.putText(info_window, control, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
            y_offset += 18
        
        cv2.imshow("Component Explorer Controls", info_window)
        
        # Render scene
        for _ in range(2):
            viewer.render()
        
        time.sleep(0.02)  # 50 FPS
    
    cv2.destroyAllWindows()
    print("Component exploration completed!")


def main(
    pkl_file: str,
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    n_components: int = 5,
):
    """
    Explore individual PCA components by sliding each from min to max while keeping others at zero.
    This helps understand what specific movement patterns each component controls.

    Args:
        pkl_file: Path to the pkl file containing PCA training data.
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
        n_components: Number of principal components to explore (default: 5).
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
    print(f"Exploring {n_components} components")
    
    start_component_exploration(str(robot_dir), str(config_path), pkl_file, n_components)
    
    print("Component exploration completed!")


if __name__ == "__main__":
    tyro.cli(main) 