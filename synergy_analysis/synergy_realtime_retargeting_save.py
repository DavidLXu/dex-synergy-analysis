import multiprocessing
import pickle
import time
from datetime import datetime
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

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector


def start_retargeting(queue: multiprocessing.Queue, robot_dir: str, config_path: str, robot_name_param: str, hand_type_param: str):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    
    # Initialize storage for recorded qpos values
    recorded_qpos = []
    timestamps = []
    save_counter = 0
    
    def save_qpos_data(qpos_list, timestamp_list, save_num):
        """Save recorded qpos data to pickle file"""
        if not qpos_list:
            return
        
        # Create filename with timestamp and save number
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recorded_qpos_{robot_name_param}_{hand_type_param}_{timestamp_str}_save{save_num:03d}.pkl"
        
        # Save data
        data = {
            'qpos': np.array(qpos_list),
            'timestamps': np.array(timestamp_list),
            'joint_names': retargeting.joint_names,
            'config_path': config_path,
            'save_number': save_num,
            'total_recordings': len(qpos_list),
            'robot_name': robot_name_param,
            'hand_type': hand_type_param
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved {len(qpos_list)} qpos recordings for {robot_name_param} {hand_type_param} hand to {filename}")
        print(f"Auto-saved data to: {filename} ({len(qpos_list)} recordings, {robot_name_param} {hand_type_param} hand)")
        return filename

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
    robot1.set_pose(sapien.Pose([0, 0.15, base_z]))  # Left robot
    robot2.set_pose(sapien.Pose([0, -0.15, base_z]))   # Right robot

    # Different robot loader may have different orders for joints
    sapien_joint_names = [joint.get_name() for joint in robot1.get_active_joints()]
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)


    while True:
        try:
            bgr = queue.get(timeout=5)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Empty:
            logger.error(
                "Fail to fetch image from camera in 5 secs. Please check your web camera device."
            )
            return

        _, joint_pos, keypoint_2d, _ = detector.detect(rgb)
        bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
        cv2.imshow("realtime_retargeting_demo", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if joint_pos is None:
            logger.warning(f"{hand_type} hand is not detected.")
        else:
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                indices = indices
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            qpos = retargeting.retarget(ref_value)
            
            # Record qpos and timestamp
            recorded_qpos.append(qpos.copy())
            timestamps.append(time.time())
            
            # Save every 100 recordings
            if len(recorded_qpos) % 500 == 0:
                save_counter += 1
                print(f"Reached {len(recorded_qpos)} recordings - auto-saving...")
                save_qpos_data(recorded_qpos, timestamps, save_counter)
            
            # Print current qpos for debugging (every 30 frames to avoid spam)
            if len(recorded_qpos) % 20 == 0:
                print(f"Recording #{len(recorded_qpos)}: qpos = {qpos[:5]}..." + 
                      f" (showing first 5 of {len(qpos)} joints)")
            
            # Apply same joint positions to both robots
            robot1.set_qpos(qpos[retargeting_to_sapien])
            robot2.set_qpos(qpos[retargeting_to_sapien])

        for _ in range(2):
            viewer.render()
    
    # Save any remaining recorded data when exiting
    if recorded_qpos:
        # Check if there's any unsaved data (less than 100 recordings since last save)
        remaining_count = len(recorded_qpos) % 100
        if remaining_count > 0:
            save_counter += 1
            filename = save_qpos_data(recorded_qpos[-remaining_count:], 
                                    timestamps[-remaining_count:], 
                                    save_counter)
            print(f"Final save: {remaining_count} remaining recordings")
        
        print(f"Session complete!")
        print(f"Total recordings: {len(recorded_qpos)}")
        print(f"Total save files: {save_counter}")
        if timestamps:
            print(f"Recording duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
        print(f"Files saved with pattern: recorded_qpos_{robot_name_param}_{hand_type_param}_*_save*.pkl")


def retarget_from_video(video_path: str, robot_dir: str, config_path: str, robot_name_param: str, hand_type_param: str, playback_speed: float = 1.0):
    """
    Retarget hand movements from an existing mp4 video file.
    
    Args:
        video_path: Path to the mp4 video file
        robot_dir: Path to robot directory
        config_path: Path to retargeting config
        robot_name_param: Robot name for saving
        hand_type_param: Hand type for saving
        playback_speed: Speed multiplier for processing (1.0 = normal speed, 2.0 = 2x speed)
    """
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start video retargeting with config {config_path}")
    logger.info(f"Processing video: {video_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()
    
    # Initialize storage for recorded qpos values
    recorded_qpos = []
    timestamps = []
    save_counter = 0
    
    def save_qpos_data(qpos_list, timestamp_list, save_num):
        """Save recorded qpos data to pickle file"""
        if not qpos_list:
            return
        
        # Create filename with timestamp and save number
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"recorded_qpos_{robot_name_param}_{hand_type_param}_video_{timestamp_str}_save{save_num:03d}.pkl"
        
        # Save data
        data = {
            'qpos': np.array(qpos_list),
            'timestamps': np.array(timestamp_list),
            'joint_names': retargeting.joint_names,
            'config_path': config_path,
            'save_number': save_num,
            'total_recordings': len(qpos_list),
            'robot_name': robot_name_param,
            'hand_type': hand_type_param,
            'source_video': video_path,
            'playback_speed': playback_speed
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Saved {len(qpos_list)} qpos recordings for {robot_name_param} {hand_type_param} hand to {filename}")
        print(f"Auto-saved data to: {filename} ({len(qpos_list)} recordings, {robot_name_param} {hand_type_param} hand)")
        return filename

    hand_type = "Right" if "right" in config_path.lower() else "Left"
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)

    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video file: {video_path}")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps
    
    print(f"Video properties:")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Duration: {duration:.2f} seconds")
    print(f"  Playback speed: {playback_speed}x")
    print(f"  Processing time estimate: {duration/playback_speed:.2f} seconds")
    
    frame_count = 0
    start_time = time.time()
    
    # Calculate frame delay for playback speed
    frame_delay = (1.0 / fps) / playback_speed if playback_speed > 0 else 0
    
    print(f"\nStarting video processing...")
    print("Controls: 'q' to quit early, 's' to skip current video")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video")
            break
        
        frame_count += 1
        
        # Convert BGR to RGB for detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        _, joint_pos, keypoint_2d, _ = detector.detect(rgb)
        
        # Draw skeleton on frame for display
        frame_display = detector.draw_skeleton_on_image(frame, keypoint_2d, style="default")
        
        # Add progress info to frame
        progress = (frame_count / total_frames) * 100
        progress_text = f"Frame: {frame_count}/{total_frames} ({progress:.1f}%) - {len(recorded_qpos)} recordings"
        cv2.putText(frame_display, progress_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Video Retargeting", frame_display)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Processing interrupted by user")
            break
        elif key == ord('s'):
            print("Skipping video...")
            break
        
        if joint_pos is None:
            if frame_count % 100 == 0:  # Print warning every 100 frames
                logger.warning(f"{hand_type} hand not detected in frame {frame_count}")
        else:
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices
            if retargeting_type == "POSITION":
                indices = indices
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            qpos = retargeting.retarget(ref_value)
            
            # Record qpos and timestamp
            recorded_qpos.append(qpos.copy())
            timestamps.append(time.time())
            
            # Save every 500 recordings
            if len(recorded_qpos) % 1000 == 0:
                save_counter += 1
                print(f"\nReached {len(recorded_qpos)} recordings - auto-saving...")
                save_qpos_data(recorded_qpos, timestamps, save_counter)
            
            # Print progress every 100 successful detections
            if len(recorded_qpos) % 100 == 0:
                elapsed = time.time() - start_time
                eta = (elapsed / frame_count) * (total_frames - frame_count)
                print(f"Progress: {frame_count}/{total_frames} frames, {len(recorded_qpos)} recordings, ETA: {eta:.1f}s")
        
        # Control playback speed
        if frame_delay > 0:
            time.sleep(frame_delay)
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Save any remaining recorded data
    if recorded_qpos:
        # Check if there's any unsaved data (less than 500 recordings since last save)
        remaining_count = len(recorded_qpos) % 500
        if remaining_count > 0:
            save_counter += 1
            filename = save_qpos_data(recorded_qpos[-remaining_count:], 
                                    timestamps[-remaining_count:], 
                                    save_counter)
            print(f"Final save: {remaining_count} remaining recordings")
        
        total_time = time.time() - start_time
        print(f"\nVideo processing complete!")
        print(f"Processed {frame_count} frames in {total_time:.2f} seconds")
        print(f"Total recordings: {len(recorded_qpos)}")
        print(f"Total save files: {save_counter}")
        print(f"Success rate: {len(recorded_qpos)/frame_count*100:.1f}%")
        if timestamps:
            print(f"Recording duration: {timestamps[-1] - timestamps[0]:.2f} seconds")
        print(f"Files saved with pattern: recorded_qpos_{robot_name_param}_{hand_type_param}_video_*_save*.pkl")
    else:
        print("No hand poses were detected in the video.")


def produce_frame(queue: multiprocessing.Queue, camera_path: Optional[str] = None):
    if camera_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(camera_path)

    while cap.isOpened():
        success, image = cap.read()
        # time.sleep(1 / 30.0)
        if not success:
            continue
        queue.put(image)


def main_video(
    video_path: str,
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    playback_speed: float = 1.0,
):
    """
    Process an existing mp4 video file and retarget hand movements to robot poses.

    Args:
        video_path: Path to the mp4 video file to process.
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
        playback_speed: Speed multiplier for processing (1.0 = normal speed, 2.0 = 2x speed, 0.5 = half speed).
    """
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent / "assets" / "robots" / "hands"
    )

    print(f"Processing video: {video_path}")
    print(f"Robot: {robot_name.value}")
    print(f"Hand type: {hand_type.value}")
    print(f"Retargeting type: {retargeting_type.value}")
    print(f"Playback speed: {playback_speed}x")
    
    retarget_from_video(
        video_path=video_path,
        robot_dir=str(robot_dir),
        config_path=str(config_path),
        robot_name_param=robot_name.value,
        hand_type_param=hand_type.value,
        playback_speed=playback_speed
    )
    
    print("Video processing completed!")


def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    camera_path: Optional[str] = None,
):
    """
    Detects the human hand pose from a live camera feed and translates the human pose trajectory into a robot pose trajectory.
    Automatically detects if camera_path is a video file and switches to video processing mode.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
        camera_path: the device path to feed to opencv to open the web camera (default: 0), or path to video file.
            If a video file is detected (.mp4, .avi, .mov, .mkv), automatically switches to video processing mode.
    """
    # Check if camera_path is a video file
    if camera_path and Path(camera_path).suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
        print(f"Detected video file: {camera_path}")
        print("Switching to video processing mode...")
        main_video(camera_path, robot_name, retargeting_type, hand_type)
        return
    
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )

    queue = multiprocessing.Queue(maxsize=1000)
    producer_process = multiprocessing.Process(
        target=produce_frame, args=(queue, camera_path)
    )
    consumer_process = multiprocessing.Process(
        target=start_retargeting, args=(queue, str(robot_dir), str(config_path), robot_name.value, hand_type.value)
    )

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()
    time.sleep(5)

    print("done")


if __name__ == "__main__":
    import sys
    
    # Check if user wants video processing mode
    if len(sys.argv) > 1 and sys.argv[1] == "video":
        # Remove 'video' from args and use main_video
        sys.argv.pop(1)
        tyro.cli(main_video)
    else:
        # Use regular main function
        tyro.cli(main)
