# Complete Synergy Reconstruction Workflow

This document provides a comprehensive overview of the synergy reconstruction system, explaining how all the scripts work together to enable dimensionality reduction and analysis of hand movements.

## System Overview

The synergy reconstruction system consists of five main components:

1. **Data Collection** (`synergy_analysis/synergy_realtime_retargeting_save.py`)
2. **Offline Analysis** (`synergy_analysis/synergy_reconstruct.py`) 
3. **Component Exploration** (`synergy_analysis/synergy_component_explorer.py`)
4. **Real-time Reconstruction** (`synergy_analysis/synergy_realtime_reconstruct.py`)
5. **Video Processing** (integrated in data collection script)

## Complete Workflow

### Phase 1: Data Collection

#### Live Camera Recording
```bash
python synergy_analysis/synergy_realtime_retargeting_save.py --robot-name allegro --retargeting-type vector --hand-type right
```
- Records hand movements from live camera
- Saves qpos data every 500 detections
- Creates files: `recorded_qpos_allegro_hand_right_YYYYMMDD_HHMMSS_save001.pkl`

#### Video Processing
```bash
python synergy_analysis/synergy_realtime_retargeting_save.py --robot-name allegro --retargeting-type vector --hand-type right --camera-path hand_demo.mp4
# OR
python synergy_analysis/synergy_realtime_retargeting_save.py --video hand_demo.mp4 --robot-name allegro --retargeting-type vector --hand-type right --playback-speed 2.0
```
- Processes pre-recorded videos
- Batch processing for large datasets
- Creates files: `recorded_qpos_allegro_hand_right_video_YYYYMMDD_HHMMSS_save001.pkl`

### Phase 2: Offline Analysis

#### PCA Analysis and Validation
```bash
python synergy_analysis/synergy_reconstruct.py --pkl-file recorded_qpos_allegro_hand_right_20241220_143022_save001.pkl --robot-name allegro --retargeting-type vector --hand-type right --n-components 5
```
- Performs PCA on recorded data
- Shows variance explained by each component
- Visual comparison: original vs reconstructed movements
- Helps determine optimal number of components

#### Component Exploration
```bash
python synergy_analysis/synergy_component_explorer.py --pkl-file recorded_qpos_allegro_hand_right_20241220_143022_save001.pkl --robot-name allegro --retargeting-type vector --hand-type right --n-components 5
```
- Interactive exploration of individual principal components
- Slide each component from min to max while others stay at zero
- Understand what specific movement patterns each component controls
- Identify natural hand coordination patterns

### Phase 3: Real-time Application

#### Live Synergy Reconstruction
```bash
python synergy_analysis/synergy_realtime_reconstruct.py --pkl-file recorded_qpos_allegro_hand_right_20241220_143022_save001.pkl --robot-name allegro --retargeting-type vector --hand-type right --n-components 3
```
- Uses PCA parameters from training data
- Real-time decomposition and reconstruction
- Side-by-side comparison of original vs compressed movements
- Live performance metrics

## File Types and Formats

### PKL Files Structure
All saved files contain:
```python
{
    'qpos': np.array,              # Joint positions [N_frames x N_joints]
    'timestamps': np.array,        # Timestamps [N_frames]
    'joint_names': list,           # Joint name mapping
    'config_path': str,            # Retargeting configuration
    'robot_name': str,             # Robot identifier
    'hand_type': str,              # Left/Right hand
    'save_number': int,            # File sequence number
    'total_recordings': int,       # Number of recordings
    # Video files also include:
    'source_video': str,           # (Video only) Source video path
    'playback_speed': float        # (Video only) Processing speed
}
```

### File Naming Convention
- **Live recording**: `recorded_qpos_{robot}_{hand}_{timestamp}_save{num}.pkl`
- **Video processing**: `recorded_qpos_{robot}_{hand}_video_{timestamp}_save{num}.pkl`

## Key Features

### Data Collection Features
- ✅ Auto-save every 500 detections (configurable)
- ✅ Real-time progress tracking
- ✅ Video file auto-detection
- ✅ Playback speed control for videos
- ✅ Enhanced metadata storage
- ✅ Robust error handling

### Analysis Features
- ✅ Principal Component Analysis with detailed reporting
- ✅ Top contributing joints identification
- ✅ Reconstruction error metrics (MSE/RMSE)
- ✅ Interactive playback controls
- ✅ Side-by-side visual comparison
- ✅ Individual component exploration and visualization
- ✅ Auto-sweep and auto-cycle modes for component analysis

### Real-time Features
- ✅ Live PCA decomposition/reconstruction
- ✅ Real-time performance metrics
- ✅ Dynamic error monitoring
- ✅ Side-by-side live comparison
- ✅ Overlay statistics display

## Use Cases

### Research Applications
- **Hand Synergy Analysis**: Identify natural hand movement patterns
- **Dimensionality Reduction**: Reduce control complexity from 20+ to 3-5 DOF
- **Motor Learning**: Study how humans coordinate finger movements
- **Prosthetic Control**: Develop intuitive control interfaces

### Engineering Applications
- **Robot Hand Control**: Simplify control algorithms
- **Teleoperation**: Reduce bandwidth requirements
- **Motion Planning**: Generate natural hand trajectories
- **Real-time Systems**: Enable responsive hand control

### Educational Applications
- **Machine Learning**: Demonstrate PCA on real-world data
- **Robotics**: Show human-robot interface concepts
- **Biomechanics**: Visualize hand movement coordination

## Performance Optimization

### Data Collection Tips
- **Lighting**: Ensure consistent, good lighting
- **Background**: Use clean, uncluttered backgrounds
- **Motion Variety**: Include diverse hand movements
- **Session Length**: Aim for 1000+ successful detections

### Analysis Optimization
- **Component Selection**: Start with 3-5 components
- **Validation**: Use reconstruction error to guide choices
- **Cross-validation**: Test on different motion types

### Real-time Performance
- **Hardware**: Use decent CPU/GPU for smooth operation
- **Camera**: Stable, high-quality camera feed
- **Environment**: Consistent lighting conditions

## Integration with Other Systems

### ROS Integration
The pkl files can be easily integrated with ROS:
```python
import rospy
from sensor_msgs.msg import JointState

# Load and publish joint states
data = pickle.load(open('recorded_qpos_...pkl', 'rb'))
joint_names = data['joint_names']
qpos_data = data['qpos']
```

### Machine Learning Pipelines
```python
# Use as training data for neural networks
X = data['qpos']  # Input: full joint space
y = pca.transform(X)  # Target: principal components
```

### Control Systems
```python
# Real-time control with reduced DOF
pca_coefficients = get_user_input()  # 3-5 values
full_qpos = pca.inverse_transform(pca_coefficients)  # Expand to full space
send_to_robot(full_qpos)
```

## Troubleshooting Guide

### Common Issues
1. **No hand detection**: Check lighting and camera setup
2. **High reconstruction error**: Increase n_components or collect more data
3. **Jerky motions**: Use motion smoothing or more training data
4. **File compatibility**: Ensure robot/hand type matches between recording and analysis

### Performance Issues
1. **Slow detection**: Reduce camera resolution or use faster hardware
2. **Memory issues**: Process data in smaller chunks
3. **Storage space**: Clean up old pkl files regularly

## Future Extensions

### Planned Features
- [ ] Multi-hand synergy analysis
- [ ] Temporal pattern recognition
- [ ] Adaptive PCA updating
- [ ] Neural network integration
- [ ] Haptic feedback integration

### Research Directions
- [ ] Cross-subject synergy transfer
- [ ] Task-specific synergy learning
- [ ] Reinforcement learning integration
- [ ] Real-time adaptation algorithms

This comprehensive system provides a complete pipeline from data collection to real-time application, enabling advanced research and development in hand synergy analysis and robot control. 