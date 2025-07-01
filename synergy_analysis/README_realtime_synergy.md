# Realtime Synergy Reconstruction

This script performs real-time synergy-based hand reconstruction using PCA parameters learned from previously recorded data. It shows a side-by-side comparison of original retargeted poses vs PCA-reconstructed poses in real-time.

## Overview

The system:
1. **Loads PCA parameters** from a previously saved pkl file (training data)
2. **Detects hand poses** from live camera feed in real-time
3. **Applies retargeting** to convert human poses to robot poses
4. **Performs PCA decomposition** and **reconstruction** on-the-fly
5. **Displays both hands** side-by-side: original vs reconstructed

## Usage

```bash
python synergy_analysis/synergy_realtime_reconstruct.py --pkl-file <pkl_file> --robot-name <robot_name> --retargeting-type <retargeting_type> --hand-type <hand_type> --n-components <num> --camera-path <path>
```

### Parameters

- `--pkl-file`: Path to the pkl file containing training data (from recording session)
- `--robot-name`: Robot type (e.g., `allegro`, `shadow`, `leap`)
- `--retargeting-type`: Retargeting algorithm (e.g., `vector`, `position`, `dexpilot`)
- `--hand-type`: `left` or `right`
- `--n-components`: Number of PCA components to use (default: 5)
- `--camera-path`: Camera device or video file path (default: 0)

### Example

```bash
python synergy_analysis/synergy_realtime_reconstruct.py --pkl-file recorded_qpos_allegro_hand_right_20241220_143022_save001.pkl --robot-name allegro --retargeting-type vector --hand-type right --n-components 3
```

## Workflow

### Step 1: Record Training Data
First, collect training data using the recording script:
```bash
python synergy_analysis/synergy_realtime_retargeting_save.py --robot-name allegro --retargeting-type vector --hand-type right
```
This creates pkl files like: `recorded_qpos_allegro_hand_right_20241220_143022_save001.pkl`

### Step 2: Analyze Training Data (Optional)
Review the training data quality:
```bash
python synergy_analysis/synergy_reconstruct.py --pkl-file recorded_qpos_allegro_hand_right_20241220_143022_save001.pkl --robot-name allegro --retargeting-type vector --hand-type right --n-components 5
```

### Step 3: Real-time Reconstruction
Use the training data for real-time synergy reconstruction:
```bash
python synergy_analysis/synergy_realtime_reconstruct.py --pkl-file recorded_qpos_allegro_hand_right_20241220_143022_save001.pkl --robot-name allegro --retargeting-type vector --hand-type right --n-components 3
```

## What You'll See

### Real-time Display
- **Camera window**: Shows hand detection with skeleton overlay and statistics
  - Frame count and detection success rate
  - Real-time reconstruction error
  - PCA component information

- **3D Viewer**: Shows two robot hands side-by-side
  - **Left robot**: PCA-reconstructed movements (using N principal components)
  - **Right robot**: Original retargeted movements (full dimensionality)

### Console Output
- PCA parameter loading and analysis
- Variance explained by each principal component
- Real-time statistics every 100 detections
- Final session summary with error statistics

## Understanding the Results

### PCA Components
- **Fewer components** = more compression, less detail, potentially smoother motion
- **More components** = better reconstruction quality, more detail retained
- **Optimal number** depends on the complexity of recorded motions

### Reconstruction Error
- **Low error** (< 0.01): Excellent reconstruction, most motion patterns captured
- **Medium error** (0.01-0.05): Good reconstruction, minor details lost
- **High error** (> 0.05): Poor reconstruction, may need more components or better training data

### Visual Comparison
Compare the left (reconstructed) vs right (original) hands to evaluate:
- **Motion smoothness**: PCA often produces smoother motions
- **Detail preservation**: How well fine finger movements are captured
- **Latency**: Real-time performance vs reconstruction quality trade-off

## Controls

During real-time operation:
- **'q'**: Quit the application
- **ESC**: Close camera window

## Applications

### Research & Development
- **Analyze hand synergies**: Understand which movements are most important
- **Evaluate dimensionality reduction**: Test different numbers of components
- **Real-time control**: Use reduced-dimension control for robot hands

### Optimization
- **Bandwidth reduction**: Transmit only principal component coefficients
- **Computational efficiency**: Control robots with fewer degrees of freedom
- **Motion planning**: Generate natural hand motions in reduced space

## Tips for Best Results

### Training Data Quality
- **Diverse motions**: Record varied hand movements for comprehensive coverage
- **Sufficient data**: At least 500-1000 successful detections recommended
- **Consistent conditions**: Similar lighting and camera setup

### PCA Component Selection
- **Start with 3-5 components**: Good balance between compression and quality
- **Experiment**: Try different numbers to find optimal trade-off
- **Monitor error**: Use reconstruction error to guide component selection

### Real-time Performance
- **Good lighting**: Ensures consistent hand detection
- **Stable camera**: Reduces detection noise
- **Clear background**: Improves detection reliability

## Troubleshooting

- **High reconstruction error**: Try more PCA components or collect more training data
- **Jerky reconstructed motion**: Consider smoothing or more training data
- **Poor detection rate**: Check lighting and camera setup
- **Mismatched robot/hand type**: Ensure training data matches current setup 