# Video Retargeting

This document explains how to use the video retargeting functionality in `synergy_realtime_retargeting_save.py` to process existing mp4 videos instead of live camera feeds.

## Usage

There are three ways to process video files:

### 1. Automatic Detection (Recommended)

The script automatically detects video files when you provide a video path:

```bash
python synergy_analysis/synergy_realtime_retargeting_save.py --robot-name allegro --retargeting-type vector --hand-type right --camera-path /path/to/your/video.mp4
```

### 2. Explicit Video Mode

Use the `video` command explicitly:

```bash
python synergy_analysis/synergy_realtime_retargeting_save.py --video /path/to/your/video.mp4 --robot-name allegro --retargeting-type vector --hand-type right
```

### 3. With Playback Speed Control

Control processing speed (useful for fast processing or detailed analysis):

```bash
python synergy_analysis/synergy_realtime_retargeting_save.py --video /path/to/your/video.mp4 --robot-name allegro --retargeting-type vector --hand-type right --playback-speed 2.0
```

## Parameters

- `--video`: Path to the mp4 video file
- `--robot-name`: Robot type (e.g., `allegro`, `shadow`, `leap`)
- `--retargeting-type`: Retargeting algorithm (e.g., `vector`, `position`, `dexpilot`)
- `--hand-type`: `left` or `right`
- `--playback-speed`: Speed multiplier (default: 1.0)
  - `2.0` = 2x speed (faster processing)
  - `0.5` = half speed (more detailed processing)
  - `0.0` = maximum speed (no delay)

## Supported Video Formats

- `.mp4` (recommended)
- `.avi`
- `.mov`
- `.mkv`

## Features

### Progress Tracking
- Real-time progress display showing frame count and percentage
- ETA estimation for completion
- Success rate tracking (percentage of frames with successful hand detection)

### Interactive Controls
During video processing:
- **'q'**: Quit processing early
- **'s'**: Skip current video
- **ESC**: Close video window

### Enhanced Output Files
Video-processed files include additional metadata:
- `source_video`: Original video file path
- `playback_speed`: Processing speed used
- Files named with `_video_` prefix: `recorded_qpos_{robot}_{hand}_video_{timestamp}_save{num}.pkl`

### Auto-saving
- Saves every 500 successful hand detections
- Final save for remaining data when processing completes
- Robust error handling and progress reporting

## Example Workflow

1. **Record or obtain a video** with clear hand movements
2. **Process the video**:
   ```bash
   python synergy_analysis/synergy_realtime_retargeting_save.py --robot-name allegro --retargeting-type vector --hand-type right --camera-path hand_demo.mp4
   ```
3. **Review the output**:
   - Video window shows detection progress with skeleton overlay
   - Console shows frame progress, detection success rate, and file saves
   - PKL files saved automatically every 500 detections

4. **Analyze results** using the synergy reconstruction script:
   ```bash
   python synergy_analysis/synergy_reconstruct.py --pkl-file recorded_qpos_allegro_hand_right_video_20241220_143022_save001.pkl --robot-name allegro --retargeting-type vector --hand-type right
   ```

## Tips for Best Results

- **Good lighting**: Ensure hands are well-lit in the video
- **Clear background**: Avoid cluttered backgrounds that might confuse detection
- **Appropriate distance**: Hands should be clearly visible but not too close/far
- **Steady camera**: Minimize camera shake for better detection
- **Single hand**: Focus on one hand at a time for better detection accuracy

## Troubleshooting

- **"No hand poses detected"**: Check video quality, lighting, and hand visibility
- **Low success rate**: Try adjusting video quality or recording conditions
- **Processing too slow**: Increase `playback_speed` parameter
- **Memory issues**: Large videos are processed in chunks automatically 