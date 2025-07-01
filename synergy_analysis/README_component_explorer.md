# PCA Component Explorer

This tool allows you to explore individual Principal Components (PCs) by sliding each component from its minimum to maximum value while keeping all others at zero. This visualization helps understand what specific movement patterns each principal component controls.

## Purpose

Understanding hand synergies is crucial for:
- **Research**: Identifying natural coordination patterns in human hand movements
- **Robot Control**: Designing intuitive control interfaces with fewer degrees of freedom
- **Biomechanics**: Studying how humans coordinate complex finger movements
- **Prosthetics**: Developing natural control schemes for artificial hands

## Usage

```bash
python synergy_analysis/synergy_component_explorer.py --pkl-file <pkl_file> --robot-name <robot_name> --retargeting-type <retargeting_type> --hand-type <hand_type> --n-components <num>
```

### Parameters

- `--pkl-file`: Path to the pkl file containing PCA training data
- `--robot-name`: Robot type (e.g., `allegro`, `shadow`, `leap`)
- `--retargeting-type`: Retargeting algorithm (e.g., `vector`, `position`, `dexpilot`)
- `--hand-type`: `left` or `right`
- `--n-components`: Number of PCA components to explore (default: 5)

### Example

```bash
python synergy_analysis/synergy_component_explorer.py --pkl-file recorded_qpos_allegro_hand_right_20241220_143022_save001.pkl --robot-name allegro --retargeting-type vector --hand-type right --n-components 5
```

## What You'll See

### 3D Visualization
- **Robot hand** that moves according to the current component values
- **Real-time updates** as you adjust component values
- **Smooth animations** during auto-sweep and auto-cycle modes

### Control Window
- **Current component information**: Which PC is selected and its variance contribution
- **Component values**: Real-time display of all PC values and their ranges
- **Status indicator**: Shows current mode (manual, auto-sweep, auto-cycle)
- **Control reminders**: Quick reference for all available controls

## Interactive Controls

### Component Selection
- **'1'-'9'**: Select which component to explore (1=PC1, 2=PC2, etc.)
- **Current component** is highlighted in green in the control window

### Automatic Modes
- **'a'**: Auto-sweep current component from min to max and back
- **'s'**: Stop auto-sweep
- **'space'**: Auto-cycle through all components (3 seconds each with smooth sweeping)

### Manual Control
- **'+'/'='**: Increase current component value
- **'-'**: Decrease current component value
- **'r'**: Reset all components to zero (neutral pose)

### General
- **'q'**: Quit the application

## Understanding the Results

### Component Analysis
Each component shows:
- **Variance explained**: How much of the total movement variation this component captures
- **Value range**: Min/max values observed in the training data
- **Current value**: Real-time component coefficient

### Movement Patterns
As you explore each component, you'll observe:

#### PC1 (Usually ~40-60% variance)
- **Primary movement pattern**: Often represents the most common hand coordination
- **Examples**: Opening/closing the whole hand, or thumb opposition

#### PC2 (Usually ~15-25% variance)
- **Secondary pattern**: Often independent finger movements
- **Examples**: Index finger extension, or finger spreading

#### PC3+ (Decreasing variance)
- **Fine details**: More specific finger coordination patterns
- **Examples**: Individual finger curling, precise pinch movements

### What to Look For

1. **Dominant Patterns**: PC1 typically shows the most natural, common movements
2. **Independent Motions**: Higher PCs often show more isolated finger movements
3. **Coupling Effects**: Notice which fingers move together in each component
4. **Natural Transitions**: Smooth component changes should produce natural-looking motions

## Applications

### Research Applications
- **Synergy Identification**: Discover which fingers naturally coordinate together
- **Dimensionality Analysis**: Determine how many components capture most natural movements
- **Cross-Subject Studies**: Compare component patterns across different users

### Engineering Applications
- **Control Interface Design**: Use dominant components for primary control axes
- **Motion Planning**: Generate natural hand motions by varying component coefficients
- **Bandwidth Optimization**: Transmit only significant component values for teleoperation

### Educational Use
- **Biomechanics Teaching**: Visualize human motor control principles
- **Machine Learning**: Demonstrate PCA and dimensionality reduction concepts
- **Robotics Education**: Show human-robot interface design principles

## Workflow Integration

### Typical Research Workflow
1. **Record Training Data**: Use `synergy_realtime_retargeting_save.py`
2. **Analyze Components**: Use this explorer to understand each PC
3. **Validate Quality**: Use `synergy_reconstruct.py` for quantitative analysis
4. **Apply in Real-time**: Use `synergy_realtime_reconstruct.py` for live applications

### Component Selection Strategy
1. **Start with PC1**: Understand the primary movement pattern
2. **Add PC2**: See what secondary coordination is important
3. **Evaluate PC3-5**: Determine if additional detail is needed
4. **Balance trade-offs**: More components = better quality but higher complexity

## Tips for Best Results

### Data Quality
- **Diverse movements**: Ensure training data covers various hand motions
- **Sufficient samples**: At least 500-1000 recordings for stable PCA
- **Natural motions**: Record realistic, coordinated hand movements

### Exploration Strategy
- **Start with auto-cycle**: Get overview of all components quickly
- **Focus on high-variance PCs**: Spend more time understanding PC1-3
- **Note movement quality**: Natural-looking motions indicate good components
- **Test edge cases**: Check component extremes for unrealistic poses

### Analysis
- **Document patterns**: Record what each component seems to control
- **Compare with literature**: Relate findings to known hand synergies
- **Test applications**: Use insights to design better control interfaces

## Troubleshooting

### Common Issues
- **Jerky movements**: Training data may need smoothing or more samples
- **Unnatural poses**: Component ranges may be too extreme
- **Poor component separation**: May need more diverse training data
- **Low variance**: First few components should explain >80% total variance

### Performance Issues
- **Slow rendering**: Reduce rendering frequency or use simpler robot models
- **Memory usage**: Large training datasets may require chunked processing
- **Control responsiveness**: Adjust sweep speed for better interaction

This tool provides invaluable insights into hand movement coordination and is essential for developing effective synergy-based control systems. 