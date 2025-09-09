# Standalone Head Tracking Module

A comprehensive Python module for real-time head tracking and video stabilization using computer vision techniques.

## Features

- **Dual Face Detection**: Combines Haar cascades and MTCNN for robust face detection
- **Real-time Processing**: Efficient video processing with configurable detection intervals
- **Smooth Tracking**: Applies Savitzky-Golay filtering for smooth head position tracking
- **Video Stabilization**: Automatically crops and stabilizes video based on head position
- **Zoom Effects**: Built-in zoom-in effect functionality
- **Audio Preservation**: Maintains original audio in processed videos

## Installation

1. Install the required dependencies:
```bash
pip install -r head_tracking_requirements.txt
```

2. The Haar cascade file will be automatically downloaded on first use, or you can download it manually from:
   https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml

## Quick Start

### Basic Usage

```python
from standalone_head_tracking import HeadTracker

# Initialize the tracker
tracker = HeadTracker()

# Process a video with head tracking
result_video = tracker.process_video_realtime("input.mp4", "output.mp4")
```

### Advanced Usage

```python
from standalone_head_tracking import HeadTracker

# Initialize with custom parameters
tracker = HeadTracker()

# Process with custom settings
result_video = tracker.process_video_realtime(
    video_path="input.mp4",
    output_path="output.mp4",
    crop_ratio=9/16,           # Output aspect ratio
    smoothing_window=25,       # Smoothing window size
    detection_interval=15      # Process every Nth frame
)

# Apply zoom effect
zoomed_video = tracker.zoom_in_effect(result_video)
```

### Standalone Functions

For backward compatibility, you can also use the standalone functions:

```python
from standalone_head_tracking import process_video_realtime, zoom_in_effect

# Process video
result = process_video_realtime("input.mp4", "output.mp4")

# Apply zoom effect
zoomed = zoom_in_effect(result)
```

## API Reference

### HeadTracker Class

#### `__init__(haar_cascade_path=None)`
Initialize the HeadTracker with face detection models.

**Parameters:**
- `haar_cascade_path` (str, optional): Path to Haar cascade XML file. If None, will download automatically.

#### `process_video_realtime(video_path, output_path, crop_ratio=9/16, smoothing_window=25, detection_interval=15)`
Process video with real-time head tracking and stabilization.

**Parameters:**
- `video_path` (str): Path to input video file
- `output_path` (str): Path to output video file
- `crop_ratio` (float): Aspect ratio for output video (width/height)
- `smoothing_window` (int): Window size for smoothing face positions
- `detection_interval` (int): Process every Nth frame for face detection

**Returns:**
- `moviepy.VideoFileClip`: Processed video clip

#### `zoom_in_effect(clip)`
Apply a zoom-in effect to a video clip.

**Parameters:**
- `clip`: MoviePy video clip

**Returns:**
- `moviepy.VideoFileClip`: Clip with zoom effect applied

### Standalone Functions

#### `haar_cascades_detect(frame, cascade_path=None)`
Detect faces using Haar cascades.

**Parameters:**
- `frame`: Input video frame (numpy array)
- `cascade_path` (str, optional): Path to Haar cascade XML file

**Returns:**
- `list`: [use_cascades, faces, confidence]

#### `mtcnn_detect(frame)`
Detect faces using MTCNN.

**Parameters:**
- `frame`: Input video frame (numpy array)

**Returns:**
- `list`: [use_mtcnn, faces, confidence]

## How It Works

1. **Face Detection**: The module uses two complementary face detection methods:
   - **Haar Cascades**: Fast, lightweight detection suitable for most scenarios
   - **MTCNN**: More accurate detection for challenging lighting/angle conditions

2. **Tracking Strategy**: 
   - Face detection is performed on every Nth frame (configurable)
   - Intermediate frames inherit the position from the last detected frame
   - This balances accuracy with processing speed

3. **Smoothing**: 
   - Savitzky-Golay filtering is applied to smooth out jittery head movements
   - Configurable window size for different smoothing levels

4. **Video Processing**:
   - Video is cropped around the detected head position
   - Maintains specified aspect ratio
   - Preserves original audio

## Configuration Options

### Detection Parameters
- `detection_interval`: How often to run face detection (default: every 15 frames)
- `smoothing_window`: Size of smoothing filter window (default: 25)

### Video Parameters
- `crop_ratio`: Output video aspect ratio (default: 9/16 for vertical videos)
- `min_face_size`: Minimum face size for detection (default: 50x50 pixels)

### Quality Settings
- Haar cascade confidence threshold: 4.0
- MTCNN confidence threshold: 0.999

## Performance Tips

1. **Detection Interval**: Increase `detection_interval` for faster processing on stable videos
2. **Smoothing Window**: Larger windows provide smoother tracking but may lag behind quick movements
3. **Video Resolution**: Lower resolution videos process faster
4. **Face Size**: Ensure faces are large enough in the video for reliable detection

## Troubleshooting

### Common Issues

1. **No faces detected**: 
   - Ensure good lighting and clear face visibility
   - Try adjusting the minimum face size
   - Check that the face is facing the camera

2. **Jittery tracking**:
   - Increase the smoothing window size
   - Decrease the detection interval for more frequent updates

3. **Slow processing**:
   - Increase the detection interval
   - Reduce video resolution
   - Use a smaller smoothing window

### Dependencies Issues

If you encounter issues with MTCNN or TensorFlow:
```bash
pip install tensorflow==2.13.0
pip install mtcnn==0.1.1
```

## License

This module is open source. Please check the original project license for usage terms.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Acknowledgments

- OpenCV for Haar cascade face detection
- MTCNN implementation for deep learning-based face detection
- MoviePy for video processing capabilities
