# Head Tracking CLI Tool

A simple command-line tool for automatic head tracking and video stabilization.

## Features

- **Dual Face Detection**: Combines Haar cascades and MTCNN for robust face detection
- **Automatic Processing**: Simple one-command video processing
- **Smooth Tracking**: Applies Savitzky-Golay filtering for smooth head position tracking
- **Video Stabilization**: Automatically crops and stabilizes video based on head position
- **Audio Preservation**: Maintains original audio in processed videos
- **Social Media Ready**: Outputs in 1080x1920 vertical format

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Download the Haar cascade file:
```bash
wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
```

## Usage

### Basic Usage

```bash
python head_tracking.py input_video.mp4
```

This will create `input_video_tracked_final.mp4` with head tracking applied.

### What it does

1. Detects faces in the video using AI models
2. Tracks head movement and smooths the motion
3. Crops the video to focus on the face (9:16 vertical format)
4. Resizes to 1080x1920 for social media compatibility
5. Preserves the original audio

## Requirements

- Python 3.7+
- OpenCV
- MoviePy
- MTCNN
- NumPy
- SciPy
- tqdm
- PIL

## How It Works

1. **Face Detection**: Uses two AI models for robust detection:
   - Haar Cascades for fast detection
   - MTCNN for challenging lighting/angles

2. **Tracking**: 
   - Detects faces every 15 frames
   - Smooths motion with Savitzky-Golay filter
   - Fills in missing frames with previous position

3. **Video Processing**:
   - Crops video around detected head position
   - Converts to 9:16 vertical format
   - Resizes to 1080x1920 for social media
   - Preserves original audio

## Troubleshooting

**No faces detected**: Ensure good lighting and face is clearly visible

**Slow processing**: This is normal for longer videos - the tool processes every 15th frame for efficiency

**File not found**: Make sure the Haar cascade file is in the same directory as the script

## License

Open source - feel free to use and modify!
