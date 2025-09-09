"""
Standalone Head Tracking Module

This module provides real-time head tracking functionality for video processing:
- Face detection using Haar cascades and MTCNN
- Head position tracking and smoothing
- Video cropping and stabilization based on head position
- Zoom effects and video resizing

Dependencies:
- opencv-python
- moviepy
- mtcnn
- scipy
- numpy
- pillow
- tqdm

Usage:
    from standalone_head_tracking import HeadTracker
    
    tracker = HeadTracker()
    result_video = tracker.process_video_realtime("input.mp4", "output.mp4")
"""

import cv2
import moviepy.editor as mp
import moviepy.video as mpv
from tqdm import tqdm
import math
from PIL import Image
import numpy
from moviepy.video.fx.all import resize
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
from mtcnn.mtcnn import MTCNN
import sys
import os
import urllib.request
import tempfile


class PrintSuppressor:
    """Context manager to suppress print statements during execution."""
    
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def block_printing(func):
    """Decorator to suppress print statements during function execution."""
    def func_wrapper(*args, **kwargs):
        with PrintSuppressor():
            return func(*args, **kwargs)
    return func_wrapper


class HeadTracker:
    """
    A comprehensive head tracking class that provides face detection and video stabilization.
    
    This class combines Haar cascades and MTCNN for robust face detection and provides
    video processing capabilities including cropping, stabilization, and zoom effects.
    """
    
    def __init__(self, haar_cascade_path=None):
        """
        Initialize the HeadTracker with face detection models.
        
        Args:
            haar_cascade_path (str, optional): Path to Haar cascade XML file.
                                             If None, will attempt to download it.
        """
        self.haar_cascade_path = haar_cascade_path or self._get_haar_cascade()
        self.cascade = cv2.CascadeClassifier(self.haar_cascade_path)
        self.detector = MTCNN()
    
    def _get_haar_cascade(self):
        """
        Download Haar cascade file if not present locally.
        
        Returns:
            str: Path to the Haar cascade XML file
        """
        cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
        cascade_path = "haarcascade_frontalface_default.xml"
        
        if not os.path.exists(cascade_path):
            print("Downloading Haar cascade file...")
            try:
                urllib.request.urlretrieve(cascade_url, cascade_path)
                print("Haar cascade file downloaded successfully.")
            except Exception as e:
                print(f"Failed to download Haar cascade file: {e}")
                print("Please download it manually from:")
                print(cascade_url)
                raise
        
        return cascade_path
    
    @block_printing
    def haar_cascades_detect(self, frame):
        """
        Detect faces using Haar cascades.
        
        Args:
            frame: Input video frame (numpy array)
            
        Returns:
            list: [use_cascades, faces, confidence]
        """
        faces = self.cascade.detectMultiScale3(
            frame, scaleFactor=1.4, minNeighbors=3, minSize=(50, 50), 
            outputRejectLevels=True
        )
        confidence = 0
        use_cascades = True
        
        if len(faces[2]) > 0:
            confidence = faces[2][0]
            if confidence < 4:
                use_cascades = False
        else:
            use_cascades = False
            
        if len(faces[0]) > 0:
            faces = faces[0][0]
        else:
            use_cascades = False
            faces = []
            
        return [use_cascades, faces, confidence]
    
    @block_printing
    def mtcnn_detect(self, frame):
        """
        Detect faces using MTCNN.
        
        Args:
            frame: Input video frame (numpy array)
            
        Returns:
            list: [use_mtcnn, faces, confidence]
        """
        use_mtcnn = True
        confidence = 0
        faces = self.detector.detect_faces(frame)
        
        if len(faces) > 0:
            confidence = faces[0]["confidence"]
        else:
            use_mtcnn = False
            
        if len(faces) > 0 and confidence > 0.999:
            faces = faces[0]["box"]
        else:
            faces = []
            use_mtcnn = False
            
        return [use_mtcnn, faces, confidence]
    
    def process_video_realtime(self, video_path, output_path, crop_ratio=9/16, 
                             smoothing_window=25, detection_interval=15):
        """
        Process video with real-time head tracking and stabilization.
        
        Args:
            video_path (str): Path to input video file
            output_path (str): Path to output video file
            crop_ratio (float): Aspect ratio for output video (width/height)
            smoothing_window (int): Window size for smoothing face positions
            detection_interval (int): Process every Nth frame for face detection
            
        Returns:
            moviepy.VideoFileClip: Processed video clip
        """
        video_clip = mp.VideoFileClip(video_path)
        
        width = int(video_clip.size[0])
        height = int(video_clip.size[1])
        fps = video_clip.fps
        frame_count = int(video_clip.duration * fps)

        face_x = {}     # x pos of face at frame i
        face_y = {}     # y pos of face at frame i
        confidence = [] # confidence of face at frame i

        # Iterate through the frames of the video to detect the face position
        prev_face_pos = []
        print("Processing video frames for face detection...")
        
        for i in tqdm(range(0, frame_count), desc="Detecting faces"):
            frame = video_clip.get_frame(i / fps)
            faces = []
            use_cascades = False
            use_mtcnn = False

            # only apply head tracking to every Nth frame
            if (i % detection_interval == 0 or len(prev_face_pos) == 0):
                use_cascades, faces, conf = self.haar_cascades_detect(frame)

                # if haar cascades has trouble detecting face, use mtcnn
                if not use_cascades:
                    use_mtcnn, faces, conf = self.mtcnn_detect(frame)

                # if either of the models work, that's good
                if use_cascades or use_mtcnn:
                    x, y, w, h = faces
                    prev_face_pos = [x, y, w, h]
                    face_x[i] = (x + w // 2)
                    face_y[i] = (y + h // 2)
                    confidence.append(conf)
                # otherwise skip face detection
                else:
                    face_x[i] = (width // 2)
                    face_y[i] = (height // 2)
                    confidence.append(0)
            # for all other frames, inherit face position of prev frame 
            else:
                x, y, w, h = prev_face_pos
                face_x[i] = (x + w // 2)
                face_y[i] = (y + h // 2)
                confidence.append(confidence[-1])

        x_points = list(face_x.values())
        y_points = list(face_y.values())

        # Apply smoothing to face positions
        print("Smoothing face positions...")
        x_smooth = savgol_filter(x_points, smoothing_window, 2, mode="nearest")
        y_smooth = savgol_filter(y_points, smoothing_window, 2, mode="nearest")

        face_x_smooth = {}
        face_y_smooth = {}
        for index, key in enumerate(face_x):
            face_x_smooth[key] = round(x_smooth[index])

        for index, key in enumerate(face_y):
            face_y_smooth[key] = round(y_smooth[index])

        # Calculate the dimensions of the output video
        out_width = int(height * crop_ratio)
        out_height = height

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

        video_clip.reader.initialize()

        print("Processing video with head tracking...")
        for i in tqdm(range(0, frame_count), desc="Processing frames"):
            frame = video_clip.get_frame(i / fps)

            # if face detected in current frame, then crop
            if confidence[i] > 0:
                left = max(face_x_smooth[i] - out_width // 2, 0)
                right = min(left + out_width, width)
                top = max(face_y_smooth[i] - out_height // 2, 0)
                bottom = min(top + out_height, height)
                cropped_frame = frame[top:bottom, left:right]
                resized_frame = cv2.resize(cropped_frame, (out_width, out_height))
                resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                out.write(resized_frame)
            # if no face detected, center crop
            else:
                left = max((width//2) - out_width // 2, 0)
                right = min(left + out_width, width)
                top = max((height//2) - out_height // 2, 0)
                bottom = min(top + out_height, height)
                cropped_frame = frame[top:bottom, left:right]
                resized_frame = cv2.resize(cropped_frame, (out_width, out_height))
                resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                out.write(resized_frame)
        
        out.release()

        # Add audio to the processed video
        input_audio = video_clip.audio
        processed_video = mp.VideoFileClip(output_path)
        final_video = processed_video.set_audio(input_audio)
        
        # Resize to final output dimensions
        final_video = final_video.resize(height=1920, width=1080)
        final_output_path = output_path.replace('.mp4', '_final.mp4')
        final_video.write_videofile(final_output_path, codec="libx264", audio_codec="aac")
        
        return final_video

    def zoom_in_effect(self, clip):
        """
        Apply a zoom-in effect to a video clip.
        
        Args:
            clip: MoviePy video clip
            
        Returns:
            moviepy.VideoFileClip: Clip with zoom effect applied
        """
        def effect(get_frame, t):
            img = Image.fromarray(get_frame(t))
            base_size = img.size

            # new size for zoom at time = t
            # we resize images to make them bigger, and then crop at some location
            new_size = [
                math.ceil(img.size[0] * (1 + min(math.sqrt(max(0, t)), 0.1))),
                math.ceil(img.size[1] * (1 + min(math.sqrt(max(0, t)), 0.1)))
            ]
            # The new dimensions must be even.
            new_size[0] = new_size[0] + (new_size[0] % 2)
            new_size[1] = new_size[1] + (new_size[1] % 2)

            img = img.resize(new_size, Image.LANCZOS)

            # helps center the image
            x = math.ceil((new_size[0] - base_size[0]) / 2)
            y = math.ceil((new_size[1] - base_size[1]) / 2)

            # then crops and resizes the image
            img = img.crop([
                x, y, new_size[0] - x, new_size[1] - y
            ]).resize(base_size, Image.LANCZOS)

            result = numpy.array(img)
            img.close()

            # returns numpy array of image
            return result

        return clip.fl(effect)


# Convenience functions for backward compatibility
@block_printing
def haar_cascades_detect(frame, cascade_path=None):
    """Standalone function for Haar cascade face detection."""
    if cascade_path is None:
        cascade_path = "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    faces = cascade.detectMultiScale3(
        frame, scaleFactor=1.4, minNeighbors=3, minSize=(50, 50), 
        outputRejectLevels=True
    )
    confidence = 0
    use_cascades = True
    
    if len(faces[2]) > 0:
        confidence = faces[2][0]
        if confidence < 4:
            use_cascades = False
    else:
        use_cascades = False
        
    if len(faces[0]) > 0:
        faces = faces[0][0]
    else:
        use_cascades = False
        faces = []
        
    return [use_cascades, faces, confidence]


@block_printing
def mtcnn_detect(frame):
    """Standalone function for MTCNN face detection."""
    detector = MTCNN()
    use_mtcnn = True
    confidence = 0
    faces = detector.detect_faces(frame)
    
    if len(faces) > 0:
        confidence = faces[0]["confidence"]
    else:
        use_mtcnn = False
        
    if len(faces) > 0 and confidence > 0.999:
        faces = faces[0]["box"]
    else:
        faces = []
        use_mtcnn = False
        
    return [use_mtcnn, faces, confidence]


def process_video_realtime(video_path, output_path):
    """Convenience function for processing video with head tracking."""
    tracker = HeadTracker()
    return tracker.process_video_realtime(video_path, output_path)


def zoom_in_effect(clip):
    """Convenience function for applying zoom effect."""
    tracker = HeadTracker()
    return tracker.zoom_in_effect(clip)


# Example usage
if __name__ == "__main__":
    # Example usage of the HeadTracker class
    tracker = HeadTracker()
    
    # Process a video with head tracking
    input_video = "input_video.mp4"
    output_video = "output_video.mp4"
    
    if os.path.exists(input_video):
        print(f"Processing {input_video}...")
        result = tracker.process_video_realtime(input_video, output_video)
        print(f"Processed video saved as {output_video}")
    else:
        print(f"Input video {input_video} not found.")
        print("Please provide a valid video file path.")
