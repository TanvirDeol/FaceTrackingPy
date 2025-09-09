"""
Head Tracking CLI Tool

Usage:
    python head_tracking.py input_video.mp4
"""

import cv2
import moviepy.editor as mp
from tqdm import tqdm
import math
from PIL import Image
import numpy
from scipy.signal import savgol_filter
import numpy as np
from mtcnn.mtcnn import MTCNN
import sys
import os
import urllib.request
import argparse


class HeadTracker:
    def __init__(self):
        self.cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.detector = MTCNN()
    
    def detect_face(self, frame):
        faces = self.cascade.detectMultiScale3(frame, scaleFactor=1.4, minNeighbors=3, 
                                             minSize=(50, 50), outputRejectLevels=True)
        if len(faces[0]) > 0 and len(faces[2]) > 0 and faces[2][0] >= 4:
            return faces[0][0]
        
        faces = self.detector.detect_faces(frame)
        if len(faces) > 0 and faces[0]["confidence"] > 0.999:
            return faces[0]["box"]
        
        return None
    
    def process_video(self, video_path, output_path):
        video_clip = mp.VideoFileClip(video_path)
        width, height = int(video_clip.size[0]), int(video_clip.size[1])
        fps = video_clip.fps
        frame_count = int(video_clip.duration * fps)

        face_x, face_y = {}, {}
        prev_face = None

        for i in tqdm(range(0, frame_count, 15), desc="Detecting faces"):
            frame = video_clip.get_frame(i / fps)
            face = self.detect_face(frame)
            
            if face is not None:
                x, y, w, h = face
                prev_face = [x, y, w, h]
                face_x[i] = x + w // 2
                face_y[i] = y + h // 2
            elif prev_face is not None:
                x, y, w, h = prev_face
                face_x[i] = x + w // 2
                face_y[i] = y + h // 2
            else:
                face_x[i] = width // 2
                face_y[i] = height // 2

        # Fill missing frames
        for i in range(frame_count):
            if i not in face_x:
                face_x[i] = face_x.get(i-1, width // 2)
                face_y[i] = face_y.get(i-1, height // 2)

        # Smooth positions
        x_smooth = savgol_filter(list(face_x.values()), 25, 2, mode="nearest")
        y_smooth = savgol_filter(list(face_y.values()), 25, 2, mode="nearest")

        # Process video
        out_width = int(height * 9/16)
        out_height = height
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

        for i in tqdm(range(frame_count), desc="Processing"):
            frame = video_clip.get_frame(i / fps)
            x, y = int(x_smooth[i]), int(y_smooth[i])
            
            left = max(x - out_width // 2, 0)
            right = min(left + out_width, width)
            top = max(y - out_height // 2, 0)
            bottom = min(top + out_height, height)
            
            cropped = frame[top:bottom, left:right]
            resized = cv2.resize(cropped, (out_width, out_height))
            out.write(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

        out.release()

        # Add audio and resize
        processed = mp.VideoFileClip(output_path).set_audio(video_clip.audio)
        final = processed.resize(height=1920, width=1080)
        final_output = output_path.replace('.mp4', '_final.mp4')
        final.write_videofile(final_output, codec="libx264", audio_codec="aac")
        
        return final

def main():
    parser = argparse.ArgumentParser(description="Head Tracking Tool")
    parser.add_argument('input_video', help='Path to input video file')
    args = parser.parse_args()
    
    output_path = args.input_video.replace('.mp4', '_tracked.mp4')
    
    tracker = HeadTracker()
    tracker.process_video(args.input_video, output_path)
    print(f"Output saved as: {output_path.replace('.mp4', '_final.mp4')}")


# CLI entry point
if __name__ == "__main__":
    main()
