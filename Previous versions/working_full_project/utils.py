import cv2
import numpy as np
import os
import json
import time
from typing import Tuple, List, Dict, Any

class VideoProcessor:
    """Base class for video processing operations"""
    
    def __init__(self):
        self.start_time = time.time()
        self.timing_data = {}
    
    def read_video(self, video_path: str) -> Tuple[List[np.ndarray], Dict]:
        """Read video and return frames with metadata"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Reading video: {frame_count} frames, {width}x{height}, {fps} FPS")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        metadata = {
            'fps': fps,
            'width': width,
            'height': height,
            'frame_count': len(frames)
        }
        
        return frames, metadata
    
    def write_video(self, frames: List[np.ndarray], output_path: str, 
                   fps: float = 30.0, codec: str = 'XVID'):
        """Write frames to video file"""
        if not frames:
            raise ValueError("No frames to write")
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            # Ensure frame is in correct format
            if len(frame.shape) == 2:  # Grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            out.write(frame)
        
        out.release()
        print(f"Saved video: {output_path} ({len(frames)} frames)")
    
    def record_timing(self, stage_name: str):
        """Record timing for current stage"""
        current_time = time.time() - self.start_time
        self.timing_data[stage_name] = current_time
        print(f"Stage '{stage_name}' completed at {current_time:.2f}s")
    
    def save_timing_json(self, output_dir: str):
        """Save timing data to JSON file"""
        timing_path = os.path.join(output_dir, 'timing.json')
        with open(timing_path, 'w') as f:
            json.dump(self.timing_data, f, indent=2)
        print(f"Timing data saved to {timing_path}")

def validate_paths():
    """Validate that required input files exist"""
    required_files = [
        'Inputs/INPUT.avi',
        'Inputs/background.jpg'
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")
    
    # Create output directory if it doesn't exist
    os.makedirs('Outputs', exist_ok=True)
    print("Path validation completed successfully")

def get_student_ids() -> Tuple[str, str]:
    """Get student IDs for file naming"""
    # TODO: Replace with actual student IDs
    ID1 = "123456789"  # Replace with first student ID
    ID2 = "987654321"  # Replace with second student ID
    ID3 = "123456789"  # Replace with third student ID
    return ID1, ID2, ID3