from utils import VideoProcessor
import json
import os
import cv2
import numpy as np
from tqdm import tqdm
import time

class PersonTracker(VideoProcessor):
    def __init__(self):
        super().__init__()
    
    def get_bounding_box_from_binary(self, binary_frame):
        """Extract bounding box from binary frame by finding white pixels"""
        # Find all white pixels (person pixels)
        indices = np.argwhere(binary_frame == 255)
        
        if len(indices) == 0:
            # No person found, return default box
            return 100, 100, 200, 300
        
        # Get min/max coordinates
        min_y = np.min(indices[:, 0])
        max_y = np.max(indices[:, 0])
        min_x = np.min(indices[:, 1])
        max_x = np.max(indices[:, 1])
        
        # Convert to [row, col, height, width] format
        row = min_y
        col = min_x
        height = max_y - min_y
        width = max_x - min_x
        
        return row, col, height, width
    
    def get_box_list_from_binary_video(self, binary_video_path):
        """Extract bounding boxes from all frames in binary video"""
        frames, _ = self.read_video(binary_video_path)
        box_list = []
        
        for frame in tqdm(frames, desc="Extracting bounding boxes", leave=False, ncols=80):
            # Convert to grayscale if needed
            if len(frame.shape) == 3:
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray_frame = frame
            
            bbox = self.get_bounding_box_from_binary(gray_frame)
            box_list.append(bbox)
        
        return box_list
    
    def draw_bounding_box(self, frame, bbox):
        """Draw bounding box on frame"""
        row, col, height, width = bbox
        
        # Draw rectangle
        top_left = (col, row)
        bottom_right = (col + width, row + height)
        
        frame_with_box = frame.copy()
        cv2.rectangle(frame_with_box, top_left, bottom_right, (0, 255, 0), 2)
        
        return frame_with_box
    
    def track_person(self, input_path: str, output_path: str):
        """Track person using binary video bounding boxes applied to matted video"""
        start_time = time.time()
        
        # Get bounding boxes from binary video
        binary_video_path = input_path.replace('matted', 'binary')
        box_list = self.get_box_list_from_binary_video(binary_video_path)
        
        # Read matted video frames
        matted_frames, metadata = self.read_video(input_path)
        
        if not matted_frames:
            raise ValueError(f"No frames found in {input_path}")
        
        tracking_results = {}
        output_frames = []
        
        # Apply bounding boxes to matted frames
        n_frames = min(len(matted_frames), len(box_list))
        
        for frame_idx in tqdm(range(n_frames), desc="Tracking frames", leave=False, ncols=80):
            frame = matted_frames[frame_idx]
            bbox = box_list[frame_idx]
            
            # Store tracking result (convert numpy int64 to regular int for JSON serialization)
            tracking_results[str(frame_idx)] = [int(x) for x in bbox]
            
            # Draw bounding box on frame
            frame_with_box = self.draw_bounding_box(frame, bbox)
            
            # Add frame counter
            cv2.putText(frame_with_box, f"Frame {frame_idx}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            output_frames.append(frame_with_box)
        
        # Save output video
        self.write_video(output_frames, output_path, metadata['fps'])
        
        total_time = time.time() - start_time
        
        return tracking_results
    
    def save_tracking_json(self, output_dir: str, tracking_results: dict):
        """Save tracking results to JSON"""
        tracking_path = os.path.join(output_dir, 'tracking.json')
        with open(tracking_path, 'w') as f:
            json.dump(tracking_results, f, indent=2)
        print(f"Tracking data saved to {tracking_path}")