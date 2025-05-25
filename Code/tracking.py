from utils import VideoProcessor
import json
import os
import cv2
import numpy as np

class PersonTracker(VideoProcessor):
    def __init__(self):
        super().__init__()
    
    def track_person(self, input_path: str, output_path: str):
        """Track person and draw bounding boxes"""
        print("Tracking: TODO - Implement in Phase 5")
        # Placeholder implementation
        
        frames, metadata = self.read_video(input_path)
        
        # Create placeholder tracking results
        tracking_results = {}
        
        # Simple placeholder: create dummy bounding boxes
        frame_height, frame_width = frames[0].shape[:2]
        center_x, center_y = frame_width // 2, frame_height // 2
        box_width, box_height = 100, 150
        
        output_frames = []
        
        for i, frame in enumerate(frames):
            # Create dummy bounding box coordinates [ROW, COL, HEIGHT, WIDTH]
            # Add slight movement for realistic placeholder
            offset_x = int(5 * np.sin(i * 0.1))
            offset_y = int(3 * np.cos(i * 0.1))
            
            row = center_y - box_height // 2 + offset_y
            col = center_x - box_width // 2 + offset_x
            
            # Ensure bounding box stays within frame
            row = max(0, min(row, frame_height - box_height))
            col = max(0, min(col, frame_width - box_width))
            
            tracking_results[str(i)] = [row, col, box_height, box_width]
            
            # Draw bounding box on frame
            frame_with_box = frame.copy()
            cv2.rectangle(frame_with_box, 
                         (col, row), 
                         (col + box_width, row + box_height), 
                         (0, 255, 0), 2)
            
            # Add frame number text
            cv2.putText(frame_with_box, f"Frame {i}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)
            
            output_frames.append(frame_with_box)
        
        # Save output video with bounding boxes
        self.write_video(output_frames, output_path, metadata['fps'])
        
        return tracking_results
    
    def save_tracking_json(self, output_dir: str, tracking_results: dict):
        """Save tracking results to JSON"""
        tracking_path = os.path.join(output_dir, 'tracking.json')
        with open(tracking_path, 'w') as f:
            json.dump(tracking_results, f, indent=2)
        print(f"Tracking data saved to {tracking_path}") 