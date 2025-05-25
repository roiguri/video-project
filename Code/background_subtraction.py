from utils import VideoProcessor
import numpy as np

class BackgroundSubtractor(VideoProcessor):
    def __init__(self):
        super().__init__()
    
    def extract_person(self, input_path: str, extracted_path: str, binary_path: str):
        """Extract person using background subtraction"""
        print("Background Subtraction: TODO - Implement in Phase 3")
        # Placeholder implementation
        frames, metadata = self.read_video(input_path)
        
        # Create placeholder binary masks (all white for now)
        binary_frames = []
        for frame in frames:
            # Create a binary mask (all ones for placeholder)
            binary_mask = np.ones((frame.shape[0], frame.shape[1]), dtype=np.uint8) * 255
            binary_frames.append(binary_mask)
        
        # Save extracted video (same as input for now)
        self.write_video(frames, extracted_path, metadata['fps'])
        
        # Save binary masks
        self.write_video(binary_frames, binary_path, metadata['fps']) 