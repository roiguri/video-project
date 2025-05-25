from utils import VideoProcessor

class VideoStabilizer(VideoProcessor):
    def __init__(self):
        super().__init__()
    
    def stabilize(self, input_path: str, output_path: str):
        """Stabilize shaky video"""
        print("Stabilization: TODO - Implement in Phase 2")
        # Placeholder: copy input to output for now
        frames, metadata = self.read_video(input_path)
        self.write_video(frames, output_path, metadata['fps']) 