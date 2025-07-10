import os
import sys
import time
from utils import VideoUtils
from stabilization import VideoStabilizer
from background_subtraction import BackgroundSubtractor
from matting import VideoMatter
from tracking import PersonTracker

# Configuration - Change these to YES/NO to control which stages run
RUN_STABILIZATION = "YES"
RUN_BACKGROUND_SUB = "YES"
RUN_MATTING = "YES"
RUN_TRACKING = "YES"

def main():
    print("=== Video Processing Project Started ===")
    start_time = time.time()
    
    try:
        # IDs for output file naming
        ID1, ID2, ID3 = "211314471", "211360136", "318835816"
        os.makedirs('Outputs', exist_ok=True)
        
        utils = VideoUtils()
        input_video = 'Input/INPUT.avi'
        background_img = 'Input/background.jpg'
        
        output_files = {
            'stabilize': f'Outputs/stabilize_{ID1}_{ID2}_{ID3}.avi',
            'extracted': f'Outputs/extracted_{ID1}_{ID2}_{ID3}.avi',
            'binary': f'Outputs/binary_{ID1}_{ID2}_{ID3}.avi',
            'alpha': f'Outputs/alpha_{ID1}_{ID2}_{ID3}.avi',
            'matted': f'Outputs/matted_{ID1}_{ID2}_{ID3}.avi',
            'output': f'Outputs/OUTPUT_{ID1}_{ID2}_{ID3}.avi'
        }
        
        # Phase 1: Video Stabilization
        if RUN_STABILIZATION == "YES":
            print("\n--- Phase 1: Video Stabilization ---")
            stabilizer = VideoStabilizer()
            stabilizer.apply_stabilization(
                input_video,
                output_files['stabilize'],
                utils
            )
        else:
            print("\n--- Phase 1: Video Stabilization (SKIPPED) ---")
        
        # Phase 2: Background Subtraction
        if RUN_BACKGROUND_SUB == "YES":
            print("\n--- Phase 2: Background Subtraction ---")
            bg_subtractor = BackgroundSubtractor()
            bg_subtractor.apply_subtraction(
                output_files['stabilize'], background_img,
                output_files['extracted'], output_files['binary'],
                utils
            )
        else:
            print("\n--- Phase 2: Background Subtraction (SKIPPED) ---")
        
        # Phase 3: Image Matting
        if RUN_MATTING == "YES":
            print("\n--- Phase 3: Image Matting ---")
            matter = VideoMatter()
            matter.apply_matting(
                output_files['extracted'], output_files['binary'], background_img,
                output_files['matted'], output_files['alpha'],
                utils
            )
        else:
            print("\n--- Phase 3: Image Matting (SKIPPED) ---")
        
        # Phase 4: Person Tracking
        if RUN_TRACKING == "YES":
            print("\n--- Phase 4: Person Tracking ---")
            tracker = PersonTracker()
            tracking_results = tracker.apply_tracking(
                output_files['matted'], output_files['binary'], output_files['output'],
                utils
            )
            tracker.save_tracking_json('Outputs', tracking_results)
        else:
            print("\n--- Phase 4: Person Tracking (SKIPPED) ---")
        
        utils.save_timing_json('Outputs')
        total_time = time.time() - start_time
        print(f"\n=== Processing Complete! Total time: {total_time:.2f}s ===")
        
        if total_time > 1200:  # 20 minutes
            print("WARNING: Processing took longer than 20 minutes!")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 