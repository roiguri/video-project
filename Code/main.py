import os
import sys
import time
from utils import VideoProcessor
from stabilization import VideoStabilizer
from background_subtraction import BackgroundSubtractor
from matting import VideoMatter
from tracking import PersonTracker

# =============================================================================
# CONFIGURATION - Change these to YES/NO to control which stages run
# =============================================================================
RUN_STABILIZATION = "YES"     # Phase 1: Video Stabilization
RUN_BACKGROUND_SUB = "NO"    # Phase 2: Background Subtraction  
RUN_MATTING = "NO"          # Phase 3: Image Matting
RUN_TRACKING = "NO"          # Phase 4: Person Tracking
# =============================================================================

def main():
    """Main execution function"""
    print("=== Video Processing Project Started ===")
    start_time = time.time()
    
    try:
        # Student IDs for output file naming
        ID1, ID2, ID3 = "123456789", "987654321", "111222333"
        
        # Create output directory if it doesn't exist
        os.makedirs('Outputs', exist_ok=True)
        
        # Initialize processor
        processor = VideoProcessor()
        
        # Define file paths
        input_video = 'Inputs/INPUT.avi'
        background_img = 'Inputs/background.jpg'
        
        output_files = {
            'stabilized': f'Outputs/stabilized_{ID1}_{ID2}_{ID3}.avi',
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
                output_files['stabilized']
            )
            processor.record_timing('stabilized')
        else:
            print("\n--- Phase 1: Video Stabilization (SKIPPED) ---")
        
        # Phase 2: Background Subtraction
        if RUN_BACKGROUND_SUB == "YES":
            print("\n--- Phase 2: Background Subtraction ---")
            bg_subtractor = BackgroundSubtractor()
            bg_subtractor.apply_subtraction(
                output_files['stabilized'],  # Use existing stabilized video
                background_img,
                output_files['extracted'],
                output_files['binary']
            )
            processor.record_timing('extracted')
            processor.record_timing('binary')
        else:
            print("\n--- Phase 2: Background Subtraction (SKIPPED) ---")
        
        # Phase 3: Image Matting
        if RUN_MATTING == "YES":
            print("\n--- Phase 3: Image Matting ---")
            matter = VideoMatter()
            matter.apply_matting(
                output_files['extracted'],   # Use existing extracted video
                output_files['binary'],      # Use existing binary mask
                background_img,              # Background image
                output_files['matted'],      # Generate matted video
                output_files['alpha']        # Generate alpha channel
            )
            processor.record_timing('matted')
            processor.record_timing('alpha')
        else:
            print("\n--- Phase 3: Image Matting (SKIPPED) ---")
        
        # Phase 4: Person Tracking
        if RUN_TRACKING == "YES":
            print("\n--- Phase 4: Person Tracking ---")
            tracker = PersonTracker()
            tracking_results = tracker.apply_tracking(
                output_files['matted'],  # Use matted video as specified
                output_files['output']
            )
            tracker.save_tracking_json('Outputs', tracking_results)
            processor.record_timing('OUTPUT')
        else:
            print("\n--- Phase 4: Person Tracking (SKIPPED) ---")
        
        # Save timing results
        processor.save_timing_json('Outputs')
        
        total_time = time.time() - start_time
        print(f"\n=== Processing Complete! Total time: {total_time:.2f}s ===")
        
        if total_time > 1200:  # 20 minutes
            print("WARNING: Processing took longer than 20 minutes!")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 