import os
import sys
import time
from utils import VideoProcessor, validate_paths, get_student_ids

# Import processing modules (to be implemented in later phases)
from stabilization import VideoStabilizer
from background_subtraction import BackgroundSubtractor
from matting import VideoMatter
from tracking import PersonTracker

def main():
    """Main execution function"""
    print("=== Video Processing Project Started ===")
    start_time = time.time()
    
    try:
        # TODO: consider removing this function
        # Validate environment
        validate_paths()
        ID1, ID2, ID3 = get_student_ids()
        
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
        # print("\n--- Phase 1: Video Stabilization ---")
        # stabilizer = VideoStabilizer()
        # stabilizer.stabilize_video(input_video, output_files['stabilized'])
        # processor.record_timing('stabilize')
        
        # Phase 2: Background Subtraction (SKIPPED - using existing outputs)
        print("\n--- Phase 2: Background Subtraction (SKIPPED) ---")
        print("Using existing background subtraction outputs...")
        processor.record_timing('binary')
        
        # Phase 3: Image Matting (SKIPPED - using existing outputs)
        print("\n--- Phase 3: Image Matting (SKIPPED) ---")
        print("Using existing matting outputs...")
        processor.record_timing('matted')
        processor.record_timing('alpha')
        
        # Phase 4: Person Tracking
        print("\n--- Phase 4: Person Tracking ---")
        tracker = PersonTracker()
        tracking_results = tracker.track_person(
            output_files['matted'],  # Use matted video as specified
            output_files['output']
        )
        processor.record_timing('OUTPUT')
        
        # Save timing and tracking results
        processor.save_timing_json('Outputs')
        tracker.save_tracking_json('Outputs', tracking_results)
        
        total_time = time.time() - start_time
        print(f"\n=== Processing Complete! Total time: {total_time:.2f}s ===")
        
        if total_time > 1200:  # 20 minutes
            print("WARNING: Processing took longer than 20 minutes!")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 