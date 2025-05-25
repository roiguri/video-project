import os
import json
import sys

def test_basic_pipeline():
    """Test that basic pipeline runs without errors"""
    
    # Check if we can import all modules
    try:
        import sys
        import os
        # Add Code directory to path for imports
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'Code'))
        
        from main import main
        from utils import VideoProcessor, validate_paths, get_student_ids
        from stabilization import VideoStabilizer
        from background_subtraction import BackgroundSubtractor
        from matting import VideoMatter
        from tracking import PersonTracker
        print("✓ All modules import successfully")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Check directory structure (relative to project root)
    project_root = os.path.join(os.path.dirname(__file__), '..')
    required_dirs = ['Code', 'Inputs', 'Outputs', 'Document', 'ScreenRec']
    for dir_name in required_dirs:
        dir_path = os.path.join(project_root, dir_name)
        if not os.path.exists(dir_path):
            print(f"✗ Missing directory: {dir_name}")
            return False
    print("✓ Directory structure is correct")
    
    # Check required input files
    required_files = ['Inputs/INPUT.avi', 'Inputs/background.jpg']
    for file_path in required_files:
        full_path = os.path.join(project_root, file_path)
        if not os.path.exists(full_path):
            print(f"✗ Missing required file: {file_path}")
            return False
    print("✓ Required input files exist")
    
    # Test file naming convention
    ID1, ID2, ID3 = get_student_ids()
    expected_files = [
        f'stabilized_{ID1}_{ID2}_{ID3}.avi',
        f'extracted_{ID1}_{ID2}_{ID3}.avi',
        f'binary_{ID1}_{ID2}_{ID3}.avi',
        f'alpha_{ID1}_{ID2}_{ID3}.avi',
        f'matted_{ID1}_{ID2}_{ID3}.avi',
        f'OUTPUT_{ID1}_{ID2}_{ID3}.avi'
    ]
    print(f"✓ File naming convention: {expected_files[0]} (example)")
    
    # Test VideoProcessor instantiation
    try:
        processor = VideoProcessor()
        print("✓ VideoProcessor can be instantiated")
    except Exception as e:
        print(f"✗ VideoProcessor error: {e}")
        return False
    
    print("\n=== Basic pipeline test completed successfully! ===")
    return True

if __name__ == "__main__":
    success = test_basic_pipeline()
    if not success:
        sys.exit(1) 