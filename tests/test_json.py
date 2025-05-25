import json
import os
import sys

def create_sample_jsons():
    """Create sample JSON files with correct structure"""
    
    # Ensure Outputs directory exists (relative to project root)
    project_root = os.path.join(os.path.dirname(__file__), '..')
    outputs_dir = os.path.join(project_root, 'Outputs')
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Sample timing.json
    timing_sample = {
        "stabilize": 45.23,
        "extracted": 89.45,
        "binary": 89.45,
        "matted": 156.78,
        "alpha": 156.78,
        "OUTPUT": 201.34
    }
    
    timing_file = os.path.join(outputs_dir, 'timing_sample.json')
    with open(timing_file, 'w') as f:
        json.dump(timing_sample, f, indent=2)
    
    # Sample tracking.json
    tracking_sample = {
        "0": [100, 150, 80, 120],  # [ROW, COL, HEIGHT, WIDTH]
        "1": [102, 148, 82, 118],
        "2": [104, 146, 84, 116],
        "3": [106, 144, 86, 114],
        "4": [108, 142, 88, 112]
    }
    
    tracking_file = os.path.join(outputs_dir, 'tracking_sample.json')
    with open(tracking_file, 'w') as f:
        json.dump(tracking_sample, f, indent=2)
    
    print("✓ Sample JSON files created:")
    print(f"  - {timing_file}")
    print(f"  - {tracking_file}")
    
    # Validate JSON structure
    try:
        # Test timing JSON
        with open(timing_file, 'r') as f:
            timing_data = json.load(f)
        
        required_timing_keys = ['stabilize', 'extracted', 'binary', 'matted', 'alpha', 'OUTPUT']
        for key in required_timing_keys:
            if key not in timing_data:
                print(f"✗ Missing timing key: {key}")
                return False
        print("✓ Timing JSON structure is correct")
        
        # Test tracking JSON
        with open(tracking_file, 'r') as f:
            tracking_data = json.load(f)
        
        # Check that all values are lists of 4 integers [ROW, COL, HEIGHT, WIDTH]
        for frame_id, bbox in tracking_data.items():
            if not isinstance(bbox, list) or len(bbox) != 4:
                print(f"✗ Invalid tracking bbox format for frame {frame_id}: {bbox}")
                return False
        print("✓ Tracking JSON structure is correct")
        
    except Exception as e:
        print(f"✗ JSON validation error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = create_sample_jsons()
    if success:
        print("\n=== JSON test completed successfully! ===")
    else:
        print("\n=== JSON test failed! ===")
        exit(1) 