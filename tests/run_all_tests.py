#!/usr/bin/env python3
"""
Main test runner for the video processing project.
Runs all tests and provides a summary report.
"""

import sys
import os
import traceback
from datetime import datetime

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import test modules
try:
    from tests.test_basic import test_basic_pipeline
    from tests.test_json import create_sample_jsons
except ImportError as e:
    print(f"Error importing test modules: {e}")
    sys.exit(1)

def run_test(test_name, test_function):
    """Run a single test and return results"""
    print(f"\n{'='*50}")
    print(f"Running: {test_name}")
    print('='*50)
    
    try:
        start_time = datetime.now()
        result = test_function()
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        if result:
            print(f"✅ {test_name} PASSED ({duration:.2f}s)")
            return True, duration
        else:
            print(f"❌ {test_name} FAILED ({duration:.2f}s)")
            return False, duration
            
    except Exception as e:
        print(f"❌ {test_name} CRASHED: {str(e)}")
        print("Traceback:")
        traceback.print_exc()
        return False, 0

def main():
    """Main test runner"""
    print("🧪 Video Processing Project - Test Suite")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project root: {project_root}")
    
    # Define all tests
    tests = [
        ("Basic Pipeline Test", test_basic_pipeline),
        ("JSON Structure Test", create_sample_jsons),
    ]
    
    # Run all tests
    results = []
    total_duration = 0
    
    for test_name, test_function in tests:
        success, duration = run_test(test_name, test_function)
        results.append((test_name, success, duration))
        total_duration += duration
    
    # Summary report
    print(f"\n{'='*60}")
    print("TEST SUMMARY REPORT")
    print('='*60)
    
    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed
    
    for test_name, success, duration in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:12} {test_name:30} ({duration:.2f}s)")
    
    print(f"\n📊 Results: {passed} passed, {failed} failed")
    print(f"⏱️  Total time: {total_duration:.2f}s")
    
    if failed > 0:
        print(f"\n❌ {failed} test(s) failed!")
        return False
    else:
        print(f"\n🎉 All {passed} tests passed!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 