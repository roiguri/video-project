#!/usr/bin/env python3
"""
Convenience script to run all tests from the project root.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the main test runner
from tests.run_all_tests import main

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 