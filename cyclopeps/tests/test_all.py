"""
Run all unit tests in the directory cyclomps/cyclomps/tests/

Author: Phillip Helms <phelms@caltech.edu>
Date: June 2019

"""

import unittest
import os

if __name__ == "__main__":
    # Run all unit tests in this directory and its subdirectories
    loader = unittest.TestLoader()
    test_dir = os.path.dirname(os.path.realpath(__file__))
    
    mpiprint(0,'#'*50)
    mpiprint(0,'Running unittests in: {}'.format(test_dir))
    mpiprint(0,'-'*50)

    # Find all available tests
    test_suite = loader.discover(test_dir)

    # Run all tests
    runner = unittest.TextTestRunner()
    runner.run(test_suite)
