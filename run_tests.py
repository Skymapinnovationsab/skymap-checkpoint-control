#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Test runner for SkyMap Checkpoint Control Web UI
# Author: ChatGPT for Jon Bengtsson (SkyMap)

import os
import sys
import subprocess
import unittest
from pathlib import Path

def run_tests():
    """Run all tests for the web UI application"""
    print("🧪 Running SkyMap Checkpoint Control Web UI Tests")
    print("=" * 60)
    
    # Get the project root directory
    project_root = Path(__file__).parent
    tests_dir = project_root / "tests"
    
    # Check if tests directory exists
    if not tests_dir.exists():
        print("❌ Tests directory not found!")
        return False
    
    # Add project root to Python path
    sys.path.insert(0, str(project_root))
    
    # Discover and run tests
    loader = unittest.TestLoader()
    suite = loader.discover(str(tests_dir), pattern='test_*.py')
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\n❌ Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    if result.skipped:
        print("\n⚠️  Skipped:")
        for test, reason in result.skipped:
            print(f"  - {test}: {reason}")
    
    # Return success/failure
    success = result.wasSuccessful()
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    return success

def run_specific_test(test_name):
    """Run a specific test by name"""
    print(f"🧪 Running specific test: {test_name}")
    print("=" * 60)
    
    # Get the project root directory
    project_root = Path(__file__).parent
    tests_dir = project_root / "tests"
    
    # Add project root to Python path
    sys.path.insert(0, str(project_root))
    
    # Import and run specific test
    try:
        # Try to import the test module
        test_module = __import__(f"tests.{test_name}", fromlist=[''])
        
        # Find test classes in the module
        test_classes = []
        for attr_name in dir(test_module):
            attr = getattr(test_module, attr_name)
            if (isinstance(attr, type) and 
                issubclass(attr, unittest.TestCase) and 
                attr != unittest.TestCase):
                test_classes.append(attr)
        
        if not test_classes:
            print(f"❌ No test classes found in {test_name}")
            return False
        
        # Create test suite for the specific test
        loader = unittest.TestLoader()
        suite = unittest.TestSuite()
        
        for test_class in test_classes:
            suite.addTests(loader.loadTestsFromTestCase(test_class))
        
        # Run the specific test
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
        
    except ImportError as e:
        print(f"❌ Could not import test module {test_name}: {e}")
        return False
    except Exception as e:
        print(f"❌ Error running test {test_name}: {e}")
        return False

def run_coverage():
    """Run tests with coverage reporting"""
    print("📊 Running tests with coverage reporting")
    print("=" * 60)
    
    try:
        # Check if coverage is installed
        import coverage
    except ImportError:
        print("❌ Coverage not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "coverage"])
    
    # Run coverage
    cmd = [
        sys.executable, "-m", "coverage", "run", 
        "--source=web_ui", 
        "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Tests completed successfully")
        
        # Generate coverage report
        print("\n📈 Generating coverage report...")
        subprocess.run([sys.executable, "-m", "coverage", "report"])
        subprocess.run([sys.executable, "-m", "coverage", "html"])
        print("📁 HTML coverage report generated in htmlcov/")
        
        return True
    else:
        print("❌ Tests failed during coverage run")
        print(result.stderr)
        return False

def main():
    """Main function to run tests"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run tests for SkyMap Checkpoint Control Web UI")
    parser.add_argument("--test", "-t", help="Run a specific test module")
    parser.add_argument("--coverage", "-c", action="store_true", help="Run tests with coverage")
    parser.add_argument("--all", "-a", action="store_true", help="Run all tests (default)")
    
    args = parser.parse_args()
    
    if args.coverage:
        success = run_coverage()
    elif args.test:
        success = run_specific_test(args.test)
    else:
        # Default: run all tests
        success = run_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
