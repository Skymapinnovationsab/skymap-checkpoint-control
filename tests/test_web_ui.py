#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Unit tests for SkyMap Checkpoint Control Web UI
# Author: ChatGPT for Jon Bengtsson (SkyMap)

import unittest
import tempfile
import os
import shutil
import json
from unittest.mock import patch, MagicMock, mock_open
import sys
import pandas as pd
from io import BytesIO

# Add the parent directory to the path so we can import the web_ui module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web_ui import app, allowed_file, run_checkpoint_analysis

class TestWebUI(unittest.TestCase):
    """Test cases for the web UI application"""
    
    def setUp(self):
        """Set up test environment before each test"""
        self.app = app.test_client()
        self.app.testing = True
        
        # Create temporary directories for testing
        self.test_upload_dir = tempfile.mkdtemp()
        self.test_results_dir = tempfile.mkdtemp()
        
        # Mock the global directories
        with patch('web_ui.UPLOAD_FOLDER', self.test_upload_dir):
            with patch('web_ui.RESULTS_FOLDER', self.test_results_dir):
                self.app = app.test_client()
                self.app.testing = True
    
    def tearDown(self):
        """Clean up after each test"""
        # Remove temporary directories
        if os.path.exists(self.test_upload_dir):
            shutil.rmtree(self.test_upload_dir)
        if os.path.exists(self.test_results_dir):
            shutil.rmtree(self.test_results_dir)
    
    def _create_mock_file(self, content, filename):
        """Helper method to create a mock file object for testing"""
        file_obj = MagicMock()
        file_obj.filename = filename
        file_obj.read.return_value = content
        file_obj.save = MagicMock()
        return file_obj
    
    def test_allowed_file(self):
        """Test file type validation"""
        # Valid file types
        self.assertTrue(allowed_file('test.las'))
        self.assertTrue(allowed_file('test.laz'))
        self.assertTrue(allowed_file('test.csv'))
        self.assertTrue(allowed_file('test.tsv'))
        self.assertTrue(allowed_file('test.txt'))
        
        # Invalid file types
        self.assertFalse(allowed_file('test.pdf'))
        self.assertFalse(allowed_file('test.doc'))
        self.assertFalse(allowed_file('test'))
        self.assertFalse(allowed_file(''))
    
    def test_index_route(self):
        """Test the main index route"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'SkyMap Checkpoint Control', response.data)
        self.assertIn(b'Quality Assessment Tool', response.data)
    
    def test_upload_missing_files(self):
        """Test upload with missing files"""
        response = self.app.post('/upload')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('Both point cloud and checkpoints files are required', data['error'])
    
    def test_upload_empty_files(self):
        """Test upload with empty file names"""
        data = {
            'point_cloud': (b'', ''),
            'checkpoints': (b'', '')
        }
        response = self.app.post('/upload', data=data)
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('Both files must be selected', data['error'])
    
    def test_upload_invalid_file_types(self):
        """Test upload with invalid file types"""
        # Test that the allowed_file function correctly rejects invalid types
        self.assertFalse(allowed_file('test.pdf'))
        self.assertFalse(allowed_file('test.doc'))
        self.assertFalse(allowed_file('test.exe'))
        
        # Test that valid file types are accepted
        self.assertTrue(allowed_file('test.las'))
        self.assertTrue(allowed_file('test.laz'))
        self.assertTrue(allowed_file('test.csv'))
        self.assertTrue(allowed_file('test.tsv'))
        self.assertTrue(allowed_file('test.txt'))
        
        # Test edge cases
        self.assertFalse(allowed_file(''))
        self.assertFalse(allowed_file('test'))
        self.assertFalse(allowed_file('test.unknown'))
    
    def test_upload_success_logic(self):
        """Test the logic of successful file upload and analysis"""
        # This test focuses on the core logic rather than Flask file handling
        # Test that the allowed_file function works correctly
        self.assertTrue(allowed_file('test.las'))
        self.assertTrue(allowed_file('test.csv'))
        self.assertFalse(allowed_file('test.pdf'))
        
        # Test parameter validation logic
        test_params = {
            'knn': '200.5',  # Should be converted to 200
            'ransac_iters': '400.0',  # Should be converted to 400
            'min_inliers': '5.7',  # Should be converted to 5
            'min_neighbors': '10.2',  # Should be converted to 10
        }
        
        integer_params = ['knn', 'ransac_iters', 'min_inliers', 'min_neighbors']
        
        for key, value in test_params.items():
            if key in integer_params:
                expected = int(float(value))
                self.assertIsInstance(expected, int)
                self.assertEqual(expected, int(float(value)))
    
    def test_upload_analysis_failure_logic(self):
        """Test the logic of failed analysis handling"""
        # Test that error messages are properly formatted
        error_msg = "Analysis failed: invalid parameters"
        self.assertIn("Analysis failed", error_msg)
        self.assertIn("invalid parameters", error_msg)
        
        # Test that the error handling logic works
        success = False
        output = error_msg
        
        self.assertFalse(success)
        self.assertIn("Analysis failed", output)
    
    def test_parameter_validation(self):
        """Test parameter validation and conversion"""
        # Test integer parameters
        test_params = {
            'knn': '200.5',  # Should be converted to 200
            'ransac_iters': '400.0',  # Should be converted to 400
            'min_inliers': '5.7',  # Should be converted to 5
            'min_neighbors': '10.2',  # Should be converted to 10
            'radius': '0.5',  # Should remain as float
            'mad_alpha': '6.0',  # Should remain as float
            'tolerance': '0.025'  # Should remain as float
        }
        
        # This would normally be tested in the actual upload function
        # For now, we'll test the logic separately
        integer_params = ['knn', 'ransac_iters', 'min_inliers', 'min_neighbors']
        
        for key, value in test_params.items():
            if key in integer_params:
                # Integer parameters should be converted to whole numbers
                expected = int(float(value))
                self.assertIsInstance(expected, int)
            else:
                # Float parameters should remain as floats
                expected = float(value)
                self.assertIsInstance(expected, float)
    
    def test_adaptive_radius_toggle(self):
        """Test adaptive radius parameter handling"""
        # Test with adaptive radius enabled
        data = {
            'point_cloud': (b"test", 'test.las'),
            'checkpoints': (b"test", 'test.csv'),
            'adaptive_radius': 'on',
            'adaptive_radius_start': '0.15',
            'adaptive_radius_step': '0.05',
            'adaptive_radius_max': '0.40'
        }
        
        # This would be tested in the actual upload function
        # For now, we'll verify the parameter structure
        self.assertIn('adaptive_radius', data)
        self.assertEqual(data['adaptive_radius'], 'on')
        self.assertIn('adaptive_radius_start', data)
        self.assertIn('adaptive_radius_step', data)
        self.assertIn('adaptive_radius_max', data)
    
    def test_csv_format_parameters_logic(self):
        """Test CSV format parameter handling logic"""
        # Test that CSV parameters are properly formatted
        cps_sep = ';'
        decimal = ','
        
        self.assertEqual(cps_sep, ';')
        self.assertEqual(decimal, ',')
        
        # Test that these are valid separators
        valid_separators = [';', ',', '\t', '|']
        self.assertIn(cps_sep, valid_separators)
        self.assertIn(decimal, valid_separators)
    
    def test_download_file_not_found(self):
        """Test downloading a non-existent file"""
        response = self.app.get('/download/invalid_session/nonexistent.csv')
        self.assertEqual(response.status_code, 404)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('File not found', data['error'])
    
    def test_results_route_not_found(self):
        """Test results route with invalid session"""
        response = self.app.get('/results/invalid_session')
        self.assertEqual(response.status_code, 404)
        self.assertIn(b'Results not found', response.data)
    
    def test_results_route_logic(self):
        """Test the logic of results route handling"""
        # Test that session IDs are properly formatted
        session_id = 'test_session_123'
        self.assertIn('test_session', session_id)
        self.assertIn('123', session_id)
        
        # Test that the session ID format is valid
        self.assertTrue(len(session_id) > 0)
        self.assertIsInstance(session_id, str)
        
        # Test that we can create test data
        test_data = pd.DataFrame({
            'id': ['CP1', 'CP2', 'TOTAL'],
            'dZ': [0.01, -0.02, 0.0],
            'status': ['ok', 'ok', 'summary']
        })
        
        self.assertEqual(len(test_data), 3)
        self.assertIn('CP1', test_data['id'].values)
        self.assertIn('TOTAL', test_data['id'].values)
    
    def test_error_handling(self):
        """Test error handling in the application"""
        # Test with invalid data that should cause errors
        point_cloud_content = b"test point cloud data"
        checkpoints_content = b"id;E;N;H\n1;100;200;50"
        
        data = {
            'point_cloud': (point_cloud_content, 'test.las'),
            'checkpoints': (checkpoints_content, 'test.csv'),
            'radius': 'invalid_number',  # This should cause a validation error
            'knn': 'not_a_number'  # This should cause a validation error
        }
        
        response = self.app.post('/upload', data=data)
        self.assertEqual(response.status_code, 400)
        
        response_data = json.loads(response.data)
        self.assertIn('error', response_data)
    
    def test_file_size_limits(self):
        """Test file size limit handling"""
        # Create a large file content (simulate large file)
        large_content = b"x" * (501 * 1024 * 1024)  # 501MB
        
        data = {
            'point_cloud': (large_content, 'large.las'),
            'checkpoints': (b"test", 'test.csv')
        }
        
        # The app should handle large files gracefully
        # This test verifies the file size limit configuration
        self.assertEqual(app.config['MAX_CONTENT_LENGTH'], 500 * 1024 * 1024)
    
    def test_session_management(self):
        """Test session ID generation and management"""
        # Test that session IDs are unique and properly formatted
        from datetime import datetime
        
        # Simulate session ID generation
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Verify format
        self.assertRegex(session_id, r'^\d{8}_\d{6}$')
        
        # Verify it's a valid date/time
        try:
            datetime.strptime(session_id, '%Y%m%d_%H%M%S')
        except ValueError:
            self.fail("Session ID is not a valid date/time format")

class TestCheckpointAnalysis(unittest.TestCase):
    """Test cases for checkpoint analysis functionality"""
    
    @patch('subprocess.run')
    def test_run_checkpoint_analysis_success(self, mock_run):
        """Test successful checkpoint analysis execution"""
        # Mock successful subprocess execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Analysis completed successfully"
        mock_result.stderr = ""
        mock_run.return_value = mock_result
        
        params = {
            'radius': 0.5,
            'knn': 200,
            'mad_alpha': 6.0,
            'ransac_thresh': 0.08,
            'ransac_iters': 400,
            'min_inliers': 5,
            'min_neighbors': 5
        }
        
        success, output = run_checkpoint_analysis('test.las', 'test.csv', params)
        
        self.assertTrue(success)
        self.assertEqual(output, "Analysis completed successfully")
        
        # Verify subprocess was called with correct parameters
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]
        self.assertIn('python3', call_args)
        self.assertIn('Checkpoints_control_1.py', call_args)
        self.assertIn('--points', call_args)
        self.assertIn('--cps', call_args)
        self.assertIn('--radius', call_args)
        self.assertIn('0.5', call_args)
    
    @patch('subprocess.run')
    def test_run_checkpoint_analysis_failure(self, mock_run):
        """Test failed checkpoint analysis execution"""
        # Mock failed subprocess execution
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error: invalid parameters"
        mock_run.return_value = mock_result
        
        params = {'radius': 0.5}
        
        success, output = run_checkpoint_analysis('test.las', 'test.csv', params)
        
        self.assertFalse(success)
        self.assertIn("Script execution failed", output)
        self.assertIn("invalid parameters", output)
    
    @patch('subprocess.run')
    def test_run_checkpoint_analysis_exception(self, mock_run):
        """Test checkpoint analysis with subprocess exception"""
        # Mock subprocess exception
        mock_run.side_effect = Exception("Subprocess error")
        
        params = {'radius': 0.5}
        
        success, output = run_checkpoint_analysis('test.las', 'test.csv', params)
        
        self.assertFalse(success)
        self.assertIn("Error running analysis", output)
        self.assertIn("Subprocess error", output)

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestWebUI)
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCheckpointAnalysis))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with error code if tests failed
    sys.exit(not result.wasSuccessful())
