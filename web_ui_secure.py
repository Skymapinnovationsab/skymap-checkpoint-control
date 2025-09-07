#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SECURE Web UI for Checkpoints Control Script
# Author: ChatGPT for Jon Bengtsson (SkyMap)
# Security improvements implemented

import os
import tempfile
import shutil
import re
import secrets
import hashlib
import time
from flask import Flask, render_template, request, jsonify, send_file, abort
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.utils import secure_filename
import subprocess
import json
import pandas as pd
from datetime import datetime
import zipfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# SECURITY: Generate secure secret key
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# SECURITY: Reduce file size limit for production
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# SECURITY: Rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'las', 'laz', 'csv', 'tsv', 'txt'}

# SECURITY: File size limits per type
FILE_SIZE_LIMITS = {
    'las': 50 * 1024 * 1024,  # 50MB
    'laz': 50 * 1024 * 1024,  # 50MB
    'csv': 10 * 1024 * 1024,  # 10MB
    'tsv': 10 * 1024 * 1024,  # 10MB
    'txt': 10 * 1024 * 1024,  # 10MB
}

# SECURITY: Parameter validation ranges
PARAM_RANGES = {
    'radius': (0.01, 10.0),
    'knn': (1, 1000),
    'mad_alpha': (1.0, 20.0),
    'ransac_thresh': (0.001, 1.0),
    'ransac_iters': (10, 10000),
    'min_inliers': (1, 100),
    'min_neighbors': (1, 100),
    'xy_maxdist': (0.01, 100.0),
    'z_window': (0.01, 100.0),
    'adaptive_radius_start': (0.01, 10.0),
    'adaptive_radius_step': (0.001, 1.0),
    'adaptive_radius_max': (0.01, 10.0),
    'centroid_max_offset': (0.001, 10.0),
    'tolerance': (0.001, 1.0),
}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def validate_filename(filename):
    """SECURITY: Validate filename for security"""
    if not filename:
        return False
    
    # Check for dangerous characters
    dangerous_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
    for char in dangerous_chars:
        if char in filename:
            return False
    
    # Check length
    if len(filename) > 255:
        return False
    
    return True

def allowed_file(filename):
    """SECURITY: Enhanced file validation"""
    if not validate_filename(filename):
        return False
    
    if '.' not in filename:
        return False
    
    extension = filename.rsplit('.', 1)[1].lower()
    return extension in ALLOWED_EXTENSIONS

def validate_file_size(file, extension):
    """SECURITY: Validate file size"""
    if extension not in FILE_SIZE_LIMITS:
        return False
    
    # Get file size
    file.seek(0, 2)  # Seek to end
    size = file.tell()
    file.seek(0)  # Reset to beginning
    
    return size <= FILE_SIZE_LIMITS[extension]

def validate_parameter(param_name, value):
    """SECURITY: Validate parameter values"""
    if param_name not in PARAM_RANGES:
        return False
    
    try:
        num_value = float(value)
        min_val, max_val = PARAM_RANGES[param_name]
        return min_val <= num_value <= max_val
    except (ValueError, TypeError):
        return False

def sanitize_path(path):
    """SECURITY: Sanitize file paths"""
    # Remove any path traversal attempts
    path = os.path.normpath(path)
    if '..' in path or path.startswith('/'):
        raise ValueError("Invalid path")
    return path

def run_checkpoint_analysis(point_cloud_path, cps_path, params):
    """SECURITY: Run the checkpoint analysis script with security measures"""
    try:
        # SECURITY: Validate file paths
        point_cloud_path = sanitize_path(point_cloud_path)
        cps_path = sanitize_path(cps_path)
        
        # SECURITY: Validate files exist and are readable
        if not os.path.exists(point_cloud_path) or not os.path.exists(cps_path):
            return False, "File not found"
        
        if not os.access(point_cloud_path, os.R_OK) or not os.access(cps_path, os.R_OK):
            return False, "File not readable"
        
        # SECURITY: Build command with validated parameters
        cmd = [
            'python3', 'Checkpoints_control_1.py',
            '--points', point_cloud_path,
            '--cps', cps_path
        ]
        
        # SECURITY: Add parameters with validation
        for param_name, value in params.items():
            if value is not None and value != '':
                if not validate_parameter(param_name, value):
                    return False, f"Invalid parameter value: {param_name}"
                
                # Map parameter names to command line arguments
                param_mapping = {
                    'radius': '--radius',
                    'knn': '--knn',
                    'mad_alpha': '--mad_alpha',
                    'ransac_thresh': '--ransac_thresh',
                    'ransac_iters': '--ransac_iters',
                    'min_inliers': '--min_inliers',
                    'min_neighbors': '--min_neighbors',
                    'las_keep_class': '--las-keep-class',
                    'las_keep_returns': '--las-keep-returns',
                    'xy_maxdist': '--xy-maxdist',
                    'z_window': '--z-window',
                    'adaptive_radius_start': '--adaptive-radius-start',
                    'adaptive_radius_step': '--adaptive-radius-step',
                    'adaptive_radius_max': '--adaptive-radius-max',
                    'centroid_max_offset': '--centroid-max-offset',
                    'tolerance': '--tolerance',
                    'export_stage': '--export-stage',
                    'cps_sep': '--cps-sep',
                    'decimal': '--decimal'
                }
                
                if param_name in param_mapping:
                    cmd.extend([param_mapping[param_name], str(value)])
                elif param_name == 'adaptive_radius' and value:
                    cmd.append('--adaptive-radius')
                elif param_name == 'export_points' and value:
                    cmd.append('--export-points')
        
        # SECURITY: Log command for monitoring (without sensitive data)
        logger.info(f"Running analysis with {len(cmd)} parameters")
        
        # SECURITY: Run command with timeout and restricted environment
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=os.getcwd(),
            timeout=300,  # 5 minute timeout
            env={**os.environ, 'PYTHONPATH': ''}  # Clear PYTHONPATH for security
        )
        
        if result.returncode != 0:
            logger.error(f"Analysis failed with return code {result.returncode}")
            return False, "Analysis failed"
        
        return True, result.stdout
        
    except subprocess.TimeoutExpired:
        logger.error("Analysis timed out")
        return False, "Analysis timed out"
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return False, "Analysis error"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@limiter.limit("5 per minute")  # SECURITY: Rate limiting
def upload_files():
    try:
        # SECURITY: Check if files are present
        if 'point_cloud' not in request.files or 'checkpoints' not in request.files:
            return jsonify({'error': 'Both files are required'}), 400
        
        point_cloud_file = request.files['point_cloud']
        checkpoints_file = request.files['checkpoints']
        
        # SECURITY: Validate files
        if point_cloud_file.filename == '' or checkpoints_file.filename == '':
            return jsonify({'error': 'Both files must be selected'}), 400
        
        if not (allowed_file(point_cloud_file.filename) and allowed_file(checkpoints_file.filename)):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # SECURITY: Validate file sizes
        pc_ext = point_cloud_file.filename.rsplit('.', 1)[1].lower()
        cp_ext = checkpoints_file.filename.rsplit('.', 1)[1].lower()
        
        if not validate_file_size(point_cloud_file, pc_ext):
            return jsonify({'error': 'Point cloud file too large'}), 400
        
        if not validate_file_size(checkpoints_file, cp_ext):
            return jsonify({'error': 'Checkpoints file too large'}), 400
        
        # SECURITY: Create secure session directory
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + secrets.token_hex(8)
        session_dir = os.path.join(UPLOAD_FOLDER, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # SECURITY: Save files with secure names
        pc_filename = secure_filename(point_cloud_file.filename)
        cp_filename = secure_filename(checkpoints_file.filename)
        
        point_cloud_path = os.path.join(session_dir, pc_filename)
        checkpoints_path = os.path.join(session_dir, cp_filename)
        
        point_cloud_file.save(point_cloud_path)
        checkpoints_file.save(checkpoints_path)
        
        # SECURITY: Validate and sanitize parameters
        params = {}
        for key in ['radius', 'knn', 'mad_alpha', 'ransac_thresh', 'ransac_iters', 
                   'min_inliers', 'min_neighbors', 'xy_maxdist', 'z_window',
                   'adaptive_radius_start', 'adaptive_radius_step', 'adaptive_radius_max',
                   'centroid_max_offset', 'tolerance']:
            value = request.form.get(key)
            if value and value.strip():
                if not validate_parameter(key, value):
                    return jsonify({'error': f'Invalid parameter: {key}'}), 400
                try:
                    if key in ['knn', 'ransac_iters', 'min_inliers', 'min_neighbors']:
                        params[key] = int(float(value))
                    else:
                        params[key] = float(value)
                except ValueError:
                    return jsonify({'error': f'Invalid parameter value: {key}'}), 400
        
        # Handle boolean and string parameters
        params['adaptive_radius'] = request.form.get('adaptive_radius') == 'on'
        params['export_points'] = request.form.get('export_points') == 'on'
        
        # SECURITY: Validate string parameters
        export_stage = request.form.get('export_stage')
        if export_stage and export_stage not in ['all', 'filtered', 'final']:
            return jsonify({'error': 'Invalid export stage'}), 400
        params['export_stage'] = export_stage
        
        cps_sep = request.form.get('cps_sep')
        if cps_sep and cps_sep not in [',', ';', '\t']:
            return jsonify({'error': 'Invalid CSV separator'}), 400
        params['cps_sep'] = cps_sep
        
        decimal = request.form.get('decimal')
        if decimal and decimal not in ['.', ',']:
            return jsonify({'error': 'Invalid decimal separator'}), 400
        params['decimal'] = decimal
        
        # Run analysis
        success, output = run_checkpoint_analysis(point_cloud_path, checkpoints_path, params)
        
        if success:
            # Copy results to results folder
            results_dir = os.path.join(RESULTS_FOLDER, session_id)
            os.makedirs(results_dir, exist_ok=True)
            
            # Look for output files
            output_files = []
            for file in os.listdir(session_dir):
                if file.startswith('cp_check_'):
                    src = os.path.join(session_dir, file)
                    dst = os.path.join(results_dir, file)
                    shutil.copy2(src, dst)
                    output_files.append(file)
            
            return jsonify({
                'success': True,
                'session_id': session_id,
                'output': output,
                'output_files': output_files
            })
        else:
            return jsonify({'error': output}), 500
            
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': 'Upload failed'}), 500

@app.route('/download/<session_id>/<filename>')
@limiter.limit("10 per minute")  # SECURITY: Rate limiting
def download_file(session_id, filename):
    try:
        # SECURITY: Validate session_id and filename
        if not re.match(r'^[a-zA-Z0-9_]+$', session_id):
            abort(400)
        
        if not validate_filename(filename):
            abort(400)
        
        file_path = os.path.join(RESULTS_FOLDER, session_id, filename)
        
        # SECURITY: Validate file path
        file_path = sanitize_path(file_path)
        
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            abort(404)
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        abort(500)

@app.route('/results/<session_id>')
@limiter.limit("20 per minute")  # SECURITY: Rate limiting
def view_results(session_id):
    try:
        # SECURITY: Validate session_id
        if not re.match(r'^[a-zA-Z0-9_]+$', session_id):
            abort(400)
        
        results_dir = os.path.join(RESULTS_FOLDER, session_id)
        results_dir = sanitize_path(results_dir)
        
        if not os.path.exists(results_dir):
            abort(404)
        
        # Read summary file
        summary_path = os.path.join(results_dir, 'cp_check_results_summary.txt')
        summary = ""
        if os.path.exists(summary_path):
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = f.read()
        
        # Read CSV results
        csv_path = os.path.join(results_dir, 'cp_check_results.csv')
        results_data = None
        if os.path.exists(csv_path):
            try:
                results_data = pd.read_csv(csv_path)
            except:
                pass
        
        return render_template('results.html', 
                             session_id=session_id, 
                             summary=summary, 
                             results_data=results_data)
    except Exception as e:
        logger.error(f"Results view error: {str(e)}")
        abort(500)

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({'error': 'Rate limit exceeded'}), 429

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=False)
