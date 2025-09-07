#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Web UI for Checkpoints Control Script
# Author: ChatGPT for Jon Bengtsson (SkyMap)

import os
import tempfile
import shutil
from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for
from werkzeug.utils import secure_filename
import subprocess
import json
import pandas as pd
from datetime import datetime
import zipfile

app = Flask(__name__)
app.secret_key = 'skymap_checkpoints_2025'
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024 * 1024  # 3GB max file size

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'las', 'laz', 'csv', 'tsv', 'txt'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_checkpoint_analysis(point_cloud_path, cps_path, params):
    """Run the checkpoint analysis script with given parameters"""
    try:
        # Build command
        cmd = [
            'python3', 'Checkpoints_control_1.py',
            '--points', point_cloud_path,
            '--cps', cps_path
        ]
        
        # Add parameters
        if params.get('radius'):
            cmd.extend(['--radius', str(params['radius'])])
        if params.get('knn'):
            cmd.extend(['--knn', str(params['knn'])])
        if params.get('mad_alpha'):
            cmd.extend(['--mad_alpha', str(params['mad_alpha'])])
        if params.get('ransac_thresh'):
            cmd.extend(['--ransac_thresh', str(params['ransac_thresh'])])
        if params.get('ransac_iters'):
            cmd.extend(['--ransac_iters', str(params['ransac_iters'])])
        if params.get('min_inliers'):
            cmd.extend(['--min_inliers', str(params['min_inliers'])])
        if params.get('min_neighbors'):
            cmd.extend(['--min_neighbors', str(params['min_neighbors'])])
        if params.get('las_keep_class'):
            cmd.extend(['--las-keep-class', str(params['las_keep_class'])])
        if params.get('las_keep_returns'):
            cmd.extend(['--las-keep-returns', str(params['las_keep_returns'])])
        if params.get('xy_maxdist'):
            cmd.extend(['--xy-maxdist', str(params['xy_maxdist'])])
        if params.get('z_window'):
            cmd.extend(['--z-window', str(params['z_window'])])
        if params.get('adaptive_radius'):
            cmd.append('--adaptive-radius')
        if params.get('adaptive_radius_start'):
            cmd.extend(['--adaptive-radius-start', str(params['adaptive_radius_start'])])
        if params.get('adaptive_radius_step'):
            cmd.extend(['--adaptive-radius-step', str(params['adaptive_radius_step'])])
        if params.get('adaptive_radius_max'):
            cmd.extend(['--adaptive-radius-max', str(params['adaptive_radius_max'])])
        if params.get('centroid_max_offset'):
            cmd.extend(['--centroid-max-offset', str(params['centroid_max_offset'])])
        if params.get('tolerance'):
            cmd.extend(['--tolerance', str(params['tolerance'])])
        if params.get('export_points'):
            cmd.append('--export-points')
        if params.get('export_stage'):
            cmd.extend(['--export-stage', params['export_stage']])
        if params.get('cps_sep'):
            cmd.extend(['--cps-sep', params['cps_sep']])
        if params.get('decimal'):
            cmd.extend(['--decimal', params['decimal']])
        
        # Run command
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        if result.returncode != 0:
            return False, f"Script execution failed: {result.stderr}"
        
        return True, result.stdout
        
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Analysis error: {str(e)}")
        print(f"Traceback: {error_details}")
        return False, f"Error running analysis: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'point_cloud' not in request.files or 'checkpoints' not in request.files:
        return jsonify({'error': 'Both point cloud and checkpoints files are required'}), 400
    
    point_cloud_file = request.files['point_cloud']
    checkpoints_file = request.files['checkpoints']
    
    if point_cloud_file.filename == '' or checkpoints_file.filename == '':
        return jsonify({'error': 'Both files must be selected'}), 400
    
    if not (allowed_file(point_cloud_file.filename) and allowed_file(checkpoints_file.filename)):
        return jsonify({'error': 'Invalid file type. Allowed: LAS, LAZ, CSV, TSV, TXT'}), 400
    
    try:
        # Create unique session directory
        session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_dir = os.path.join(UPLOAD_FOLDER, session_id)
        os.makedirs(session_dir, exist_ok=True)
        
        # Save uploaded files
        point_cloud_path = os.path.join(session_dir, secure_filename(point_cloud_file.filename))
        checkpoints_path = os.path.join(session_dir, secure_filename(checkpoints_file.filename))
        
        point_cloud_file.save(point_cloud_path)
        checkpoints_file.save(checkpoints_path)
        
        # Get parameters from form
        params = {
            'radius': request.form.get('radius'),
            'knn': request.form.get('knn'),
            'mad_alpha': request.form.get('mad_alpha'),
            'ransac_thresh': request.form.get('ransac_thresh'),
            'ransac_iters': request.form.get('ransac_iters'),
            'min_inliers': request.form.get('min_inliers'),
            'min_neighbors': request.form.get('min_neighbors'),
            'las_keep_class': request.form.get('las_keep_class'),
            'las_keep_returns': request.form.get('las_keep_returns'),
            'xy_maxdist': request.form.get('xy_maxdist'),
            'z_window': request.form.get('z_window'),
            'adaptive_radius': request.form.get('adaptive_radius') == 'on',
            'adaptive_radius_start': request.form.get('adaptive_radius_start'),
            'adaptive_radius_step': request.form.get('adaptive_radius_step'),
            'adaptive_radius_max': request.form.get('adaptive_radius_max'),
            'centroid_max_offset': request.form.get('centroid_max_offset'),
            'tolerance': request.form.get('tolerance'),
            'export_points': request.form.get('export_points') == 'on',
            'export_stage': request.form.get('export_stage'),
            'cps_sep': request.form.get('cps_sep'),
            'decimal': request.form.get('decimal')
        }
        
        # Convert empty strings to None for numeric parameters
        for key in ['radius', 'knn', 'mad_alpha', 'ransac_thresh', 'ransac_iters', 
                   'min_inliers', 'min_neighbors', 'xy_maxdist', 'z_window',
                   'adaptive_radius_start', 'adaptive_radius_step', 'adaptive_radius_max',
                   'centroid_max_offset', 'tolerance']:
            if params[key] == '':
                params[key] = None
            elif params[key] is not None:
                try:
                    # Integer parameters
                    if key in ['knn', 'ransac_iters', 'min_inliers', 'min_neighbors']:
                        params[key] = int(float(params[key]))
                    # Float parameters
                    else:
                        params[key] = float(params[key])
                except ValueError:
                    return jsonify({'error': f'Invalid value for {key}'}), 400
        
        # Run analysis
        success, output = run_checkpoint_analysis(point_cloud_path, checkpoints_path, params)
        
        if success:
            # Copy results to results folder
            results_dir = os.path.join(RESULTS_FOLDER, session_id)
            os.makedirs(results_dir, exist_ok=True)
            
            # Look for output files in the session directory
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
        import traceback
        error_details = traceback.format_exc()
        print(f"Upload error: {str(e)}")
        print(f"Traceback: {error_details}")
        return jsonify({'error': f'Upload failed: {str(e)}', 'details': error_details}), 500

@app.route('/download/<session_id>/<filename>')
def download_file(session_id, filename):
    file_path = os.path.join(RESULTS_FOLDER, session_id, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return jsonify({'error': 'File not found'}), 404

@app.route('/results/<session_id>')
def view_results(session_id):
    results_dir = os.path.join(RESULTS_FOLDER, session_id)
    if not os.path.exists(results_dir):
        return "Results not found", 404
    
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
