#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Startup script for SkyMap Checkpoint Control Secure Web UI with Authentication
# Author: ChatGPT for Jon Bengtsson (SkyMap)

import os
import sys
import subprocess
import webbrowser
import time

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['flask', 'pandas', 'numpy', 'laspy', 'scipy', 'flask_limiter', 'werkzeug']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("✅ Dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("❌ Failed to install dependencies. Please run manually:")
            print(f"   pip install {' '.join(missing_packages)}")
            return False
    
    return True

def check_files():
    """Check if required files exist"""
    required_files = [
        'Checkpoints_control_1.py',
        'web_ui_secure_auth.py',
        'templates/login.html',
        'templates/index_auth.html',
        'static/css/style.css',
        'static/js/main.js',
        'static/images/skymap_logo.bmp'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found!")
    return True

def start_web_ui():
    """Start the secure web UI with authentication"""
    print("\n🚀 Starting SkyMap Checkpoint Control Secure Web UI...")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check files
    if not check_files():
        return False
    
    print("\n✅ All checks passed! Starting secure web server...")
    print("\n📋 Web UI Information:")
    print("   • URL: http://localhost:5001")
    print("   • Port: 5001")
    print("   • Authentication: REQUIRED")
    print("   • Username: Jonb_skymap")
    print("   • Password: SkyMap2015")
    print("   • Press Ctrl+C to stop the server")
    print("\n🔐 Security Features:")
    print("   • Rate limiting enabled")
    print("   • File validation & sanitization")
    print("   • Command injection protection")
    print("   • Session timeout (2 hours)")
    print("   • Secure file handling")
    print("\n🌐 Opening web browser...")
    
    # Open web browser after a short delay
    def open_browser():
        time.sleep(2)
        try:
            webbrowser.open('http://localhost:5001')
        except:
            print("⚠️  Could not open browser automatically. Please navigate to: http://localhost:5001")
    
    import threading
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start the Flask application
    try:
        from web_ui_secure_auth import app
        app.run(debug=True, host='0.0.0.0', port=5001)
    except KeyboardInterrupt:
        print("\n\n🛑 Web server stopped by user.")
    except Exception as e:
        print(f"\n❌ Error starting web server: {e}")
        return False
    
    return True

def main():
    """Main function"""
    print("🎯 SkyMap Checkpoint Control - Secure Web UI with Authentication")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"   Current version: {sys.version}")
        return
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Start the web UI
    if start_web_ui():
        print("\n✅ Secure Web UI started successfully!")
    else:
        print("\n❌ Failed to start Secure Web UI. Please check the error messages above.")

if __name__ == '__main__':
    main()
