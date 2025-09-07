# 🚀 Quick Start - SkyMap Checkpoint Control Web UI

## ⚡ Get Started in 3 Steps

### 1. **Run the Startup Script**
```bash
python3 start_web_ui.py
```

This script will:
- ✅ Check Python version
- ✅ Install missing dependencies automatically
- ✅ Verify all required files are present
- ✅ Start the web server
- ✅ Open your browser automatically

### 2. **Access the Web Interface**
- **URL**: http://localhost:5001
- **Browser**: Any modern web browser (Chrome, Firefox, Safari, Edge)

### 3. **Start Using the Tool**
- Upload your point cloud file (LAS/LAZ/CSV)
- Upload your checkpoints file (CSV)
- Configure analysis parameters
- Click "Run Analysis"

## 🎯 What You'll See

- **Professional SkyMap Branding** with official logo
- **Intuitive File Upload** interface
- **Comprehensive Parameter Configuration** for all analysis options
- **Real-time Validation** and error checking
- **Progress Tracking** during analysis
- **Beautiful Results Display** with download options

## 🔧 If Something Goes Wrong

### **Dependencies Missing**
```bash
pip3 install flask pandas numpy
```

### **Manual Start**
```bash
python3 web_ui.py
```
**Note**: The web UI now runs on port 5001 to avoid conflicts with macOS AirPlay.

### **Check Files**
Ensure these files exist:
- `Checkpoints_control_1.py` (original script)
- `web_ui.py` (web application)
- `templates/` folder
- `static/` folder
- `static/images/skymap_logo.bmp` (SkyMap logo)

## 📱 Features

- **Responsive Design** - works on all devices
- **File Type Support** - LAS, LAZ, CSV, TSV, TXT
- **Parameter Control** - all analysis options configurable
- **Session Management** - unique IDs for each analysis
- **Export Options** - download results and point data
- **Professional UI** - SkyMap-branded interface

## 🌐 Access from Other Devices

The web UI runs on `0.0.0.0:5001`, so you can access it from:
- **Local**: http://localhost:5001
- **Network**: http://YOUR_IP_ADDRESS:5001
- **Other devices** on the same network

## 🛑 Stopping the Server

Press `Ctrl+C` in the terminal where the web server is running.

---

**Need Help?** Check `README_WEB_UI.md` for detailed documentation.
