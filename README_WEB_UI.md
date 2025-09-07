# SkyMap Checkpoint Control - Web UI

A modern, user-friendly web interface for the SkyMap Checkpoint Control quality assessment tool. This web application provides an intuitive way to upload point cloud files and checkpoint data, configure analysis parameters, and view results through a professional, SkyMap-branded interface.

## Features

- **Professional SkyMap Branding**: Incorporates the official SkyMap logo and design elements
- **File Upload Interface**: Drag-and-drop or click-to-upload for point cloud and checkpoint files
- **Comprehensive Parameter Configuration**: All analysis parameters are configurable through the UI
- **Real-time Validation**: Form validation with helpful error messages
- **Progress Tracking**: Visual feedback during analysis processing
- **Results Visualization**: Clean, organized display of analysis results
- **Download Capabilities**: Easy access to all output files
- **Responsive Design**: Works on desktop, tablet, and mobile devices
- **Session Management**: Unique session IDs for each analysis run

## Supported File Formats

### Point Cloud Files
- LAS (.las)
- LAZ (.laz) 
- CSV (.csv)
- TSV (.tsv)
- TXT (.txt)

### Checkpoint Files
- CSV (.csv)
- TSV (.tsv)
- TXT (.txt)

## Installation

### Prerequisites
- Python 3.8 or higher
- The original `Checkpoints_control_1.py` script must be in the same directory
- Required Python packages (see requirements_web.txt)

### Setup Steps

1. **Clone or download the project files**
   ```bash
   # Ensure you have the following files in your directory:
   # - web_ui.py
   # - Checkpoints_control_1.py
   # - templates/ (folder)
   # - static/ (folder)
   # - requirements_web.txt
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements_web.txt
   ```

3. **Verify the SkyMap logo is present**
   ```bash
   # The logo should be at: static/images/skymap_logo.bmp
   # If not present, copy it from the original location:
   cp "/path/to/SkyMap_mail.bmp" static/images/skymap_logo.bmp
   ```

4. **Run the web application**
   ```bash
   python web_ui.py
   ```

5. **Access the web interface**
   - Open your web browser
   - Navigate to: `http://localhost:5001`
   - The interface should display with SkyMap branding

## Usage

### 1. File Upload
- **Point Cloud File**: Select your LAS/LAZ point cloud or CSV point data
- **Checkpoints File**: Select your CSV file containing checkpoint coordinates
- **CSV Format Options**: Configure separator and decimal character if needed

### 2. Parameter Configuration

#### Neighborhood Selection
- **Fixed Radius**: Set a specific search radius in meters (leave empty for kNN)
- **k-Nearest Neighbors**: Number of nearest points to consider
- **Adaptive Radius**: Automatically adjust radius to find sufficient neighbors

#### Robust Filtering
- **MAD Alpha**: Outlier detection threshold (default: 6.0)
- **RANSAC Threshold**: Plane fitting tolerance in meters (default: 0.08)
- **RANSAC Iterations**: Number of RANSAC iterations (default: 400)
- **Min Inliers**: Minimum number of inlier points required (default: 5)

#### Quality Thresholds
- **Min Neighbors**: Minimum neighbors required for analysis (default: 5)
- **Height Tolerance**: Acceptable height deviation in meters (default: 0.025)
- **Centroid Max Offset**: Maximum allowed centroid offset (default: 0.15)

#### Selection Filters
- **Max XY Distance**: Hard limit on XY distance from checkpoint
- **Z Window**: Height window around checkpoint elevation

#### LAS/LAZ Filters
- **Keep Classes**: Comma-separated list of LAS classes to keep
- **Keep Returns**: Comma-separated list of return numbers to keep

#### Export Options
- **Export Point Cloud Points**: Enable to export used points
- **Export Stage**: Choose which processing stage to export

### 3. Running Analysis
- Click the "Run Analysis" button
- Monitor progress with the loading spinner
- View results summary when complete
- Access detailed results and download files

### 4. Results
- **Summary**: Overview of analysis statistics
- **Detailed Table**: Per-checkpoint results with status indicators
- **Download Links**: Access to all output files
- **Quick Stats**: Visual summary cards

## File Structure

```
project_directory/
├── web_ui.py                 # Main Flask application
├── Checkpoints_control_1.py  # Original analysis script
├── requirements_web.txt      # Python dependencies
├── README_WEB_UI.md         # This file
├── templates/               # HTML templates
│   ├── index.html          # Main interface
│   └── results.html        # Results display
├── static/                 # Static assets
│   ├── css/
│   │   └── style.css      # Custom styles
│   ├── js/
│   │   └── main.js        # JavaScript functionality
│   └── images/
│       └── skymap_logo.bmp # SkyMap logo
├── uploads/                # Temporary upload storage
└── results/                # Analysis results storage
```

## Configuration

### Customizing the Interface
- **Logo**: Replace `static/images/skymap_logo.bmp` with your preferred logo
- **Colors**: Modify CSS variables in `static/css/style.css`
- **Branding**: Update text and styling in HTML templates

### Server Configuration
- **Port**: Change port in `web_ui.py` (default: 5001)
- **Host**: Modify host binding for network access
- **File Size Limits**: Adjust `MAX_CONTENT_LENGTH` for larger files

## Troubleshooting

### Common Issues

1. **Logo not displaying**
   - Ensure the SkyMap logo file exists at `static/images/skymap_logo.bmp`
   - Check file permissions and format

2. **Analysis fails to run**
   - Verify `Checkpoints_control_1.py` is in the same directory
   - Check Python dependencies are installed
   - Review console output for error messages

3. **File upload issues**
   - Check file format is supported
   - Verify file size is within limits (500MB default)
   - Ensure proper file permissions

4. **Parameter validation errors**
   - Check numeric input ranges
   - Verify required fields are filled
   - Review parameter dependencies

### Debug Mode
The application runs in debug mode by default. For production:
```python
# In web_ui.py, change:
app.run(debug=False, host='0.0.0.0', port=5000)
```

## Security Considerations

- **File Uploads**: Only allow trusted file types
- **Input Validation**: All parameters are validated server-side
- **Session Management**: Unique session IDs prevent conflicts
- **File Cleanup**: Temporary files are managed automatically

## Performance

- **File Processing**: Large files may take several minutes
- **Memory Usage**: Point clouds are loaded into memory
- **Concurrent Users**: Single-threaded processing (consider scaling for production)

## Support

For technical support or questions about the web interface:
- Review the console output for error messages
- Check the original `Checkpoints_control_1.py` script functionality
- Verify all dependencies are properly installed

## License

This web interface is part of the SkyMap Checkpoint Control tool suite.
© 2025 SkyMap Innovations. Professional surveying solutions.
