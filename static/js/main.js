// SkyMap Checkpoint Control - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the application
    initializeApp();
});

function initializeApp() {
    // Set up event listeners
    setupEventListeners();
    
    // Initialize form validation
    setupFormValidation();
    
    // Set up adaptive radius toggle
    setupAdaptiveRadiusToggle();
    
    // Set up tooltips
    setupTooltips();
}

function setupEventListeners() {
    // Run analysis button
    const runButton = document.getElementById('submitBtn');
    if (runButton) {
        runButton.addEventListener('click', handleRunAnalysis);
    }
    
    // File input change events
    const pointCloudInput = document.getElementById('pointCloud');
    const checkpointsInput = document.getElementById('checkpoints');
    
    if (pointCloudInput) {
        pointCloudInput.addEventListener('change', handleFileSelection);
    }
    
    if (checkpointsInput) {
        checkpointsInput.addEventListener('change', handleFileSelection);
    }
    
    // Form validation on input
    const formInputs = document.querySelectorAll('input, select');
    formInputs.forEach(input => {
        input.addEventListener('input', validateForm);
        input.addEventListener('change', validateForm);
    });
}

function setupAdaptiveRadiusToggle() {
    const adaptiveRadiusCheckbox = document.getElementById('adaptive_radius');
    const adaptiveRadiusParams = document.getElementById('adaptiveRadiusParams');
    
    if (adaptiveRadiusCheckbox && adaptiveRadiusParams) {
        adaptiveRadiusCheckbox.addEventListener('change', function() {
            if (this.checked) {
                adaptiveRadiusParams.style.display = 'block';
                // Disable fixed radius when adaptive is enabled
                const radiusInput = document.getElementById('radius');
                if (radiusInput) {
                    radiusInput.disabled = true;
                    radiusInput.value = '';
                }
            } else {
                adaptiveRadiusParams.style.display = 'none';
                // Re-enable fixed radius when adaptive is disabled
                const radiusInput = document.getElementById('radius');
                if (radiusInput) {
                    radiusInput.disabled = false;
                }
            }
        });
    }
}

function setupFormValidation() {
    // Validate numeric inputs
    const numericInputs = document.querySelectorAll('input[type="number"]');
    numericInputs.forEach(input => {
        input.addEventListener('blur', function() {
            validateNumericInput(this);
        });
    });
}

function setupTooltips() {
    // Initialize Bootstrap tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

function handleFileSelection(event) {
    const file = event.target.files[0];
    if (file) {
        // Show file info
        const fileInfo = document.createElement('div');
        fileInfo.className = 'alert alert-info mt-2';
        fileInfo.innerHTML = `
            <i class="fas fa-file me-2"></i>
            <strong>${file.name}</strong> (${formatFileSize(file.size)})
        `;
        
        // Remove previous file info
        const existingInfo = event.target.parentNode.querySelector('.alert');
        if (existingInfo) {
            existingInfo.remove();
        }
        
        event.target.parentNode.appendChild(fileInfo);
        
        // Validate file type
        validateFileType(file, event.target);
    }
}

function validateFileType(file, inputElement) {
    const allowedExtensions = {
        'pointCloud': ['.las', '.laz', '.csv', '.tsv', '.txt'],
        'checkpoints': ['.csv', '.tsv', '.txt']
    };
    
    const inputType = inputElement.id;
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    
    if (!allowedExtensions[inputType].includes(fileExtension)) {
        showError(`Invalid file type for ${inputType.replace('_', ' ')}. Allowed: ${allowedExtensions[inputType].join(', ')}`);
        inputElement.value = '';
        return false;
    }
    
    return true;
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function validateNumericInput(input) {
    const value = input.value;
    const min = parseFloat(input.min);
    const max = parseFloat(input.max);
    
    if (value !== '' && !isNaN(value)) {
        const numValue = parseFloat(value);
        
        // Check if integer fields have whole numbers
        const integerFields = ['knn', 'ransac_iters', 'min_inliers', 'min_neighbors'];
        if (integerFields.includes(input.id) && !Number.isInteger(numValue)) {
            showInputError(input, 'This field requires a whole number');
            return false;
        }
        
        if (min !== NaN && numValue < min) {
            showInputError(input, `Value must be at least ${min}`);
            return false;
        }
        
        if (max !== NaN && numValue > max) {
            showInputError(input, `Value must be at most ${max}`);
            return false;
        }
        
        clearInputError(input);
        return true;
    }
    
    return true;
}

function showInputError(input, message) {
    clearInputError(input);
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'invalid-feedback d-block';
    errorDiv.textContent = message;
    
    input.classList.add('is-invalid');
    input.parentNode.appendChild(errorDiv);
}

function clearInputError(input) {
    input.classList.remove('is-invalid');
    const errorDiv = input.parentNode.querySelector('.invalid-feedback');
    if (errorDiv) {
        errorDiv.remove();
    }
}

function validateForm() {
    const runButton = document.getElementById('submitBtn');
    const pointCloudInput = document.getElementById('pointCloud');
    const checkpointsInput = document.getElementById('checkpoints');
    
    let isValid = true;
    
    // Check if files are selected
    if (!pointCloudInput || !pointCloudInput.files[0]) {
        isValid = false;
    }
    
    if (!checkpointsInput || !checkpointsInput.files[0]) {
        isValid = false;
    }
    
    // Validate numeric inputs
    const numericInputs = document.querySelectorAll('input[type="number"]');
    numericInputs.forEach(input => {
        if (!validateNumericInput(input)) {
            isValid = false;
        }
    });
    
    // Enable/disable run button
    if (runButton) {
        runButton.disabled = !isValid;
        runButton.classList.toggle('btn-secondary', !isValid);
        runButton.classList.toggle('btn-primary', isValid);
    }
    
    return isValid;
}

function handleRunAnalysis() {
    if (!validateForm()) {
        showError('Please fix the form errors before running analysis.');
        return;
    }
    
    // Show progress section
    showProgress();
    
    // Prepare form data
    const formData = new FormData();
    
    // Add files
    const pointCloudFile = document.getElementById('pointCloud').files[0];
    const checkpointsFile = document.getElementById('checkpoints').files[0];
    
    formData.append('point_cloud', pointCloudFile);
    formData.append('checkpoints', checkpointsFile);
    
    // Add all form parameters
    const paramsForm = document.getElementById('uploadForm');
    const formInputs = paramsForm.querySelectorAll('input, select');
    
    formInputs.forEach(input => {
        if (input.type === 'checkbox') {
            formData.append(input.name, input.checked ? 'on' : '');
        } else if (input.type === 'radio') {
            if (input.checked) {
                formData.append(input.name, input.value);
            }
        } else {
            if (input.value !== '') {
                // Ensure integer fields are sent as whole numbers
                const integerFields = ['knn', 'ransac_iters', 'min_inliers', 'min_neighbors'];
                if (integerFields.includes(input.id)) {
                    formData.append(input.name, Math.round(parseFloat(input.value)));
                } else {
                    formData.append(input.name, input.value);
                }
            }
        }
    });
    
    // Send request
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            showResults(data);
        } else {
            showError(data.error || 'An unknown error occurred.');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showError('Network error occurred. Please try again.');
    });
}

function showProgress() {
    hideAllSections();
    document.getElementById('progressSection').style.display = 'block';
}

function showResults(data) {
    hideAllSections();
    
    const resultsSection = document.getElementById('resultsSection');
    const resultsContent = document.getElementById('resultsContent');
    
    // Create results content
    resultsContent.innerHTML = `
        <div class="alert alert-success">
            <h5><i class="fas fa-check-circle me-2"></i>Analysis completed successfully!</h5>
            <p class="mb-2"><strong>Session ID:</strong> ${data.session_id}</p>
            <p class="mb-0"><strong>Output Files:</strong> ${data.output_files.join(', ')}</p>
        </div>
        
        <div class="mb-3">
            <h6>Analysis Output:</h6>
            <pre class="bg-light p-3 rounded" style="white-space: pre-wrap; font-family: monospace; max-height: 300px; overflow-y: auto;">${data.output}</pre>
        </div>
        
        <div class="text-center">
            <a href="/results/${data.session_id}" class="btn btn-success btn-lg">
                <i class="fas fa-chart-bar me-2"></i>View Detailed Results
            </a>
        </div>
    `;
    
    resultsSection.style.display = 'block';
}

function showError(message) {
    hideAllSections();
    
    const errorSection = document.getElementById('errorSection');
    const errorContent = document.getElementById('errorContent');
    
    errorContent.innerHTML = `
        <div class="alert alert-danger">
            <h5><i class="fas fa-exclamation-triangle me-2"></i>Error</h5>
            <p class="mb-0">${message}</p>
        </div>
        
        <div class="text-center">
            <button class="btn btn-primary" onclick="hideAllSections()">
                <i class="fas fa-times me-2"></i>Close
            </button>
        </div>
    `;
    
    errorSection.style.display = 'block';
}

function hideAllSections() {
    const sections = ['progressSection', 'resultsSection', 'errorSection'];
    sections.forEach(sectionId => {
        const section = document.getElementById(sectionId);
        if (section) {
            section.style.display = 'none';
        }
    });
}

// Utility functions
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

// Export functions for debugging
window.skymapApp = {
    validateForm,
    showNotification,
    showError,
    showResults
};
