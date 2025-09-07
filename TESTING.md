# 🧪 Testing Guide - SkyMap Checkpoint Control Web UI

This document provides comprehensive information about testing the web UI application, including how to run tests, what they cover, and how to contribute to testing.

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Test Structure](#test-structure)
- [Running Tests](#running-tests)
- [Test Coverage](#test-coverage)
- [Continuous Integration](#continuous-integration)
- [Writing Tests](#writing-tests)
- [Troubleshooting](#troubleshooting)

## 🚀 Quick Start

### Run All Tests
```bash
python run_tests.py
```

### Run Tests with Coverage
```bash
python run_tests.py --coverage
```

### Run Specific Test Module
```bash
python run_tests.py --test test_web_ui
```

## 🏗️ Test Structure

```
tests/
├── __init__.py              # Package initialization
├── test_web_ui.py          # Main web UI tests
└── conftest.py             # Test configuration (future)

run_tests.py                 # Test runner script
requirements_test.txt        # Testing dependencies
.github/workflows/test.yml   # CI/CD pipeline
```

### Test Categories

1. **Unit Tests** (`test_web_ui.py`)
   - File validation
   - Parameter handling
   - Route functionality
   - Error handling
   - Session management

2. **Integration Tests**
   - File upload workflow
   - Analysis execution
   - Results processing

3. **API Tests**
   - REST endpoint validation
   - Response format checking
   - Error code verification

## 🏃‍♂️ Running Tests

### Prerequisites
```bash
# Install testing dependencies
pip install -r requirements_test.txt

# Or install individually
pip install pytest pytest-cov coverage
```

### Test Commands

#### Basic Test Execution
```bash
# Run all tests
python run_tests.py

# Run with verbose output
python -m unittest discover tests -v

# Run specific test file
python -m unittest tests.test_web_ui

# Run specific test method
python -m unittest tests.test_web_ui.TestWebUI.test_index_route
```

#### Advanced Test Execution
```bash
# Run with pytest (if installed)
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=web_ui --cov-report=html

# Run specific test categories
pytest tests/ -k "upload"  # Only upload-related tests
pytest tests/ -k "route"   # Only route-related tests
```

#### Using the Test Runner Script
```bash
# Run all tests
python run_tests.py

# Run specific test module
python run_tests.py --test test_web_ui

# Run with coverage
python run_tests.py --coverage

# Show help
python run_tests.py --help
```

## 📊 Test Coverage

### Current Coverage Areas

- ✅ **File Upload Validation**
  - File type checking
  - File size limits
  - Required file presence

- ✅ **Parameter Processing**
  - Integer/float conversion
  - Parameter validation
  - Default value handling

- ✅ **Route Functionality**
  - Index page rendering
  - Upload endpoint
  - Results display
  - File download

- ✅ **Error Handling**
  - Invalid input handling
  - File processing errors
  - Analysis failures

- ✅ **Session Management**
  - Unique session IDs
  - File organization
  - Cleanup procedures

### Coverage Goals

- **Target**: 90%+ code coverage
- **Critical Paths**: 100% coverage
- **Error Handling**: 100% coverage
- **User Input**: 100% coverage

## 🔄 Continuous Integration

### GitHub Actions

The project includes automated testing via GitHub Actions:

- **Triggered on**: Push to main/develop, Pull Requests
- **Python Versions**: 3.9, 3.10, 3.11, 3.12
- **Test Matrix**: Multiple Python versions
- **Quality Checks**: Linting, security scanning
- **Coverage Reports**: Automated coverage analysis

### CI Pipeline Steps

1. **Setup**: Python environment, dependencies
2. **Testing**: Run all tests across Python versions
3. **Coverage**: Generate and upload coverage reports
4. **Linting**: Code style and quality checks
5. **Security**: Vulnerability scanning
6. **Artifacts**: Upload test results and reports

## ✍️ Writing Tests

### Test Naming Conventions

```python
def test_functionality_description(self):
    """Test description of what is being tested"""
    # Arrange
    # Act
    # Assert
```

### Test Structure (AAA Pattern)

```python
def test_upload_success(self):
    """Test successful file upload and analysis"""
    # Arrange - Set up test data and mocks
    mock_run_analysis.return_value = (True, "Success")
    test_data = {...}
    
    # Act - Execute the function being tested
    response = self.app.post('/upload', data=test_data)
    
    # Assert - Verify the expected outcomes
    self.assertEqual(response.status_code, 200)
    self.assertIn('success', response.json)
```

### Mocking Best Practices

```python
@patch('web_ui.run_checkpoint_analysis')
def test_analysis_execution(self, mock_analysis):
    """Test checkpoint analysis execution"""
    # Setup mock
    mock_analysis.return_value = (True, "Success")
    
    # Test the function
    result = some_function()
    
    # Verify mock was called correctly
    mock_analysis.assert_called_once_with(expected_args)
```

### Test Data Management

```python
def setUp(self):
    """Set up test environment before each test"""
    # Create temporary directories
    self.test_upload_dir = tempfile.mkdtemp()
    self.test_results_dir = tempfile.mkdtemp()
    
    # Mock global directories
    with patch('web_ui.UPLOAD_FOLDER', self.test_upload_dir):
        with patch('web_ui.RESULTS_FOLDER', self.test_results_dir):
            self.app = app.test_client()
            self.app.testing = True

def tearDown(self):
    """Clean up after each test"""
    # Remove temporary directories
    shutil.rmtree(self.test_upload_dir, ignore_errors=True)
    shutil.rmtree(self.test_results_dir, ignore_errors=True)
```

## 🐛 Troubleshooting

### Common Test Issues

#### Import Errors
```bash
# Problem: Module not found
ModuleNotFoundError: No module named 'web_ui'

# Solution: Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:/path/to/project"
# Or use the test runner script
python run_tests.py
```

#### Permission Errors
```bash
# Problem: Cannot create temporary directories
PermissionError: [Errno 13] Permission denied

# Solution: Check directory permissions
chmod 755 tests/
chmod 755 uploads/
chmod 755 results/
```

#### Mock Issues
```bash
# Problem: Mock not working as expected
AssertionError: Expected 'mock_function' to have been called

# Solution: Check mock setup and assertions
@patch('module.function_name')  # Use full import path
def test_function(self, mock_function):
    mock_function.return_value = expected_value
    # ... test code ...
    mock_function.assert_called_once()
```

### Debug Mode

Run tests with debug output:
```bash
# Verbose unittest output
python -m unittest discover tests -v

# Pytest debug output
pytest tests/ -v -s --tb=long

# Coverage with detailed output
python run_tests.py --coverage
```

### Test Isolation

Ensure tests don't interfere with each other:
```python
def setUp(self):
    """Fresh environment for each test"""
    # Use unique temporary directories
    self.test_dir = tempfile.mkdtemp(prefix=f"test_{self._testMethodName}_")
    
def tearDown(self):
    """Clean up after each test"""
    # Always clean up, even if test fails
    shutil.rmtree(self.test_dir, ignore_errors=True)
```

## 📈 Performance Testing

### Load Testing (Future)

```python
# Example load test structure
def test_concurrent_uploads(self):
    """Test multiple simultaneous file uploads"""
    import threading
    import time
    
    def upload_file(file_id):
        # Simulate file upload
        pass
    
    # Create multiple threads
    threads = [threading.Thread(target=upload_file, args=(i,)) 
               for i in range(10)]
    
    # Start all threads
    start_time = time.time()
    for thread in threads:
        thread.start()
    
    # Wait for completion
    for thread in threads:
        thread.join()
    
    # Verify performance
    elapsed_time = time.time() - start_time
    self.assertLess(elapsed_time, 30.0)  # Should complete within 30 seconds
```

## 🔒 Security Testing

### Input Validation Tests

```python
def test_sql_injection_prevention(self):
    """Test SQL injection prevention"""
    malicious_input = "'; DROP TABLE users; --"
    
    response = self.app.post('/upload', data={
        'point_cloud': (b"test", 'test.las'),
        'checkpoints': (b"test", 'test.csv'),
        'parameter': malicious_input
    })
    
    # Should not crash or expose sensitive data
    self.assertNotIn('error', response.data.decode())
```

### File Upload Security

```python
def test_malicious_file_upload(self):
    """Test malicious file upload prevention"""
    # Create file with suspicious content
    malicious_content = b"<script>alert('xss')</script>"
    
    response = self.app.post('/upload', data={
        'point_cloud': (malicious_content, 'test.las'),
        'checkpoints': (b"test", 'test.csv')
    })
    
    # Should handle malicious content gracefully
    self.assertNotEqual(response.status_code, 500)
```

## 📝 Contributing to Tests

### Adding New Tests

1. **Identify the functionality** to test
2. **Create test method** following naming conventions
3. **Use appropriate assertions** for the test type
4. **Add proper cleanup** in tearDown if needed
5. **Document the test** with clear docstrings

### Test Review Checklist

- [ ] Test name clearly describes what is tested
- [ ] Test follows AAA pattern (Arrange, Act, Assert)
- [ ] Proper mocking is used for external dependencies
- [ ] Test data is isolated and cleaned up
- [ ] Edge cases and error conditions are covered
- [ ] Test passes consistently (no flaky tests)

### Running Tests Before Committing

```bash
# Quick test run
python run_tests.py

# Full test suite with coverage
python run_tests.py --coverage

# Specific test category
python -m pytest tests/ -k "upload" -v
```

## 📚 Additional Resources

- [Python unittest documentation](https://docs.python.org/3/library/unittest.html)
- [Pytest documentation](https://docs.pytest.org/)
- [Coverage.py documentation](https://coverage.readthedocs.io/)
- [Flask testing guide](https://flask.palletsprojects.com/en/2.3.x/testing/)

---

**Remember**: Good tests are the foundation of reliable software. Write tests that are clear, comprehensive, and maintainable! 🧪✨
