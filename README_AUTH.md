# SkyMap Checkpoint Control - Secure Web UI with Authentication

## 🔐 Authentication Features

This version includes comprehensive security features and basic authentication to protect your checkpoint analysis system.

### Login Credentials
- **Username:** `Jonb_skymap`
- **Password:** `SkyMap2015`

### Security Features
- ✅ **Basic Authentication** - Username/password required
- ✅ **Session Management** - 2-hour session timeout
- ✅ **Rate Limiting** - Protection against brute force attacks
- ✅ **File Validation** - Secure file upload handling
- ✅ **Command Injection Protection** - Safe subprocess execution
- ✅ **Input Validation** - All parameters validated
- ✅ **Secure Error Handling** - No system information disclosure
- ✅ **File Size Limits** - 2GB for point clouds, 100MB for CSV files

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_secure.txt
```

### 2. Start the Secure Server
```bash
python start_secure_auth.py
```

### 3. Access the Application
- Open your browser to: http://localhost:5001
- Login with your credentials:
  - Username: `Jonb_skymap`
  - Password: `SkyMap2015`
- Upload your files and start analysis

## 🔧 Configuration

### Environment Variables
Set these environment variables for production:

```bash
# Generate a secure secret key
export SECRET_KEY="your-super-secret-key-here"

# Change default credentials
export AUTH_USERNAME="your-username"
export AUTH_PASSWORD="your-secure-password"

# Set Flask environment
export FLASK_ENV="production"
```

### Customizing Credentials
You can change the default credentials by:

1. **Environment Variables** (Recommended for production):
   ```bash
   export AUTH_USERNAME="your-username"
   export AUTH_PASSWORD="your-secure-password"
   ```

2. **Direct Code Modification** (Not recommended for production):
   Edit `web_ui_secure_auth.py` lines 60-61:
   ```python
   DEFAULT_USERNAME = "your-username"
   DEFAULT_PASSWORD = "your-secure-password"
   ```

## 🛡️ Security Best Practices

### For Production Deployment:

1. **Change Default Credentials**
   ```bash
   export AUTH_USERNAME="admin"
   export AUTH_PASSWORD="$(openssl rand -base64 32)"
   ```

2. **Use HTTPS**
   - Deploy behind a reverse proxy (nginx/Apache)
   - Use SSL certificates
   - Force HTTPS redirects

3. **Monitor Access**
   - Check logs for failed login attempts
   - Monitor file uploads
   - Set up alerts for suspicious activity

4. **Regular Updates**
   - Keep dependencies updated
   - Monitor security advisories
   - Regular security audits

## 📁 File Structure

```
├── web_ui_secure_auth.py      # Main secure application with auth
├── start_secure_auth.py       # Startup script
├── requirements_secure.txt    # Dependencies
├── Procfile                   # Production deployment config
├── templates/
│   ├── login.html            # Login page
│   └── index_auth.html       # Main interface with auth
├── static/                   # CSS, JS, images
└── README_AUTH.md           # This file
```

## 🔄 Migration from Non-Authenticated Version

If you're upgrading from the non-authenticated version:

1. **Backup your data**
2. **Update templates**: Use `index_auth.html` instead of `index.html`
3. **Update startup script**: Use `start_secure_auth.py`
4. **Set environment variables** for credentials
5. **Test thoroughly** before going live

## 🚨 Important Security Notes

- **Default credentials are for development only**
- **Always change credentials in production**
- **Use strong passwords** (minimum 12 characters)
- **Enable HTTPS** for production deployments
- **Monitor access logs** regularly
- **Keep the system updated**

## 📞 Support

For security-related questions or issues:
- Check the logs for error messages
- Verify environment variables are set correctly
- Ensure all dependencies are installed
- Contact SkyMap support for assistance

---

**© 2025 SkyMap Innovations AB - Secure Surveying Solutions**
