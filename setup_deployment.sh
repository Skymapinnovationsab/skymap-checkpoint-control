#!/bin/bash
# SkyMap Checkpoint Control - Deployment Setup Script
# This script helps you prepare your project for online deployment

echo "🎯 SkyMap Checkpoint Control - Deployment Setup"
echo "=============================================="

# Check if git is installed
if ! command -v git &> /dev/null; then
    echo "❌ Git is not installed. Please install Git first."
    echo "   Download from: https://git-scm.com/downloads"
    exit 1
fi

echo "✅ Git is installed"

# Check if we're in a git repository
if [ ! -d ".git" ]; then
    echo "📁 Initializing Git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Add all files to git
echo "📝 Adding files to Git..."
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "ℹ️  No changes to commit"
else
    echo "💾 Committing changes..."
    git commit -m "Initial commit - SkyMap Checkpoint Control with Authentication"
    echo "✅ Changes committed"
fi

echo ""
echo "🚀 Next Steps:"
echo "=============="
echo "1. Create a GitHub repository:"
echo "   - Go to https://github.com/Skymapinnovationsab"
echo "   - Click 'New repository'"
echo "   - Name: skymap-checkpoint-control"
echo "   - Make it PUBLIC"
echo "   - Don't initialize with README"
echo ""
echo "2. Push to GitHub:"
echo "   git remote add origin https://github.com/Skymapinnovationsab/skymap-checkpoint-control.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Deploy on Railway:"
echo "   - Go to https://railway.app"
echo "   - Sign up with GitHub"
echo "   - Create new project from your repository"
echo "   - Set environment variables (see DEPLOYMENT_GUIDE.md)"
echo ""
echo "📖 For detailed instructions, see DEPLOYMENT_GUIDE.md"
echo ""
echo "🎉 Your secure checkpoint analysis system is ready for deployment!"
