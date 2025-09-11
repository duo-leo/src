#!/usr/bin/env python3
"""
SemiAutoFixer Setup Script

This script helps you set up the SemiAutoFixer environment quickly.
Run this script to install dependencies and prepare the workspace.

Usage:
    python setup.py
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, but you have Python {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor} is compatible!")
    return True

def install_requirements():
    """Install required packages"""
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found!")
        return False
    
    return run_command(
        f'"{sys.executable}" -m pip install -r requirements.txt',
        "Installing required packages"
    )

def setup_directories():
    """Set up necessary directories"""
    directories = [
        "WORKING_FOLDER",
        "WORKING_FOLDER/NewProject"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    return True

def verify_installation():
    """Verify that the installation was successful"""
    try:
        # Test core imports
        import pandas
        import torch
        import transformers
        import tqdm
        import underthesea
        
        print("✅ Core packages imported successfully!")
        
        # Test SemiAutoFixer imports
        sys.path.insert(0, '.')
        from utils import AutoFixer, Data
        
        print("✅ SemiAutoFixer modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False

def main():
    """Main setup function"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    SemiAutoFixer Setup                       ║
    ║                                                              ║
    ║  This script will help you set up the SemiAutoFixer tool    ║
    ║  with all necessary dependencies and configuration.          ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Step 1: Check Python version
    print("Step 1: Checking Python version...")
    if not check_python_version():
        return False
    
    # Step 2: Install requirements
    print("\nStep 2: Installing requirements...")
    if not install_requirements():
        print("❌ Failed to install requirements. Please check the error messages above.")
        return False
    
    # Step 3: Set up directories
    print("\nStep 3: Setting up directories...")
    if not setup_directories():
        return False
    
    # Step 4: Verify installation
    print("\nStep 4: Verifying installation...")
    if not verify_installation():
        print("❌ Installation verification failed. Please check the error messages above.")
        return False
    
    # Success message
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    Setup Completed! 🎉                      ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Next steps:
    
    1. Run the usage guide:
       python how_to_use.py
    
    2. Prepare your data:
       - Place input files in WORKING_FOLDER/YourProject/
       - Supported formats: text files or Excel
    
    3. Configure the tool:
       - Edit runner.py to set DIR_PATH to your project
       - Optionally configure LLM settings in config_template.py
    
    4. Run the processing:
       python runner.py
    
    For detailed usage instructions, run:
       python how_to_use.py
    
    Happy text processing! 🚀
    """)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
