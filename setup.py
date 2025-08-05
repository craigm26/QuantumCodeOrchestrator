#!/usr/bin/env python3
"""
Setup script for Quantum-Enhanced Multi-Agent VS Code Extension
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_prerequisites():
    """Check if required tools are installed"""
    print("🔍 Checking prerequisites...")
    
    # Check Node.js
    if not run_command("node --version", "Checking Node.js"):
        print("❌ Node.js is required. Please install Node.js 16+ from https://nodejs.org/")
        return False
    
    # Check npm
    if not run_command("npm --version", "Checking npm"):
        print("❌ npm is required. Please install npm")
        return False
    
    # Check Python
    if not run_command("python --version", "Checking Python"):
        print("❌ Python is required. Please install Python 3.8+")
        return False
    
    # Check pip
    if not run_command("pip --version", "Checking pip"):
        print("❌ pip is required. Please install pip")
        return False
    
    print("✅ All prerequisites are satisfied")
    return True

def install_node_dependencies():
    """Install Node.js dependencies"""
    return run_command("npm install", "Installing Node.js dependencies")

def install_python_dependencies():
    """Install Python dependencies"""
    return run_command("pip install -r python/requirements.txt", "Installing Python dependencies")

def compile_extension():
    """Compile the TypeScript extension"""
    return run_command("npm run compile", "Compiling TypeScript extension")

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    
    directories = [
        "models",
        "vector_db",
        "out",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def create_config():
    """Create default configuration"""
    print("⚙️ Creating configuration...")
    
    config = {
        "quantumAI": {
            "modelPath": "./models/10b-model",
            "quantumEnabled": True,
            "vectorDBPath": "./vector_db",
            "maxContextLength": 8192,
            "agentTemperature": {
                "researcher": 0.7,
                "engineer": 0.3,
                "analyst": 0.5
            }
        }
    }
    
    with open("config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ Created config.json")

def create_model_placeholder():
    """Create a placeholder for the 10B model"""
    print("🤖 Creating model placeholder...")
    
    model_info = {
        "name": "10B Parameter Model",
        "description": "Placeholder for the 10B parameter language model",
        "status": "not_installed",
        "instructions": [
            "Download your preferred 10B parameter model",
            "Place it in the ./models/10b-model directory",
            "Update the modelPath in config.json if needed"
        ]
    }
    
    model_dir = Path("models/10b-model")
    model_dir.mkdir(exist_ok=True)
    
    with open(model_dir / "model_info.json", "w") as f:
        json.dump(model_info, f, indent=2)
    
    print("✅ Created model placeholder")

def main():
    """Main setup function"""
    print("🚀 Setting up Quantum-Enhanced Multi-Agent VS Code Extension")
    print("=" * 60)
    
    # Check prerequisites
    if not check_prerequisites():
        print("❌ Setup failed due to missing prerequisites")
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_node_dependencies():
        print("❌ Failed to install Node.js dependencies")
        sys.exit(1)
    
    if not install_python_dependencies():
        print("❌ Failed to install Python dependencies")
        sys.exit(1)
    
    # Compile extension
    if not compile_extension():
        print("❌ Failed to compile extension")
        sys.exit(1)
    
    # Create configuration
    create_config()
    create_model_placeholder()
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Download a 10B parameter model to ./models/10b-model/")
    print("2. Start the Python backend: python python/quantum_ai_backend.py")
    print("3. Press F5 in VS Code to run the extension")
    print("4. Use Ctrl+Shift+A to analyze code with quantum AI")
    print("\n📚 For more information, see README.md")
    print("=" * 60)

if __name__ == "__main__":
    main() 