#!/usr/bin/env python3
"""
Setup script for LM Studio integration with Quantum Code Orchestrator
"""

import subprocess
import sys
import os
import json
from pathlib import Path

def check_lm_studio():
    """Check if LM Studio is running"""
    try:
        import requests
        response = requests.get("http://localhost:1234/v1/models", timeout=5)
        if response.status_code == 200:
            print("✅ LM Studio is running and accessible")
            return True
        else:
            print("❌ LM Studio is running but API returned error")
            return False
    except Exception as e:
        print(f"❌ LM Studio is not running or not accessible: {e}")
        return False

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing dependencies...")
    
    # Install aiohttp for LM Studio API
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "aiohttp"], check=True)
        print("✅ aiohttp installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install aiohttp: {e}")
        return False
    
    # Install other requirements
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements_web.txt"], check=True)
        print("✅ All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def create_lmstudio_config():
    """Create LM Studio specific configuration"""
    config = {
        "model_path": "./models/10b-model",
        "db_path": "./vector_db",
        "monitor_paths": ["./src", "./python", "./test_example.py"],
        "port": 8000,
        "host": "127.0.0.1",
        "log_level": "INFO",
        "quantum_enabled": True,
        "max_context_length": 8192,
        "agent_temperatures": {
            "researcher": 0.7,
            "engineer": 0.3,
            "analyst": 0.5
        },
        "file_extensions": [".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".h"],
        "debounce_delay": 1.0,
        "update_interval": 1.0,
        "max_events": 1000,
        "chart_update_interval": 10,
        "lm_studio": {
            "enabled": True,
            "api_url": "http://localhost:1234/v1",
            "api_key": "not-needed",
            "model_name": "gpt-oss-20b",
            "max_tokens": 2048,
            "timeout": 30
        }
    }
    
    with open("config_lmstudio.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("✅ LM Studio configuration created: config_lmstudio.json")

def test_lmstudio_connection():
    """Test connection to LM Studio"""
    print("🔍 Testing LM Studio connection...")
    
    if check_lm_studio():
        print("✅ LM Studio connection test successful")
        return True
    else:
        print("❌ LM Studio connection test failed")
        print("\n📋 To start LM Studio:")
        print("1. Download and install LM Studio from https://lmstudio.ai/")
        print("2. Load your gpt-oss-20b model")
        print("3. Start the local server (usually on port 1234)")
        print("4. Run this setup script again")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Quantum Code Orchestrator with LM Studio")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Create LM Studio configuration
    create_lmstudio_config()
    
    # Test LM Studio connection
    if not test_lmstudio_connection():
        print("\n⚠️  LM Studio is not running. You can still run the system in demo mode.")
        print("   To use full functionality, start LM Studio and run this setup again.")
    
    print("\n🎉 Setup completed!")
    print("\n📋 Next steps:")
    print("1. Start LM Studio and load your model")
    print("2. Run the CLI: python quantum_code_orchestrator_lmstudio.py --cli")
    print("3. Or run the web server: python quantum_code_orchestrator_lmstudio.py --port 8000")
    print("4. Open http://localhost:8000 in your browser")
    
    return True

if __name__ == "__main__":
    main() 