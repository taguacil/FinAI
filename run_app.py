#!/usr/bin/env python3
"""
Simple script to run the Portfolio Tracker Streamlit app.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the Streamlit app with proper configuration."""
    
    # Ensure we're in the right directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    
    # Add src to Python path
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    # Set environment variable for Python path
    env = os.environ.copy()
    env['PYTHONPATH'] = str(src_path)
    
    # Streamlit command
    cmd = [
        sys.executable, "-m", "streamlit", "run", 
        "src/ui/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "localhost",
        "--server.headless", "false",
        "--browser.gatherUsageStats", "false"
    ]
    
    print("🚀 Starting AI Portfolio Tracker...")
    print("📊 Open your browser to: http://localhost:8501")
    print("🔄 Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        print("\n👋 Goodbye! Portfolio Tracker stopped.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting app: {e}")
        print("💡 Try: uv sync (to install dependencies)")
    except FileNotFoundError:
        print("❌ Streamlit not found. Installing dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"], check=True)
        print("✅ Streamlit installed. Please run the script again.")

if __name__ == "__main__":
    main()