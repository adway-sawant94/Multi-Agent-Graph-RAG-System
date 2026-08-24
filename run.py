import os
import sys
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def install_dependencies():
    """Checks and installs missing dependencies from requirements.txt."""
    print("Checking project dependencies...")
    try:
        import streamlit
        import networkx
        import numpy
        import dotenv
        import pyvis
        print("All dependencies are already installed.")
    except ImportError:
        print("Some dependencies are missing. Installing from requirements.txt...")
        req_file = PROJECT_ROOT / "requirements.txt"
        if req_file.exists():
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
                    check=True
                )
                print("Dependencies installed successfully.")
            except subprocess.CalledProcessError as e:
                print(f"Error installing dependencies: {e}")
                print("Please install requirements manually using: pip install -r requirements.txt")
        else:
            print("requirements.txt not found. Cannot auto-install dependencies.")

def launch_streamlit():
    """Launches the Streamlit visual dashboard."""
    app_file = PROJECT_ROOT / "app.py"
    if not app_file.exists():
        print("Error: app.py not found.")
        sys.exit(1)
        
    print("Launching Streamlit dashboard...")
    try:
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(app_file)],
            check=True
        )
    except KeyboardInterrupt:
        print("\nDashboard shut down.")
    except Exception as e:
        print(f"Failed to launch Streamlit: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python run.py web       - Start the Streamlit web dashboard")
        print("  python run.py cli ...   - Run CLI command (e.g., python run.py cli index)")
        sys.exit(1)
        
    mode = sys.argv[1].lower()
    
    if mode == "web":
        install_dependencies()
        launch_streamlit()
    elif mode == "cli":
        # Pass remaining arguments to cli.py
        cli_file = PROJECT_ROOT / "cli.py"
        if not cli_file.exists():
            print("Error: cli.py not found.")
            sys.exit(1)
            
        cmd = [sys.executable, str(cli_file)] + sys.argv[2:]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            sys.exit(e.returncode)
    else:
        print(f"Unknown command: {mode}")
        print("Use 'web' to run the Streamlit app or 'cli' to run the command-line interface.")
        sys.exit(1)

if __name__ == "__main__":
    main()
