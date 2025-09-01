#!/usr/bin/env python3
"""
Setup script for mdviz package development and testing.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"   Command: {cmd}")
        print(f"   Error: {e.stderr}")
        sys.exit(1)


def main():
    """Main setup function."""
    print("🧬 mdviz Package Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("pyproject.toml").exists():
        print("❌ Error: pyproject.toml not found. Please run this script from the package root directory.")
        sys.exit(1)
    
    # Clean previous builds
    print("🧹 Cleaning previous builds...")
    for path in ["build", "dist", "*.egg-info"]:
        if Path(path).exists():
            run_command(f"rm -rf {path}", f"Removing {path}")
    
    # Install/upgrade build tools
    run_command(
        f"{sys.executable} -m pip install --upgrade pip build wheel setuptools",
        "Installing/upgrading build tools"
    )
    
    # Build the package
    run_command(
        f"{sys.executable} -m build",
        "Building package"
    )
    
    # Install in development mode
    run_command(
        f"{sys.executable} -m pip install -e .",
        "Installing package in development mode"
    )
    
    # Test the installation
    print("\n🧪 Testing installation...")
    try:
        import mdviz
        print(f"✅ mdviz version {mdviz.__version__} imported successfully!")
        
        # Test command-line tool
        result = subprocess.run([sys.executable, "-m", "mdviz.examples.demo"], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Command-line demo works!")
        else:
            print("⚠️  Command-line demo had issues:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Installation test failed: {e}")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("  • Test with your own data")
    print("  • Run the examples in the examples/ directory")
    print("  • Check out the tutorial notebook")
    print("\nTo build for distribution:")
    print(f"  {sys.executable} -m build")
    print("  twine upload dist/*")


if __name__ == "__main__":
    main()
