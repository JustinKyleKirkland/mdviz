# Installation Instructions for mdviz

## Quick Installation

```bash
# Install from PyPI (when available)
pip install mdviz

# Or install from source
git clone https://github.com/JustinKyleKirkland/mdviz.git
cd mdviz
pip install -e .
```

## Development Installation

```bash
# Clone the repository
git clone https://github.com/JustinKyleKirkland/mdviz.git
cd mdviz

# Run the development setup script
python setup_dev.py

# Or install manually
pip install -e ".[dev]"
```

## Test Installation

```bash
# Run the demo
mdviz-demo

# Or run it manually
python -m mdviz.examples.demo
```

## Dependencies

### Core Dependencies
- Python 3.8+
- MDAnalysis >= 2.4.0
- scikit-learn >= 1.3.0
- numpy >= 1.21.0
- pandas >= 1.5.0
- plotly >= 5.0.0
- matplotlib >= 3.5.0
- py3Dmol >= 2.0.0
- seaborn >= 0.11.0
- scipy >= 1.9.0

### Optional Dependencies (for development)
- pytest >= 7.0.0
- pytest-cov >= 4.0.0
- black >= 22.0.0
- isort >= 5.10.0
- flake8 >= 5.0.0
- jupyter >= 1.0.0
- ipywidgets >= 8.0.0

## Troubleshooting

### MDAnalysis Installation Issues
MDAnalysis has some system dependencies. If you encounter issues:

```bash
# On macOS
brew install netcdf hdf5

# On Ubuntu/Debian
sudo apt-get install libnetcdf-dev libhdf5-dev

# Then reinstall
pip install --no-cache-dir MDAnalysis
```

### py3Dmol Issues
If py3Dmol fails to install:

```bash
pip install --upgrade py3Dmol
```

## Verification

After installation, verify everything works:

```python
import mdviz
print(f"mdviz version: {mdviz.__version__}")

# Test basic functionality
from mdviz import TrajectoryAnalyzer, ClusterAnalyzer, Visualizer
print("✅ All core classes imported successfully!")
```
