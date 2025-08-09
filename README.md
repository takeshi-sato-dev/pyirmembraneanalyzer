# PyIRMembraneAnalyzer

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.16758554.svg)](https://doi.org/10.5281/zenodo.16758554)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Automated analysis of polarized ATR-FTIR spectra for membrane protein orientation determination.

## Overview

PyIRMembraneAnalyzer is a Python tool for analyzing polarized attenuated total reflection Fourier transform infrared (ATR-FTIR) spectra to determine the orientation of membrane proteins. The software automates the entire analysis pipeline from raw spectra to helix tilt angles.

### Key Features

- 📊 **Automated Spectral Analysis**: Baseline correction, smoothing, and peak deconvolution
- 🔄 **Smart Model Selection**: Automatic selection of optimal peak numbers (5, 6, or 7) using AIC/BIC
- 📐 **Dichroic Ratio Calculation**: Automated calculation from perpendicular/parallel spectra  
- 🎯 **Tilt Angle Determination**: Convert dichroic ratios to transmembrane helix orientations
- 📈 **Publication-Ready Output**: High-quality figures (PNG/SVG) and detailed CSV reports
- ⚡ **Batch Processing**: Analyze multiple spectra simultaneously

## Installation

### Requirements

- Python 3.10 or higher
- NumPy >= 1.19.0
- SciPy >= 1.5.0
- pandas >= 1.1.0
- matplotlib >= 3.3.0
- scikit-learn >= 0.23.0

### Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/pyirmembraneanalyzer.git
cd pyirmembraneanalyzer

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .