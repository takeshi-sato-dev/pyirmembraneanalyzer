#!/bin/bash

echo "=========================================="
echo "PyIRMembraneAnalyzer - Cleanup & Organize"
echo "=========================================="

# 1. Remove unnecessary files
echo ""
echo "Removing unnecessary files..."

rm -f all_files.txt
rm -f templated.txt
rm -f pyirmembraneanalyzer___init__.py
rm -f tests_test_analyzer.py
rm -f examples_basic_usage.py
rm -f examples_advanced_usage.py
rm -f CHANGE.md
rm -f contribution.md
rm -f README
rm -f license.txt
rm -f zenodo.json
rm -f Makefile
rm -f pytest.ini
rm -f pyproject.toml
rm -f MANIFEST.in
rm -f test_quick.py
rm -rf docs
rm -rf pyirmembraneanalyzer

echo "✓ Cleaned up files"

# 2. Rename files
echo ""
echo "Renaming files..."

if [ -f "manuscript.md" ]; then
    mv manuscript.md paper.md
    echo "✓ Renamed manuscript.md to paper.md"
fi

if [ -f "LICENSE" ]; then
    echo "✓ LICENSE exists"
else
    echo "⚠ LICENSE missing"
fi

# 3. Organize test files
echo ""
echo "Organizing test files..."

mkdir -p tests
mkdir -p examples/demo_data
mkdir -p results

# 4. Move demo data
echo ""
echo "Moving demo data..."

if [ -f "example_0.csv" ]; then
    cp example_0.csv examples/demo_data/sample_0.csv
    rm -f example_0.csv
fi

if [ -f "example_90.csv" ]; then
    cp example_90.csv examples/demo_data/sample_90.csv
    rm -f example_90.csv
fi

if [ -f "demo_0.csv" ]; then
    cp demo_0.csv examples/demo_data/sample_0.csv
    rm -f demo_0.csv
fi

if [ -f "demo_90.csv" ]; then
    cp demo_90.csv examples/demo_data/sample_90.csv
    rm -f demo_90.csv
fi

echo "✓ Organized demo data"

# 5. Check required files
echo ""
echo "=========================================="
echo "Checking required files for JOSS..."
echo "=========================================="

if [ -f "ir_membrane_analyzer.py" ]; then
    echo "✓ ir_membrane_analyzer.py"
else
    echo "✗ ir_membrane_analyzer.py (MISSING)"
fi

if [ -f "paper.md" ]; then
    echo "✓ paper.md"
else
    echo "✗ paper.md (MISSING)"
fi

if [ -f "paper.bib" ]; then
    echo "✓ paper.bib"
else
    echo "✗ paper.bib (MISSING)"
fi

if [ -f "README.md" ]; then
    echo "✓ README.md"
else
    echo "✗ README.md (MISSING)"
fi

if [ -f "LICENSE" ]; then
    echo "✓ LICENSE"
else
    echo "✗ LICENSE (MISSING)"
fi

if [ -f "setup.py" ]; then
    echo "✓ setup.py"
else
    echo "✗ setup.py (MISSING)"
fi

if [ -f "requirements.txt" ]; then
    echo "✓ requirements.txt"
else
    echo "✗ requirements.txt (MISSING)"
fi

if [ -f "CONTRIBUTING.md" ]; then
    echo "✓ CONTRIBUTING.md"
else
    echo "✗ CONTRIBUTING.md (MISSING)"
fi

# 6. List current structure
echo ""
echo "=========================================="
echo "Current files:"
echo "=========================================="
ls -la

echo ""
echo "Done!"