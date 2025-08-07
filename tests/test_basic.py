"""Basic tests for PyIRMembraneAnalyzer"""

import sys
import os
import tempfile

# 親ディレクトリをパスに追加（ir_membrane_analyzer.pyがある場所）
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import ir_membrane_analyzer as ima

def test_import():
    """Test that module can be imported"""
    assert hasattr(ima, 'process_ir_spectrum')
    print("✓ Import test passed")

def test_functions_exist():
    """Test that main functions exist"""
    assert hasattr(ima, 'calculate_helix_tilt_angle')
    assert hasattr(ima, 'load_ir_data')
    assert hasattr(ima, 'get_amide_peaks')
    print("✓ Function existence test passed")

def test_helix_tilt_angle():
    """Test tilt angle calculation"""
    angle, s_meas = ima.calculate_helix_tilt_angle(1.5)
    assert angle is not None
    assert 0 <= angle <= 90
    print(f"✓ Tilt angle test passed (angle={angle:.1f}°)")

def test_peak_models():
    """Test peak model functions"""
    peaks_standard = ima.get_amide_peaks('standard')
    assert len(peaks_standard) == 5
    
    peaks_complex = ima.get_amide_peaks('complex')
    assert len(peaks_complex) == 6
    
    peaks_extended = ima.get_amide_peaks('extended')
    assert len(peaks_extended) == 7
    print("✓ Peak models test passed")

def test_load_data():
    """Test data loading with synthetic file"""
    # Create temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        for i in range(100):
            f.write(f"{1600+i},{0.1+i*0.01}\n")
        temp_file = f.name
    
    try:
        result = ima.load_ir_data(temp_file)
        assert result is not None
        x, y = result
        assert len(x) == 100
        assert len(y) == 100
        print("✓ Data loading test passed")
    finally:
        os.unlink(temp_file)

if __name__ == "__main__":
    print("Running basic tests...")
    print("-" * 40)
    try:
        test_import()
        test_functions_exist()
        test_helix_tilt_angle()
        test_peak_models()
        test_load_data()
        print("-" * 40)
        print("✓ All tests passed!")
    except AssertionError as e:
        print(f"✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)