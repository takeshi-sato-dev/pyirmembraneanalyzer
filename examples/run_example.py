#!/usr/bin/env python3
"""
Example usage of PyIRMembraneAnalyzer
"""

import sys
import os
sys.path.insert(0, os.path.abspath('..'))

import ir_membrane_analyzer as ima

def main():
    """Run example analysis"""
    print("PyIRMembraneAnalyzer - Example Analysis")
    print("=" * 50)
    
    # Input files
    files = [
        'demo_data/sample_90.csv',
        'demo_data/sample_0.csv'
    ]
    
    # Check files exist
    for f in files:
        if not os.path.exists(f):
            print(f"Error: {f} not found!")
            return
    
    # Process files
    print("\nProcessing spectra...")
    results = ima.process_multiple_files(
        files,
        use_model_selection=True,
        use_auto_optimize=True
    )
    
    if results:
        print("\n✓ Analysis complete!")
        print("Results saved in '../results' directory")
        
        # Calculate dichroic ratio
        dichroic_ratios = ima.calculate_dichroic_ratio(results)
        if dichroic_ratios:
            print("\nDichroic ratios calculated:")
            for ratio_name, structures in dichroic_ratios.items():
                for label, data in structures.items():
                    if 'helix' in label.lower():
                        angle, _ = ima.calculate_helix_tilt_angle(data['area_ratio'])
                        if angle:
                            print(f"  α-helix tilt angle: {angle:.1f}°")
                        break
                break

if __name__ == "__main__":
    main()
