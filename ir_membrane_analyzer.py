#!/usr/bin/env python3
"""
IR spectral analysis with automatic model selection
Version 9.0: Integrated AIC/BIC-based model selection for optimal peak numbers
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit, minimize, differential_evolution
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from datetime import datetime
import os
import pathlib
import traceback
import sys

# Global variable to store peak positions
PEAK_POSITIONS = []

def get_amide_peaks(spectrum_type='standard'):
    """
    Return amide I band peak positions based on spectrum type
    
    Args:
        spectrum_type: 'standard' (5 peaks), 'complex' (6 peaks), or 'extended' (7 peaks)
    
    Returns:
        list of tuples: (peak_position, default_width)
    """
    if spectrum_type == 'complex':
        # 6-peak model for moderately complex spectra
        return [
            (1683, 5),   # β-turn/shoulder
            (1678, 6),   # β-turn / antiparallel β-sheet
            (1658, 9),   # α-helix (main peak)
            (1648, 6),   # Random coil
            (1638, 6),   # β-sheet (antiparallel)
            (1628, 6),   # β-sheet (parallel)
        ]
    elif spectrum_type == 'extended':
        # 7-peak model (only for very complex cases)
        return [
            (1685, 5),   # β-turn (high freq)
            (1678, 6),   # β-turn / antiparallel β-sheet
            (1668, 5),   # β-turn / irregular structure
            (1658, 9),   # α-helix (main peak)
            (1648, 6),   # Random coil
            (1638, 6),   # β-sheet (antiparallel)
            (1628, 6),   # β-sheet (parallel)
        ]
    else:
        # Standard 5-peak model (default)
        return [
            (1678, 6),   # β-turn / antiparallel β-sheet
            (1658, 9),   # α-helix (main peak)
            (1648, 6),   # Random coil
            (1638, 6),   # β-sheet (antiparallel)
            (1628, 6),   # β-sheet (parallel)
        ]

def get_peak_labels(spectrum_type='standard'):
    """
    Get appropriate labels for peaks based on model type
    """
    if spectrum_type == 'complex':
        return [
            "β-turn/shoulder",
            "β-turn/antiparallel β-sheet", 
            "α-helix",
            "Random coil",
            "β-sheet (antiparallel)",
            "β-sheet (parallel)"
        ]
    elif spectrum_type == 'extended':
        return [
            "β-turn (high freq)",
            "β-turn/antiparallel β-sheet",
            "β-turn/irregular",
            "α-helix",
            "Random coil",
            "β-sheet (antiparallel)",
            "β-sheet (parallel)"
        ]
    else:
        return [
            "β-turn/antiparallel β-sheet",
            "α-helix",
            "Random coil",
            "β-sheet (antiparallel)",
            "β-sheet (parallel)"
        ]

def multi_gaussian(x, *params):
    """Multiple Gaussian functions with fixed positions"""
    y = np.zeros_like(x)
    for i in range(0, len(params), 2):
        if i+1 >= len(params):
            continue
        amp = params[i]
        wid = params[i+1]
        
        peak_idx = i // 2
        if peak_idx < len(PEAK_POSITIONS):
            cen = PEAK_POSITIONS[peak_idx]
            y += amp * np.exp(-(x - cen)**2 / (2 * wid**2))
    return y

def multi_gaussian_with_positions(x, *params):
    """Multiple Gaussian functions with adjustable positions"""
    y = np.zeros_like(x)
    for i in range(0, len(params), 3):
        if i+2 >= len(params):
            continue
        amp = params[i]
        cen = params[i+1]
        wid = params[i+2]
        y += amp * np.exp(-(x - cen)**2 / (2 * wid**2))
    return y

def multi_gaussian_for_aic(x, positions, amplitudes, widths):
    """
    Calculate multiple Gaussian functions for AIC calculation
    
    Args:
        x: wavenumber array
        positions: list of peak positions
        amplitudes: list of peak amplitudes
        widths: list of peak widths
    
    Returns:
        y: calculated spectrum
    """
    y = np.zeros_like(x)
    for pos, amp, wid in zip(positions, amplitudes, widths):
        y += amp * np.exp(-(x - pos)**2 / (2 * wid**2))
    return y

def find_local_minima(x, y, window_size=5):
    """
    Find local minima in a spectrum
    
    Args:
        x: x-coordinates (wavenumbers)
        y: y-coordinates (intensities)
        window_size: size of the window to check for local minima
    
    Returns:
        tuple of (indices of minima, x-coordinates of minima, y-coordinates of minima)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    window_size = max(5, min(window_size, len(y) // 4))
    
    minima_indices = []
    minima_x = []
    minima_y = []
    
    for i in range(window_size, len(y) - window_size):
        is_minimum = (
            all(y[i] <= y[j] for j in range(i - window_size, i)) and
            all(y[i] <= y[j] for j in range(i + 1, i + window_size + 1))
        )
        
        if is_minimum:
            minima_indices.append(i)
            minima_x.append(x[i])
            minima_y.append(y[i])
    
    minima_indices = np.array(minima_indices, dtype=int)
    minima_x = np.array(minima_x)
    minima_y = np.array(minima_y)
    
    if len(minima_x) == 0:
        return (
            np.array([], dtype=int),
            np.array([x[0], x[-1]]),
            np.array([y[0], y[-1]])
        )
    
    return (minima_indices, minima_x, minima_y)

def improved_baseline(x, y, degree=3):
    """
    Improved baseline correction with focus on 1690 cm⁻¹ region
    
    Args:
        x: wavenumber array
        y: intensity array
        degree: polynomial degree
    
    Returns:
        Baseline-corrected intensity array and baseline
    """
    min_indices, min_x, min_y = find_local_minima(x, y)
    
    baseline_x = np.concatenate(([x[0], x[-1]], min_x))
    baseline_y = np.concatenate(([y[0], y[-1]], min_y))
    
    target_regions = [(1600, 1610), (1685, 1695)]
    
    priority_mask = np.zeros(len(baseline_x), dtype=bool)
    for region_min, region_max in target_regions:
        region_mask = (baseline_x >= region_min) & (baseline_x <= region_max)
        priority_mask |= region_mask
    
    priority_inds = np.argsort(priority_mask)[::-1]
    baseline_x = baseline_x[priority_inds]
    baseline_y = baseline_y[priority_inds]
    
    sort_inds = np.argsort(baseline_x)
    baseline_x = baseline_x[sort_inds]
    baseline_y = baseline_y[sort_inds]
    
    if len(baseline_x) > degree:
        poly_coeffs = np.polyfit(baseline_x, baseline_y, degree)
        baseline = np.polyval(poly_coeffs, x)
        y_corrected = y - baseline
        
        min_val = np.min(y_corrected)
        if min_val < 0:
            y_corrected = y_corrected - min_val
        
        return y_corrected, baseline
    else:
        coeffs = np.polyfit([x[0], x[-1]], [y[0], y[-1]], 1)
        baseline = np.polyval(coeffs, x)
        y_corrected = y - baseline
        
        min_val = np.min(y_corrected)
        if min_val < 0:
            y_corrected = y_corrected - min_val
        
        return y_corrected, baseline

def fit_model_for_aic(x, y_smooth, peak_positions_widths, max_iterations=3):
    """
    Fit a model with given peak positions for AIC calculation
    
    Args:
        x: wavenumber array
        y_smooth: smoothed intensity array
        peak_positions_widths: list of (position, default_width) tuples
        max_iterations: maximum number of optimization attempts
    
    Returns:
        tuple: (fitted_sum, residuals, n_params, success)
    """
    n_peaks = len(peak_positions_widths)
    positions = [p[0] for p in peak_positions_widths]
    default_widths = [p[1] for p in peak_positions_widths]
    
    def objective_function(params):
        try:
            amplitudes = params[:n_peaks]
            widths = params[n_peaks:]
            
            y_pred = multi_gaussian_for_aic(x, positions, amplitudes, widths)
            residuals = y_smooth - y_pred
            return np.sum(residuals**2)
        except:
            return 1e10
    
    max_height = np.max(y_smooth)
    bounds = []
    
    for pos, _ in peak_positions_widths:
        bounds.append((0.001 * max_height, 2.0 * max_height))
    
    for pos, default_width in peak_positions_widths:
        if 1655 <= pos <= 1660:
            bounds.append((5.0, 10.0))
        elif pos >= 1675:
            bounds.append((3.0, 8.0))
        else:
            bounds.append((4.0, 9.0))
    
    initial_guess = []
    
    for pos, _ in peak_positions_widths:
        idx = np.abs(x - pos).argmin()
        amp = max(0.05 * max_height, y_smooth[idx] * 0.6)
        initial_guess.append(amp)
    
    initial_guess.extend(default_widths)
    
    best_result = None
    best_score = np.inf
    
    for attempt in range(max_iterations):
        try:
            result = differential_evolution(
                objective_function,
                bounds,
                seed=42 + attempt * 100,
                maxiter=300,
                popsize=15,
                atol=1e-4,
                tol=1e-4,
                mutation=(0.5, 1.0),
                recombination=0.7,
                workers=1,
                disp=False
            )
            
            if result.fun < best_score:
                best_score = result.fun
                best_result = result
                
        except Exception as e:
            continue
    
    if best_result is None:
        fitted_sum = multi_gaussian_for_aic(x, positions, 
                                           initial_guess[:n_peaks], 
                                           initial_guess[n_peaks:])
        residuals = y_smooth - fitted_sum
        n_params = len(initial_guess)
        return fitted_sum, residuals, n_params, False
    
    best_amplitudes = best_result.x[:n_peaks]
    best_widths = best_result.x[n_peaks:]
    
    fitted_sum = multi_gaussian_for_aic(x, positions, best_amplitudes, best_widths)
    residuals = y_smooth - fitted_sum
    n_params = len(best_result.x)
    
    return fitted_sum, residuals, n_params, True

def calculate_aic(x, y_smooth, fitted_sum, n_params):
    """
    Calculate Akaike Information Criterion (AIC)
    """
    n_data = len(x)
    residuals = y_smooth - fitted_sum
    rss = np.sum(residuals**2)
    
    if rss <= 0:
        rss = 1e-10
    
    aic = n_data * np.log(rss / n_data) + 2 * n_params
    
    if n_data / n_params < 40:
        aic_correction = (2 * n_params * (n_params + 1)) / (n_data - n_params - 1)
        aic += aic_correction
    
    return aic

def calculate_bic(x, y_smooth, fitted_sum, n_params):
    """
    Calculate Bayesian Information Criterion (BIC)
    """
    n_data = len(x)
    residuals = y_smooth - fitted_sum
    rss = np.sum(residuals**2)
    
    if rss <= 0:
        rss = 1e-10
    
    bic = n_data * np.log(rss / n_data) + n_params * np.log(n_data)
    
    return bic

def select_optimal_model(x, y_smooth, criterion='aic', verbose=True):
    """
    Select optimal number of peaks using information criterion
    """
    models = {
        'standard': get_amide_peaks('standard'),
        'complex': get_amide_peaks('complex'),
        'extended': get_amide_peaks('extended')
    }
    
    model_scores = {}
    model_details = {}
    
    if verbose:
        print(f"\nModel selection using {criterion.upper()}:")
        print("-" * 60)
    
    for model_name, peaks in models.items():
        if verbose:
            print(f"\nTesting {model_name} model ({len(peaks)} peaks)...")
        
        fitted_sum, residuals, n_params, success = fit_model_for_aic(x, y_smooth, peaks)
        
        r2 = r2_score(y_smooth, fitted_sum)
        
        if criterion.lower() == 'bic':
            score = calculate_bic(x, y_smooth, fitted_sum, n_params)
        else:
            score = calculate_aic(x, y_smooth, fitted_sum, n_params)
        
        model_scores[model_name] = score
        model_details[model_name] = {
            'score': score,
            'r2': r2,
            'n_peaks': len(peaks),
            'n_params': n_params,
            'rss': np.sum(residuals**2),
            'success': success
        }
        
        if verbose:
            print(f"  {criterion.upper()}: {score:.2f}")
            print(f"  R²: {r2:.4f}")
            print(f"  RSS: {model_details[model_name]['rss']:.6f}")
            print(f"  Success: {success}")
    
    best_model = min(model_scores, key=model_scores.get)
    best_score = model_scores[best_model]
    
    if criterion.lower() == 'aic':
        min_score = min(model_scores.values())
        delta_scores = {m: s - min_score for m, s in model_scores.items()}
        weights = {m: np.exp(-0.5 * d) for m, d in delta_scores.items()}
        total_weight = sum(weights.values())
        probabilities = {m: w / total_weight for m, w in weights.items()}
        
        if verbose:
            print(f"\n{'-' * 60}")
            print("Model probabilities (Akaike weights):")
            for model_name, prob in probabilities.items():
                print(f"  {model_name}: {prob:.1%}")
    
    if verbose:
        print(f"\n{'-' * 60}")
        print(f"Best model: {best_model} ({model_details[best_model]['n_peaks']} peaks)")
        print(f"Best {criterion.upper()}: {best_score:.2f}")
        print(f"Best R²: {model_details[best_model]['r2']:.4f}")
    
    return best_model, models[best_model], model_details

def recommend_model_for_sample(x, y_smooth, sample_name=None):
    """
    Recommend the best model for a specific sample based on multiple criteria
    """
    if sample_name:
        print(f"\n{'=' * 60}")
        print(f"Model recommendation for: {sample_name}")
        print(f"{'=' * 60}")
    
    # Test with AIC
    print("\n--- AIC Analysis ---")
    aic_model, aic_peaks, aic_details = select_optimal_model(x, y_smooth, 'aic', verbose=True)
    
    # Test with BIC
    print("\n--- BIC Analysis ---")
    bic_model, bic_peaks, bic_details = select_optimal_model(x, y_smooth, 'bic', verbose=True)
    
    # Display comparison table
    print(f"\n{'=' * 60}")
    print("Model Comparison Summary:")
    print("-" * 60)
    print(f"{'Model':<15} {'Peaks':<8} {'AIC':<12} {'BIC':<12} {'R²':<8}")
    print("-" * 60)
    
    for model_name in ['standard', 'complex', 'extended']:
        n_peaks = aic_details[model_name]['n_peaks']
        aic_score = aic_details[model_name]['score']
        bic_score = bic_details[model_name]['score']
        r2 = aic_details[model_name]['r2']
        
        # Mark the selected models
        aic_mark = " (AIC)" if model_name == aic_model else ""
        bic_mark = " (BIC)" if model_name == bic_model else ""
        marks = aic_mark + bic_mark
        
        print(f"{model_name:<15} {n_peaks:<8} {aic_score:<12.2f} {bic_score:<12.2f} {r2:<8.4f}{marks}")
    
    print("-" * 60)
    
    recommendation = {
        'aic_choice': aic_model,
        'bic_choice': bic_model,
        'aic_details': aic_details,
        'bic_details': bic_details
    }
    
    # Decision logic
    if aic_model == bic_model:
        final_choice = aic_model
        reasoning = f"Both AIC and BIC agree on {final_choice} model"
        print(f"\nCONSENSUS: AIC and BIC both select {final_choice} model")
    else:
        print(f"\nDISAGREEMENT: AIC selects {aic_model}, BIC selects {bic_model}")
        
        # Compare R² values
        aic_r2 = aic_details[aic_model]['r2']
        bic_r2 = bic_details[bic_model]['r2']
        r2_diff = aic_r2 - bic_r2
        
        print(f"  R² comparison: {aic_model}={aic_r2:.4f}, {bic_model}={bic_r2:.4f} (diff={r2_diff:.4f})")
        
        if r2_diff > 0.05:
            final_choice = aic_model
            reasoning = f"AIC model ({aic_model}) has significantly better fit (R²={aic_r2:.3f} vs {bic_r2:.3f})"
            print(f"  Decision: Choose {aic_model} due to significantly better fit (>5% R² improvement)")
        else:
            final_choice = bic_model
            reasoning = f"BIC model ({bic_model}) preferred for parsimony (similar R² but fewer parameters)"
            print(f"  Decision: Choose {bic_model} for parsimony (R² difference <5%)")
    
    recommendation['final_choice'] = final_choice
    recommendation['reasoning'] = reasoning
    recommendation['peak_positions'] = get_amide_peaks(final_choice)
    
    print(f"\n{'=' * 60}")
    print(f"FINAL RECOMMENDATION: Use {final_choice} model")
    print(f"Reasoning: {reasoning}")
    print(f"{'=' * 60}\n")
    
    return recommendation

def auto_optimize_fitting(x, y_smooth, peak_positions, allow_position_shift=True, max_shift=3.0):
    """
    自動的にフィッティングパラメータを最適化
    """
    print(f"Starting automatic fitting optimization with {len(peak_positions)} peaks...")
    print("This may take 30-60 seconds...")
    
    initial_positions = [pos for pos, _ in peak_positions]
    
    if allow_position_shift:
        print(f"Peak positions can shift up to ±{max_shift} cm⁻¹")
        
        def objective_function(params):
            try:
                n_peaks = len(peak_positions)
                amplitudes = params[:n_peaks]
                positions = params[n_peaks:2*n_peaks]
                widths = params[2*n_peaks:]
                
                all_params = []
                for i in range(n_peaks):
                    all_params.extend([amplitudes[i], positions[i], widths[i]])
                
                y_pred = multi_gaussian_with_positions(x, *all_params)
                score = r2_score(y_smooth, y_pred)
                
                position_penalty = 0
                for i, (orig_pos, _) in enumerate(peak_positions):
                    shift = abs(positions[i] - orig_pos)
                    if shift > max_shift:
                        position_penalty += (shift - max_shift) * 0.1
                
                return -(score - position_penalty)
            except:
                return 1e10
        
        bounds = []
        max_height = np.max(y_smooth)
        
        for i, (pos, _) in enumerate(peak_positions):
            idx = np.abs(x - pos).argmin()
            local_height = y_smooth[idx]
            bounds.append((0.001 * max_height, 3.0 * max_height))
        
        for pos, _ in peak_positions:
            bounds.append((pos - max_shift, pos + max_shift))
        
        for pos, _ in peak_positions:
            if pos >= 1655 and pos <= 1660:
                bounds.append((6.0, 10.0))
            elif pos >= 1645 and pos <= 1650:
                bounds.append((4.0, 7.0))
            elif pos >= 1675:
                bounds.append((3.0, 7.0))
            else:
                bounds.append((4.0, 8.0))
        
        initial_guess = []
        for i, (pos, _) in enumerate(peak_positions):
            idx = np.abs(x - pos).argmin()
            amp = max(0.05 * max_height, y_smooth[idx] * 0.5)
            initial_guess.append(amp)
        initial_guess.extend(initial_positions)
        for _, width in peak_positions:
            initial_guess.append(width)
        
    else:
        global PEAK_POSITIONS
        PEAK_POSITIONS = initial_positions
        
        def objective_function(params):
            try:
                y_pred = multi_gaussian(x, *params)
                score = r2_score(y_smooth, y_pred)
                return -score
            except:
                return 1e10
        
        bounds = []
        max_height = np.max(y_smooth)
        initial_guess = []
        
        for i, (pos, default_width) in enumerate(peak_positions):
            idx = np.abs(x - pos).argmin()
            amp = max(0.05 * max_height, y_smooth[idx] * 0.5)
            initial_guess.append(amp)
            bounds.append((0.001 * max_height, 2.0 * max_height))
            
            initial_guess.append(default_width)
            if pos >= 1655 and pos <= 1660:
                bounds.append((4.0, 10.0))
            elif pos >= 1670:
                bounds.append((3.0, 9.0))
            else:
                bounds.append((3.0, 9.0))
    
    best_result = None
    best_score = -np.inf
    
    for trial in range(3):
        print(f"  Trial {trial + 1}/3...", end='', flush=True)
        try:
            result = differential_evolution(
                objective_function,
                bounds,
                seed=42 + trial * 100,
                maxiter=500,
                popsize=20,
                atol=1e-6,
                tol=1e-6,
                mutation=(0.5, 1.0),
                recombination=0.7,
                workers=1,
                disp=False
            )
            
            if allow_position_shift:
                n_peaks = len(peak_positions)
                amplitudes = result.x[:n_peaks]
                positions = result.x[n_peaks:2*n_peaks]
                widths = result.x[2*n_peaks:]
                
                all_params = []
                for i in range(n_peaks):
                    all_params.extend([amplitudes[i], positions[i], widths[i]])
                
                y_pred = multi_gaussian_with_positions(x, *all_params)
            else:
                y_pred = multi_gaussian(x, *result.x)
            
            score = r2_score(y_smooth, y_pred)
            
            print(f" R² = {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_result = result
                
        except Exception as e:
            print(f" Failed")
    
    if best_result is None:
        print("Optimization failed, using initial parameters")
        if allow_position_shift:
            return initial_guess, initial_positions, 0.0
        else:
            return initial_guess, initial_positions, 0.0
    
    print(f"Best R² score achieved: {best_score:.6f}")
    
    if allow_position_shift:
        n_peaks = len(peak_positions)
        amplitudes = best_result.x[:n_peaks]
        positions = best_result.x[n_peaks:2*n_peaks]
        widths = best_result.x[2*n_peaks:]
        
        optimized_params = []
        for i in range(n_peaks):
            optimized_params.extend([amplitudes[i], widths[i]])
        
        for i, (orig_pos, _) in enumerate(peak_positions):
            if abs(positions[i] - orig_pos) > 0.1:
                print(f"  Peak {i+1} shifted: {orig_pos:.1f} → {positions[i]:.1f} cm⁻¹")
        
        return optimized_params, positions.tolist(), best_score
    else:
        return best_result.x, initial_positions, best_score

def setup_initial_params(x, y, peak_positions, initial_widths=None, initial_amps=None):
    """Setup initial parameters with optional custom amplitudes and widths"""
    initial_params = []
    bounds_lower = []
    bounds_upper = []
    
    max_height = np.max(y)
    
    for i, (pos, default_width) in enumerate(peak_positions):
        idx = np.abs(x - pos).argmin()
        
        if initial_amps and i < len(initial_amps):
            amp = initial_amps[i] * max_height
        else:
            amp = max(0.01 * max_height, y[idx] * 0.6)
        
        if initial_widths and i < len(initial_widths):
            width = initial_widths[i]
        else:
            width = default_width
        
        initial_params.extend([amp, width])
        
        if pos >= 1655 and pos <= 1660:
            bounds_lower.extend([0.0, 5.0])
            bounds_upper.extend([max_height * 2.0, 15.0])
        else:
            bounds_lower.extend([0.0, 5.0])
            bounds_upper.extend([max_height * 2.0, 15.0])
    
    return initial_params, bounds_lower, bounds_upper

def safe_curve_fit(x, y, initial_params, bounds_lower, bounds_upper):
    """
    Perform curve fitting with strategies to prevent segmentation faults
    """
    try:
        popt, _ = curve_fit(
            multi_gaussian, 
            x, y,
            p0=initial_params,
            bounds=(bounds_lower, bounds_upper),
            maxfev=2000,
            ftol=1e-3,
            xtol=1e-3,
            method='trf',
            loss='linear'
        )
        return popt
    except Exception as e:
        print(f"First curve_fit attempt failed: {str(e)}")
        
        try:
            popt, _ = curve_fit(
                multi_gaussian, 
                x, y,
                p0=initial_params,
                bounds=(bounds_lower, bounds_upper),
                maxfev=1000,
                ftol=1e-2,
                xtol=1e-2,
                method='trf',
                loss='soft_l1'
            )
            return popt
        except Exception as e:
            print(f"Second curve_fit attempt failed: {str(e)}")
            
            try:
                popt, _ = curve_fit(
                    multi_gaussian, 
                    x, y,
                    p0=initial_params,
                    bounds=(bounds_lower, bounds_upper),
                    maxfev=800,
                    ftol=1e-1,
                    xtol=1e-1,
                    method='trf',
                    loss='cauchy'
                )
                return popt
            except Exception as e:
                print(f"Third curve_fit attempt failed: {str(e)}")
                
                print("Attempting direct minimization...")
                
                def objective(params):
                    pred = multi_gaussian(x, *params)
                    return np.sum((y - pred) ** 2)
                
                try:
                    result = minimize(
                        objective,
                        initial_params,
                        method='Nelder-Mead',
                        options={'maxfev': 2000, 'xatol': 1e-3, 'fatol': 1e-3}
                    )
                    
                    if result.success:
                        return result.x
                    else:
                        print("Minimization failed, using initial parameters")
                        return np.array(initial_params)
                except Exception as e:
                    print(f"Minimization failed: {str(e)}")
                    return np.array(initial_params)

def load_ir_data(file_path):
    """
    Load IR data with improved error handling for different file formats
    """
    try:
        with open(file_path, 'r') as f:
            first_lines = [f.readline() for _ in range(5)]
        
        delimiters = ['\t', ',', ' ', ';']
        delimiter = None
        
        for delim in delimiters:
            if any(delim in line for line in first_lines):
                delimiter = delim
                break
        
        if delimiter is None:
            delimiter = None
        
        try:
            df = pd.read_csv(file_path, header=None, delimiter=delimiter)
            
            if df.shape[1] >= 2:
                x = df.iloc[:, 0].values
                y = df.iloc[:, 1].values
                
                if len(x) > 10 and len(y) > 10:
                    return x, y
        except:
            pass
        
        try:
            df = pd.read_csv(file_path, delimiter=delimiter)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                x = df[numeric_cols[0]].values
                y = df[numeric_cols[1]].values
                
                if len(x) > 10 and len(y) > 10:
                    return x, y
        except:
            pass
        
        try:
            data = np.loadtxt(file_path)
            if data.shape[1] >= 2:
                x = data[:, 0]
                y = data[:, 1]
                
                if len(x) > 10 and len(y) > 10:
                    return x, y
        except:
            pass
        
        print(f"Could not parse data file: {file_path}")
        print(f"Expected format: two columns (wavenumber, intensity)")
        print(f"First few lines of file:")
        for i, line in enumerate(first_lines[:3]):
            print(f"  Line {i+1}: {line.strip()}")
        
        return None
        
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        return None

def process_ir_spectrum(file_path, custom_params=None, use_auto_optimize=True, use_model_selection=True):
    """
    Enhanced IR spectrum analysis with automatic model selection
    
    Args:
        file_path: path to data file
        custom_params: custom parameters
        use_auto_optimize: whether to use automatic optimization
        use_model_selection: whether to use AIC/BIC model selection
    """
    print(f"Processing file: {file_path}")
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    base_name = pathlib.Path(file_path).stem
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    custom_widths = None
    custom_amps = None
    baseline_degree = 3
    
    if custom_params and base_name in custom_params:
        params = custom_params[base_name]
        if 'widths' in params:
            custom_widths = params['widths']
        if 'amplitudes' in params:
            custom_amps = params['amplitudes']
        if 'baseline_degree' in params:
            baseline_degree = params['baseline_degree']
        
        print(f"Using custom parameters for {base_name}:")
        if custom_widths:
            print(f"  Widths: {custom_widths}")
        if custom_amps:
            print(f"  Relative amplitudes: {custom_amps}")
        print(f"  Baseline degree: {baseline_degree}")
    
    try:
        print("Loading data...")
        data = load_ir_data(file_path)
        
        if data is None:
            print(f"Failed to load data from {file_path}")
            return None
        
        x, y = data
        
        print("Selecting amide I band region...")
        mask = (x >= 1600) & (x <= 1700)
        x = x[mask]
        y = y[mask]
        
        if len(x) < 50:
            print(f"Warning: Only {len(x)} data points in range. Interpolating...")
            x_interp = np.linspace(1600, 1700, 100)
            y_interp = np.interp(x_interp, x, y)
            x = x_interp
            y = y_interp
        
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        y = y[sort_idx]
        
        min_indices, min_x, min_y = find_local_minima(x, y)
        print(f"Found {len(min_indices)} local minima in the spectrum")
        
        if np.isscalar(min_x):
            min_x = np.array([min_x])
        if np.isscalar(min_y):
            min_y = np.array([min_y])
        
        print(f"Applying improved baseline correction (degree={baseline_degree})...")
        y_corrected, baseline = improved_baseline(x, y, degree=baseline_degree)
        
        print("Smoothing data...")
        window_length = min(11, len(x) - 2)
        if window_length % 2 == 0:
            window_length -= 1
        polyorder = min(2, window_length - 1)
        
        try:
            y_smooth = savgol_filter(y_corrected, window_length=window_length, polyorder=polyorder)
            edge_size = window_length // 2
            y_smooth[:edge_size] = y_corrected[:edge_size]
            y_smooth[-edge_size:] = y_corrected[-edge_size:]
        except Exception as e:
            print(f"Smoothing error: {str(e)}")
            y_smooth = y_corrected
        
        # Model selection
        if use_model_selection:
            recommendation = recommend_model_for_sample(x, y_smooth, base_name)
            selected_model = recommendation['final_choice']
            amide_peaks = recommendation['peak_positions']
        else:
            # Default to standard model
            selected_model = 'standard'
            amide_peaks = get_amide_peaks('standard')
        
        print(f"\nUsing {selected_model} model with {len(amide_peaks)} peaks")
        peak_labels = get_peak_labels(selected_model)
        
        print("Setting up peak parameters...")
        
        global PEAK_POSITIONS
        
        if use_auto_optimize:
            print("\nUsing automatic parameter optimization...")
            
            initial_params, bounds_lower, bounds_upper = setup_initial_params(
                x, y_smooth, amide_peaks, initial_widths=custom_widths, initial_amps=custom_amps
            )
            
            optimized_params, optimized_positions, optimization_score = auto_optimize_fitting(
                x, y_smooth, amide_peaks, 
                allow_position_shift=True,
                max_shift=3.0
            )
            
            PEAK_POSITIONS = optimized_positions
            
            if optimization_score < 0.90:
                print(f"Warning: Low R² score ({optimization_score:.3f}). Consider manual adjustment.")
            
            popt = optimized_params
        else:
            PEAK_POSITIONS = [pos for pos, _ in amide_peaks]
            
            initial_params, bounds_lower, bounds_upper = setup_initial_params(
                x, y_smooth, amide_peaks, initial_widths=custom_widths, initial_amps=custom_amps
            )
            
            print("Performing curve fitting...")
            popt = safe_curve_fit(x, y_smooth, initial_params, bounds_lower, bounds_upper)
        
        print("Fitting complete!")
        
        print("Calculating fitted peaks...")
        fitted_peaks = []
        peak_areas = []
        
        for i in range(0, len(popt), 2):
            if i+1 >= len(popt):
                continue
                
            amp = popt[i]
            wid = popt[i+1]
            
            peak_idx = i // 2
            if peak_idx >= len(PEAK_POSITIONS):
                continue
                
            cen = PEAK_POSITIONS[peak_idx]
            
            y_peak = amp * np.exp(-(x - cen)**2 / (2 * wid**2))
            fitted_peaks.append(y_peak)
            
            area = amp * wid * np.sqrt(2 * np.pi)
            
            label = peak_labels[peak_idx] if peak_idx < len(peak_labels) else f"Peak {peak_idx+1}"
            
            peak_areas.append({
                'peak_number': peak_idx + 1,
                'label': label,
                'center': cen,
                'amplitude': amp,
                'width': wid,
                'area': area
            })
        
        fitted_sum = np.sum(fitted_peaks, axis=0)
        
        rmse = np.sqrt(np.mean((y_smooth - fitted_sum) ** 2))
        max_intensity = np.max(y_smooth)
        relative_rmse = (rmse / max_intensity) * 100
        
        r2 = r2_score(y_smooth, fitted_sum)
        
        print(f"Fitting error: RMSE = {rmse:.6f} ({relative_rmse:.2f}% of max intensity)")
        print(f"R² score: {r2:.4f}")
        
        print("\nResults:")
        total_area = sum(peak['area'] for peak in peak_areas)
        for peak in peak_areas:
            percentage = (peak['area'] / total_area) * 100
            peak['percentage'] = percentage
            print(f"{peak['label']}:")
            print(f"  Center: {peak['center']:.2f} cm⁻¹")
            print(f"  Amplitude: {peak['amplitude']:.6f}")
            print(f"  Width: {peak['width']:.2f}")
            print(f"  Area: {peak['area']:.6f}")
            print(f"  Percentage: {percentage:.1f}%")
        
        if peak_areas:
            df_results = pd.DataFrame(peak_areas)
            csv_filename = os.path.join(results_dir, f'peak_areas_{base_name}_{timestamp}.csv')
            df_results.to_csv(csv_filename, index=False)
            print(f"Results saved to {csv_filename}")
        
        print("Creating plots...")
        plt.figure(figsize=(10, 7))
        
        plt.plot(x, y, 'gray', alpha=0.5, linewidth=1, label='Original')
        plt.plot(x, baseline, 'g--', linewidth=1, alpha=0.7, label='Baseline')
        
        for mx, my in zip(min_x, min_y):
            if 1685 <= mx <= 1695:
                plt.plot(mx, my, 'ro', markersize=5)
            else:
                plt.plot(mx, my, 'go', markersize=3, alpha=0.5)
        
        plt.plot(x, y_smooth, color='blue', linewidth=2.0, label='Processed')
        
        for i, y_peak in enumerate(fitted_peaks):
            if i < len(peak_areas):
                plt.plot(x, y_peak, color='red', linewidth=0.8, alpha=0.6, 
                         label=f"{peak_areas[i]['label']} ({peak_areas[i]['center']:.1f})")
            else:
                plt.plot(x, y_peak, color='red', linewidth=0.8, alpha=0.6)
        
        if fitted_peaks:
            plt.plot(x, fitted_sum, 'k--', linewidth=1.5, alpha=0.8, label='Fitted Sum')
        
        plt.gca().invert_xaxis()
        plt.xlabel('Wavenumber (cm⁻¹)', fontsize=12)
        plt.ylabel('Intensity', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.ylim(bottom=0)
        
        if peak_areas:
            structure_text = "Secondary Structure:\n"
            for peak in peak_areas:
                structure_text += f"{peak['label']}: {peak['percentage']:.1f}%\n"
            structure_text += f"\nR² = {r2:.4f}"
            structure_text += f"\nModel: {selected_model} ({len(amide_peaks)} peaks)"
            
            plt.annotate(structure_text, xy=(0.02, 0.02), xycoords='axes fraction',
                         bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                         fontsize=9)
        
        plt.tight_layout()
        plot_filename_png = os.path.join(results_dir, f'spectrum_{base_name}_{timestamp}.png')
        plot_filename_svg = os.path.join(results_dir, f'spectrum_{base_name}_{timestamp}.svg')
        plt.savefig(plot_filename_png, dpi=300, bbox_inches='tight')
        plt.savefig(plot_filename_svg, format='svg', bbox_inches='tight')
        plt.close()
        print(f"Plot saved to {plot_filename_png} and {plot_filename_svg}")
        
        return {
            'x': x,
            'y_smooth': y_smooth,
            'fitted_peaks': fitted_peaks,
            'fitted_sum': fitted_sum,
            'peak_areas': peak_areas,
            'rmse': rmse,
            'minima': (min_x, min_y),
            'baseline': baseline,
            'y_original': y,
            'model_type': selected_model,
            'n_peaks': len(amide_peaks),
            'r2': r2
        }
        
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        traceback.print_exc()
        return None

def calculate_helix_tilt_angle(dichroic_ratio):
    """
    Calculate the tilt angle of transmembrane helix from dichroic ratio
    """
    import math
    
    Ex = 1.399
    Ey = 1.514
    Ez = 1.621
    
    try:
        numerator = Ex**2 - dichroic_ratio * Ey**2 + Ez**2
        denominator = Ex**2 - dichroic_ratio * Ey**2 - 2 * Ez**2
        
        if abs(denominator) < 1e-10:
            return None, None
            
        S_meas = numerator / denominator
        
        alpha = 41.8
        S_mem = 0.85
        
        alpha_rad = math.radians(alpha)
        S_dip = (3 * math.cos(alpha_rad)**2 - 1) / 2
        
        if abs(S_dip) < 1e-10 or abs(S_mem) < 1e-10:
            return None, None
            
        S_hel = S_meas / (S_dip * S_mem)
        
        cos_squared_theta = (2 * S_hel + 1) / 3
        
        if cos_squared_theta < 0 or cos_squared_theta > 1:
            if cos_squared_theta < 0:
                cos_squared_theta = 0
            elif cos_squared_theta > 1:
                cos_squared_theta = 1
            
        cos_theta = math.sqrt(cos_squared_theta)
        theta_rad = math.acos(cos_theta)
        theta_deg = math.degrees(theta_rad)
        
        return theta_deg, S_meas
        
    except Exception as e:
        print(f"Error calculating tilt angle: {str(e)}")
        return None, None

def calculate_dichroic_ratio(results_dict):
    """
    Calculate dichroic ratios between samples for each secondary structure
    """
    if len(results_dict) < 2:
        print("Need at least 2 samples to calculate dichroic ratio")
        return None
    
    sample_names = list(results_dict.keys())
    sample_names = [pathlib.Path(name).stem for name in sample_names]
    
    structure_areas = {}
    
    for filename, result in results_dict.items():
        sample_name = pathlib.Path(filename).stem
        structure_areas[sample_name] = {}
        
        if 'peak_areas' in result and result['peak_areas']:
            total_area = sum(peak['area'] for peak in result['peak_areas'])
            
            for peak in result['peak_areas']:
                label = peak['label']
                area = peak['area']
                percentage = (area / total_area) * 100
                structure_areas[sample_name][label] = {
                    'area': area,
                    'percentage': percentage
                }
    
    dichroic_ratios = {}
    reference_sample = sample_names[0]
    
    for i in range(1, len(sample_names)):
        sample_name = sample_names[i]
        
        ratio_name = f"{reference_sample}/{sample_name}"
        dichroic_ratios[ratio_name] = {}
        
        for label in structure_areas[reference_sample]:
            if label in structure_areas[sample_name]:
                ref_area = structure_areas[reference_sample][label]['area']
                sample_area = structure_areas[sample_name][label]['area']
                
                ref_pct = structure_areas[reference_sample][label]['percentage']
                sample_pct = structure_areas[sample_name][label]['percentage']
                
                area_ratio = ref_area / sample_area if sample_area > 0 else float('inf')
                pct_ratio = ref_pct / sample_pct if sample_pct > 0 else float('inf')
                
                dichroic_ratios[ratio_name][label] = {
                    'area_ratio': area_ratio,
                    'percentage_ratio': pct_ratio
                }
    
    return dichroic_ratios

def create_dichroic_ratio_plot(dichroic_ratios, results_dir="results"):
    """Create a bar chart of dichroic ratios with tilt angle annotations"""
    if not dichroic_ratios:
        return
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for ratio_name, structures in dichroic_ratios.items():
        plt.figure(figsize=(12, 8))
        
        labels = []
        area_ratios = []
        pct_ratios = []
        
        for label, data in structures.items():
            labels.append(label)
            area_ratios.append(data['area_ratio'])
            pct_ratios.append(data['percentage_ratio'])
        
        x = np.arange(len(labels))
        width = 0.35
        
        bars1 = plt.bar(x - width/2, area_ratios, width, label='Area Ratio', color='skyblue')
        bars2 = plt.bar(x + width/2, pct_ratios, width, label='Percentage Ratio', color='lightcoral')
        
        for i, (v1, v2, label) in enumerate(zip(area_ratios, pct_ratios, labels)):
            if v1 != float('inf'):
                # dichroic ratioの値を上に表示
                plt.text(i - width/2, v1 + 0.1, f"{v1:.2f}", ha='center', fontsize=9)
                
                # α-helixの場合、角度をdichroic ratioの右横に表示
                if 'helix' in label.lower() and v1 > 0:
                    tilt_angle, s_meas = calculate_helix_tilt_angle(v1)
                    if tilt_angle is not None:
                        # 値の右側に角度を表示（少し右にオフセット）
                        plt.text(i - width/2 + 0.04, v1 + 0.2, f"  θ={tilt_angle:.1f}°", 
                                ha='left', fontsize=8, color='darkblue', weight='bold')
            
            if v2 != float('inf'):
                # percentage ratioの値を上に表示
                plt.text(i + width/2, v2 + 0.1, f"{v2:.2f}", ha='center', fontsize=9)
                
                # α-helixの場合、角度をpercentage ratioの右横に表示
                if 'helix' in label.lower() and v2 > 0:
                    tilt_angle, s_meas = calculate_helix_tilt_angle(v2)
                    if tilt_angle is not None:
                        # 値の右側に角度を表示（少し右にオフセット）
                        plt.text(i + width/2 + 0.04, v2 + 0.2, f"  θ={tilt_angle:.1f}°", 
                                ha='left', fontsize=8, color='darkred', weight='bold')
        
        plt.xlabel('Secondary Structure', fontsize=12)
        plt.ylabel('Dichroic Ratio', fontsize=12)
        plt.title(f'Dichroic Ratio: {ratio_name}', fontsize=14)
        plt.xticks(x, labels, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        # 説明文を左上に配置
        plt.text(0.02, 0.98, 'θ = tilt angle of α-helix relative to membrane normal', 
                transform=plt.gca().transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
        # Y軸の範囲を調整（角度表示のスペースは不要に）
        ylim = plt.ylim()
        plt.ylim(ylim[0], ylim[1] * 1.1)  # 1.15から1.1に変更
        
        plt.tight_layout()
        ratio_filename_png = os.path.join(results_dir, f'dichroic_ratio_{ratio_name.replace("/", "_")}_{timestamp}.png')
        ratio_filename_svg = os.path.join(results_dir, f'dichroic_ratio_{ratio_name.replace("/", "_")}_{timestamp}.svg')
        plt.savefig(ratio_filename_png, dpi=300, bbox_inches='tight')
        plt.savefig(ratio_filename_svg, format='svg', bbox_inches='tight')
        plt.close()
        print(f"Dichroic ratio plot saved to {ratio_filename_png} and {ratio_filename_svg}")
        
       
        
        report_filename = os.path.join(results_dir, f'tilt_angles_{ratio_name.replace("/", "_")}_{timestamp}.txt')
        with open(report_filename, 'w') as f:
            f.write(f"Dichroic Ratio and Tilt Angle Analysis\n")
            f.write(f"=====================================\n")
            f.write(f"Sample comparison: {ratio_name}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Analysis based on: Sato & Shinohara, STAR Protocols 4, 102454 (2023)\n")
            f.write(f"Parameters used:\n")
            f.write(f"  - Electric field amplitudes: Ex=1.399, Ey=1.514, Ez=1.621\n")
            f.write(f"  - Transition dipole angle (α): 41.8°\n")
            f.write(f"  - Membrane order parameter (Smem): 0.85\n\n")
            
            for label, data in structures.items():
                f.write(f"\n{label}:\n")
                f.write(f"  Area Ratio: {data['area_ratio']:.3f}\n")
                f.write(f"  Percentage Ratio: {data['percentage_ratio']:.3f}\n")
                
                if 'helix' in label.lower():
                    if data['area_ratio'] != float('inf'):
                        angle1, s_meas1 = calculate_helix_tilt_angle(data['area_ratio'])
                        if angle1 is not None:
                            f.write(f"  Tilt angle from area ratio: {angle1:.1f}°\n")
                            f.write(f"  Order parameter (Smeas): {s_meas1:.3f}\n")
                    
                    if data['percentage_ratio'] != float('inf'):
                        angle2, s_meas2 = calculate_helix_tilt_angle(data['percentage_ratio'])
                        if angle2 is not None:
                            f.write(f"  Tilt angle from percentage ratio: {angle2:.1f}°\n")
                            f.write(f"  Order parameter (Smeas): {s_meas2:.3f}\n")
        
        print(f"Tilt angle report saved to {report_filename}")

def create_individual_spectrum_plots(results_dict):
    """Create individual spectrum plots for each sample"""
    if not results_dict:
        print("No valid results to save individually")
        return
        
    results_dir = "results"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        for filename, result in results_dict.items():
            if 'x' not in result or 'y_smooth' not in result:
                continue
                
            sample_name = pathlib.Path(filename).stem
            
            plt.figure(figsize=(10, 6))
            
            if '90' in sample_name:
                color = 'blue'
            elif '0' in sample_name:
                color = 'red'
            else:
                color = 'black'
            
            plt.plot(result['x'], result['y_smooth'], 
                    color=color, linewidth=2.5, label=f'{sample_name} (Processed)')
            
            if 'fitted_sum' in result and result['fitted_sum'] is not None:
                plt.plot(result['x'], result['fitted_sum'], 
                        'k--', linewidth=1.5, alpha=0.8, label='Fitted Sum')
            
            if 'fitted_peaks' in result and result['fitted_peaks']:
                for i, y_peak in enumerate(result['fitted_peaks']):
                    if i < len(result['peak_areas']):
                        peak_info = result['peak_areas'][i]
                        plt.plot(result['x'], y_peak, color='gray', linewidth=0.8, alpha=0.4,
                                label=f"{peak_info['label']} ({peak_info['center']:.0f} cm⁻¹)")
            
            plt.gca().invert_xaxis()
            plt.xlabel('Wavenumber (cm⁻¹)', fontsize=12)
            plt.ylabel('Intensity', fontsize=12)
            plt.title(f'IR Spectrum - {sample_name}', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=9, loc='best')
            plt.xlim(1700, 1600)
            
            plt.tight_layout()
            output_filename_png = os.path.join(results_dir, f'individual_spectrum_{sample_name}_{timestamp}.png')
            output_filename_svg = os.path.join(results_dir, f'individual_spectrum_{sample_name}_{timestamp}.svg')
            plt.savefig(output_filename_png, dpi=300, bbox_inches='tight')
            plt.savefig(output_filename_svg, format='svg', bbox_inches='tight')
            plt.close()
            
            print(f"Saved individual spectrum: {output_filename_png} and {output_filename_svg}")
            
            plt.figure(figsize=(10, 6))
            plt.plot(result['x'], result['y_smooth'], 
                    color=color, linewidth=2.5, label=sample_name)
            
            plt.gca().invert_xaxis()
            plt.xlabel('Wavenumber (cm⁻¹)', fontsize=12)
            plt.ylabel('Intensity', fontsize=12)
            plt.title(f'IR Spectrum - {sample_name} (Clean)', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=12)
            plt.xlim(1700, 1600)
            
            plt.tight_layout()
            clean_filename_png = os.path.join(results_dir, f'clean_spectrum_{sample_name}_{timestamp}.png')
            clean_filename_svg = os.path.join(results_dir, f'clean_spectrum_{sample_name}_{timestamp}.svg')
            plt.savefig(clean_filename_png, dpi=300, bbox_inches='tight')
            plt.savefig(clean_filename_svg, format='svg', bbox_inches='tight')
            plt.close()
            
            print(f"Saved clean spectrum: {clean_filename_png} and {clean_filename_svg}")
    
    except Exception as e:
        print(f"Error creating individual spectrum plots: {str(e)}")
        traceback.print_exc()

def create_comparison_plots(results_dict):
    """Create comparison plots for multiple files"""
    if not results_dict:
        print("No valid results to compare")
        return
        
    results_dir = "results"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        plt.figure(figsize=(12, 6))
        
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown']
        linestyles = ['-', '--', ':', '-.']
        
        for i, (filename, result) in enumerate(results_dict.items()):
            color = colors[i % len(colors)]
            linestyle = linestyles[(i // len(colors)) % len(linestyles)]
            
            label = pathlib.Path(filename).stem
            plt.plot(result['x'], result['y_smooth'], 
                    color=color, linestyle=linestyle, 
                    linewidth=2.0, label=label)
        
        plt.gca().invert_xaxis()
        plt.xlabel('Wavenumber (cm⁻¹)', fontsize=12)
        plt.ylabel('Intensity', fontsize=12)
        plt.title('Spectra Comparison', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, loc='best')
        
        plt.tight_layout()
        spectra_file_png = os.path.join(results_dir, f'spectra_comparison_{timestamp}.png')
        spectra_file_svg = os.path.join(results_dir, f'spectra_comparison_{timestamp}.svg')
        plt.savefig(spectra_file_png, dpi=300, bbox_inches='tight')
        plt.savefig(spectra_file_svg, format='svg', bbox_inches='tight')
        plt.close()
        print(f"Spectra comparison saved to {spectra_file_png} and {spectra_file_svg}")
        
        all_labels = set()
        for filename, result in results_dict.items():
            if 'peak_areas' in result and result['peak_areas']:
                for peak in result['peak_areas']:
                    all_labels.add(peak['label'])
        
        structure_labels = sorted(list(all_labels))
        
        samples = []
        structure_data = {label: [] for label in structure_labels}
        
        for filename, result in results_dict.items():
            if 'peak_areas' in result and result['peak_areas']:
                samples.append(pathlib.Path(filename).stem)
                
                data_by_label = {}
                total_area = sum(peak['area'] for peak in result['peak_areas'])
                
                for peak in result['peak_areas']:
                    label = peak['label']
                    percentage = (peak['area'] / total_area) * 100
                    data_by_label[label] = percentage
                
                for label in structure_labels:
                    if label in data_by_label:
                        structure_data[label].append(data_by_label[label])
                    else:
                        structure_data[label].append(0)
        
        if samples:
            plt.figure(figsize=(12, 8))
            
            x = np.arange(len(samples))
            width = 0.8 / len(structure_labels)
            colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#ff6666']
            
            for i, label in enumerate(structure_labels):
                offset = width * (i - len(structure_labels)/2 + 0.5)
                plt.bar(x + offset, structure_data[label], width, label=label, color=colors[i % len(colors)])
            
            plt.xlabel('Sample', fontsize=12)
            plt.ylabel('Percentage (%)', fontsize=12)
            plt.title('Secondary Structure Comparison', fontsize=14)
            plt.xticks(x, samples, rotation=45, ha='right')
            plt.legend(title='Structure', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            comparison_file_png = os.path.join(results_dir, f'structure_comparison_{timestamp}.png')
            comparison_file_svg = os.path.join(results_dir, f'structure_comparison_{timestamp}.svg')
            plt.savefig(comparison_file_png, dpi=300, bbox_inches='tight')
            plt.savefig(comparison_file_svg, format='svg', bbox_inches='tight')
            plt.close()
            print(f"Structure comparison saved to {comparison_file_png} and {comparison_file_svg}")
        
        plt.figure(figsize=(12, 6))
        
        for i, (filename, result) in enumerate(results_dict.items()):
            color = colors[i % len(colors)]
            linestyle = linestyles[(i // len(colors)) % len(linestyles)]
            
            label = pathlib.Path(filename).stem
            
            if 'minima' in result:
                min_x, min_y = result['minima']
                plt.scatter(min_x, min_y, color=color, marker='o', s=20, alpha=0.7, label=f"{label} minima")
        
        plt.gca().invert_xaxis()
        plt.xlabel('Wavenumber (cm⁻¹)', fontsize=12)
        plt.ylabel('Intensity', fontsize=12)
        plt.title('Baseline Anchor Points Comparison', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=9, loc='best')
        
        plt.tight_layout()
        baseline_compare_file_png = os.path.join(results_dir, f'baseline_points_comparison_{timestamp}.png')
        baseline_compare_file_svg = os.path.join(results_dir, f'baseline_points_comparison_{timestamp}.svg')
        plt.savefig(baseline_compare_file_png, dpi=300, bbox_inches='tight')
        plt.savefig(baseline_compare_file_svg, format='svg', bbox_inches='tight')
        plt.close()
        print(f"Baseline points comparison saved to {baseline_compare_file_png} and {baseline_compare_file_svg}")
    
    except Exception as e:
        print(f"Error creating comparison plots: {str(e)}")
        traceback.print_exc()

def process_multiple_files(file_paths, custom_params=None, use_auto_optimize=True, use_model_selection=True):
    """Process multiple files with optional custom parameters"""
    results = {}
    for file_path in file_paths:
        print(f"\nProcessing: {file_path}")
        try:
            if os.path.exists(file_path):
                result = process_ir_spectrum(file_path, custom_params, use_auto_optimize, use_model_selection)
                if result is not None:
                    results[file_path] = result
            else:
                print(f"File not found: {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            traceback.print_exc()
    
    if len(results) > 1:
        print("\nCreating comparison plots...")
        create_comparison_plots(results)
        
        print("\nCreating individual spectrum plots...")
        create_individual_spectrum_plots(results)
        
        print("\nCalculating dichroic ratios...")
        dichroic_ratios = calculate_dichroic_ratio(results)
        if dichroic_ratios:
            create_dichroic_ratio_plot(dichroic_ratios)
    elif len(results) == 1:
        print("\nCreating individual spectrum plot...")
        create_individual_spectrum_plots(results)
    
    return results

# Main program
if __name__ == "__main__":
    try:
        print("Starting IR analysis with automatic model selection...")
        print("Version 9.0: Integrated AIC/BIC-based model selection")
        print("=" * 70)
        
        custom_params = {
            '90': {
                'widths': [6, 9, 6, 6, 6],
                'amplitudes': [0.2, 1.0, 0.2, 0.2, 0.1],
                'baseline_degree': 3
            },
            '0': {
                'widths': [5, 6, 5, 9, 6, 6, 6],
                'amplitudes': [0.3, 0.3, 0.2, 0.7, 0.4, 0.4, 0.3],
                'baseline_degree': 3
            }
        }
        
        import sys
        
        if len(sys.argv) > 1:
            files = sys.argv[1:]
        else:
            files = ['example_90.csv', 'example_0.csv']
        
        print(f"Files to process: {', '.join(files)}")
        
        existing_files = []
        for file in files:
            if os.path.exists(file):
                existing_files.append(file)
                print(f"Found file: {file}")
            else:
                print(f"Warning: File not found: {file}")
        
        if existing_files:
            # Process with automatic model selection enabled
            results = process_multiple_files(
                existing_files, 
                custom_params,
                use_auto_optimize=True,
                use_model_selection=True  # Enable AIC/BIC model selection
            )
            
            if results:
                print("\n" + "=" * 70)
                print("Analysis completed successfully!")
                print("=" * 70)
                print("\nFeatures used:")
                print("✓ Automatic model selection (AIC/BIC)")
                print("✓ Automatic parameter optimization")
                print("✓ Adaptive peak numbers (5, 6, or 7 peaks)")
                print("✓ Dichroic ratio calculation for membrane orientation")
                
                print("\nSummary of results:")
                print("-" * 70)
                for filepath, result in results.items():
                    filename = pathlib.Path(filepath).stem
                    model_type = result.get('model_type', 'unknown')
                    n_peaks = result.get('n_peaks', 0)
                    r2 = result.get('r2', 0)
                    
                    print(f"\n{filename}:")
                    print(f"  Model selected: {model_type} ({n_peaks} peaks)")
                    print(f"  R² score: {r2:.4f}")
                    
                    if 'peak_areas' in result:
                        print("  Secondary structure composition:")
                        for peak in result['peak_areas']:
                            if peak['percentage'] > 5:
                                print(f"    {peak['label']}: {peak['percentage']:.1f}%")
                
                print("\n" + "=" * 70)
                print("All results saved in the 'results' directory")
                print("=" * 70)
            else:
                print("\nNo valid results were produced.")
        else:
            print("No valid files to process")
    
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        traceback.print_exc()