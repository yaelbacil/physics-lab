import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema, savgol_filter
import scipy.stats as stats

# calling data from github reposatory
exp4_m1 = r"https://raw.githubusercontent.com/yaelbacil/physics-lab/refs/heads/main/exp%204%20mesure%201.csv"
exp4_m2 = r"https://raw.githubusercontent.com/yaelbacil/physics-lab/refs/heads/main/exp%204%20mesure%202.csv"

df_exp4_m1 = pd.read_csv(exp4_m1)
df_exp4_m2 = pd.read_csv(exp4_m2)


def smooth_signal(amplitude, num_minima=6):
    """Smooth the signal lightly to reduce noise-driven minima.

    Uses a Savitzky-Golay filter when possible, with a safe fallback to a
    centered moving average for short arrays.
    """
    if len(amplitude) < 5:
        return amplitude

    window_length = max(5, len(amplitude) // (num_minima * 3))
    window_length = min(window_length, len(amplitude) if len(amplitude) % 2 == 1 else len(amplitude) - 1)
    if window_length < 5:
        window_length = 5 if len(amplitude) >= 5 else len(amplitude)
    if window_length % 2 == 0:
        window_length -= 1

    if window_length >= 5:
        polyorder = min(3, window_length - 2)
        try:
            return savgol_filter(amplitude, window_length=window_length, polyorder=polyorder)
        except Exception:
            pass

    kernel_size = min(5, len(amplitude))
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(amplitude, kernel, mode='same')

def find_minimum(distance, amplitude, num_minima=6):
    smoothed_amplitude = smooth_signal(amplitude, num_minima=num_minima)

    # Adjust order based on data length to find approximately num_minima peaks
    # order determines how many neighbors on each side must be higher
    order = max(2, len(distance) // (2 * num_minima))
    
    # Find local minima indices
    local_minima_idx = argrelextrema(smoothed_amplitude, np.less, order=order)[0]
    
    if len(local_minima_idx) == 0:
        print("Warning: No local minima found. Returning global minimum.")
        min_idx = np.argmin(smoothed_amplitude)
        local_minima_idx = [min_idx]
    elif len(local_minima_idx) > num_minima:
        # If too many minima found, keep the most prominent ones by smoothed amplitude
        sorted_indices = sorted(local_minima_idx, key=lambda i: smoothed_amplitude[i])
        local_minima_idx = np.sort(sorted_indices[:num_minima])
    
    results = []
    
    for min_idx in local_minima_idx:
        min_distance = distance[min_idx]
        min_amplitude = amplitude[min_idx]
        smoothed_min_amplitude = smoothed_amplitude[min_idx]
        
        # Use points around minimum to fit parabola (at least 3 points)
        window_size = min(7, len(distance) // 3)
        start_idx = max(0, min_idx - window_size // 2)
        end_idx = min(len(distance), min_idx + window_size // 2 + 1)
        
        x_window = distance[start_idx:end_idx]
        y_window = smoothed_amplitude[start_idx:end_idx]
        
        # Fit parabola: y = a*x^2 + b*x + c
        def parabola(x, a, b, c):
            return a * x**2 + b * x + c
        
        try:
            popt, pcov = curve_fit(parabola, x_window, y_window)
            a, b, c = popt
            
            # Vertex of parabola is at x = -b/(2a)
            vertex_x = -b / (2 * a)
            
            # Uncertainty from covariance matrix
            a_err, b_err = np.sqrt(np.diag(pcov))[0], np.sqrt(np.diag(pcov))[1]
            distance_uncertainty = np.sqrt((b_err / (2 * a))**2 + (b * a_err / (2 * a**2))**2)
            
            results.append((vertex_x, distance_uncertainty, smoothed_min_amplitude))
        
        except Exception as e:
            # Fallback: use spacing between nearest points
            if min_idx > 0 and min_idx < len(distance) - 1:
                spacing = (distance[min_idx + 1] - distance[min_idx - 1]) / 2
                distance_uncertainty = spacing / np.sqrt(12)
            else:
                distance_uncertainty = np.mean(np.diff(distance)) / 2
            
            results.append((min_distance, distance_uncertainty, smoothed_min_amplitude))
    
    # Sort by distance
    results.sort(key=lambda x: x[0])
    
    return results


def highlight_minima(ax, minima, color='red'):
    for i, (dist, unc, amp) in enumerate(minima, 1):
        ax.errorbar(dist, amp, xerr=unc, fmt='x', color=color, markersize=9,
                    markeredgewidth=2, capsize=3, label='Minimum' if i == 1 else None)
        ax.annotate(str(i), (dist, amp), textcoords='offset points', xytext=(6, 6),
                    fontsize=9, color=color)

distance1 = pd.to_numeric(df_exp4_m1['Distance [mm] - Plot 0'], errors='coerce').to_numpy()
amplitude1 = pd.to_numeric(df_exp4_m1['Amplitude [mV] - Plot 0'], errors='coerce').to_numpy()

distance2 = pd.to_numeric(df_exp4_m2['Distance [mm] - Plot 0'], errors='coerce').to_numpy()
amplitude2 = pd.to_numeric(df_exp4_m2['Amplitude [mV] - Plot 0'], errors='coerce').to_numpy()

# remove (0,0) point
mask1 = (distance1 != 0) & (amplitude1 != 0)
distance1 = distance1[mask1]
amplitude1 = amplitude1[mask1]

mask2 = (distance2 != 0) & (amplitude2 != 0)
distance2 = distance2[mask2]
amplitude2 = amplitude2[mask2]

# Find all minima for both measurements
minima_m1 = find_minimum(distance1, amplitude1)
minima_m2 = find_minimum(distance2, amplitude2)

print("="*60)
print("MINIMA ANALYSIS")
print("="*60)
print(f"Measurement 1 (m1): {len(minima_m1)} minima found")
print("-"*60)
for i, (dist, unc, amp) in enumerate(minima_m1, 1):
    print(f"  Minimum {i}: Distance = {dist:.4f} ± {unc:.4f} mm,  Amplitude = {amp:.4f} mV")

print()
print(f"Measurement 2 (m2): {len(minima_m2)} minima found")
print("-"*60)
for i, (dist, unc, amp) in enumerate(minima_m2, 1):
    print(f"  Minimum {i}: Distance = {dist:.4f} ± {unc:.4f} mm,  Amplitude = {amp:.4f} mV")
print("="*60)
print()

# Plot 1: Measurement 1
fig1, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.scatter(distance1, amplitude1, c='C0', edgecolor='k', alpha=0.8, s=50, label='Data')
highlight_minima(ax1, minima_m1)
ax1.set_xlabel('Distance (mm)')
ax1.set_ylabel('Amplitude (mV)')
ax1.set_title('Measurement 1: Amplitude vs. Distance')
ax1.grid(alpha=0.3)
ax1.legend()
fig1.tight_layout()
plt.show()

# Plot 2: Measurement 2
fig2, ax2 = plt.subplots(figsize=(7, 4.5))
ax2.scatter(distance2, amplitude2, c='C1', edgecolor='k', alpha=0.8, s=50, label='Data')
highlight_minima(ax2, minima_m2)
ax2.set_xlabel('Distance (mm)')
ax2.set_ylabel('Amplitude (mV)')
ax2.set_title('Measurement 2: Amplitude vs. Distance')
ax2.grid(alpha=0.3)
ax2.legend()
fig2.tight_layout()
plt.show()