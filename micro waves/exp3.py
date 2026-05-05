import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, savgol_filter
from scipy.optimize import curve_fit

# calling data from github reposatory
exp3 = 'https://raw.githubusercontent.com/yaelbacil/physics-lab/refs/heads/main/exp%203%20mesure%201.csv'

# creating dataframe
df_exp3 = pd.read_csv(exp3)

# exp 3 data plotting
x_col = df_exp3.columns[0]
y_col = df_exp3.columns[1]

# Convert to numeric values
df_exp3[x_col] = pd.to_numeric(df_exp3[x_col], errors='coerce')
df_exp3[y_col] = pd.to_numeric(df_exp3[y_col], errors='coerce')

# Remove any NaN values
df_exp3_clean = df_exp3.dropna().reset_index(drop=True)

x_values = df_exp3_clean[x_col].values
y_values = df_exp3_clean[y_col].values


def smooth_signal(amplitude, num_minima=5):
    """Smooth the signal more to reduce noise-driven minima."""
    if len(amplitude) < 5:
        return amplitude

    window_length = max(7, len(amplitude) // (num_minima * 2))
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


def find_minimum(distance, amplitude, num_minima=5):
    """Find local minima and return with uncertainties from parabola fitting."""
    smoothed_amplitude = smooth_signal(amplitude, num_minima=num_minima)

    order = max(2, len(distance) // (2 * num_minima))
    local_minima_idx = argrelextrema(smoothed_amplitude, np.less, order=order)[0]

    if len(local_minima_idx) == 0:
        min_idx = np.argmin(smoothed_amplitude)
        local_minima_idx = [min_idx]
    elif len(local_minima_idx) > num_minima:
        sorted_indices = sorted(local_minima_idx, key=lambda i: smoothed_amplitude[i])
        local_minima_idx = np.sort(sorted_indices[:num_minima])

    results = []

    for min_idx in local_minima_idx:
        min_distance = distance[min_idx]
        min_amplitude = amplitude[min_idx]
        smoothed_min_amplitude = smoothed_amplitude[min_idx]

        window_size = min(7, len(distance) // 3)
        start_idx = max(0, min_idx - window_size // 2)
        end_idx = min(len(distance), min_idx + window_size // 2 + 1)

        x_window = distance[start_idx:end_idx]
        y_window = smoothed_amplitude[start_idx:end_idx]

        def parabola(x, a, b, c):
            return a * x**2 + b * x + c

        try:
            popt, pcov = curve_fit(parabola, x_window, y_window)
            a, b, c = popt
            vertex_x = -b / (2 * a)
            a_err, b_err = np.sqrt(np.diag(pcov))[0], np.sqrt(np.diag(pcov))[1]
            distance_uncertainty = np.sqrt((b_err / (2 * a))**2 + (b * a_err / (2 * a**2))**2)
            results.append((vertex_x, distance_uncertainty, smoothed_min_amplitude))
        except Exception:
            if min_idx > 0 and min_idx < len(distance) - 1:
                spacing = (distance[min_idx + 1] - distance[min_idx - 1]) / 2
                distance_uncertainty = spacing / np.sqrt(12)
            else:
                distance_uncertainty = np.mean(np.diff(distance)) / 2
            results.append((min_distance, distance_uncertainty, smoothed_min_amplitude))

    results.sort(key=lambda x: x[0])
    return results


def highlight_minima(ax, minima, color='red'):
    """Plot minima as round dots with error bars and numbered labels."""
    for i, (dist, unc, amp) in enumerate(minima, 1):
        ax.errorbar(dist, amp, xerr=unc, fmt='o', color=color, markersize=8,
                    markeredgewidth=2, capsize=3, label='Minima' if i == 1 else None)
        ax.annotate(str(i), (dist, amp), textcoords='offset points', xytext=(6, 6),
                    fontsize=9, color=color)


# Find minima
minima_results = find_minimum(x_values, y_values, num_minima=5)
minima_x = np.array([m[0] for m in minima_results])
minima_y = np.array([m[2] for m in minima_results])

print("Minima X values (sorted):")
for i, (dist, unc, amp) in enumerate(minima_results, 1):
    print(f"  Minimum {i}: x = {dist:.4f} ± {unc:.4f} mm, amplitude = {amp:.0f} mV")

# Calculate and print delta x between consecutive minima points
print("\nDelta X between consecutive minima:")
delta_x_values = []
for i in range(len(minima_x) - 1):
    delta_x = minima_x[i+1] - minima_x[i]
    delta_x_values.append(delta_x)
    print(f"  Delta x M{i+1} --> M{i+2}: {delta_x:.4f} mm")

# Calculate wavelength using distance from M1 to M5 divided by number of intervals
print(f"\n--- Wavelength Analysis ---")
distance_m1_to_m5 = minima_x[4] - minima_x[0]  # Distance from first to last minima
num_intervals = len(minima_x) - 1  # Number of intervals between minima
wavelength_alt = (distance_m1_to_m5 / num_intervals) * 2

print(f"Distance from M1 to M5: {distance_m1_to_m5:.4f} mm")
print(f"Number of intervals: {num_intervals}")
print(f"Delta x (M1 to M5 / intervals): {distance_m1_to_m5 / num_intervals:.4f} mm")
print(f"Wavelength (lambda = 2 * (M1 to M5 distance / intervals)): {wavelength_alt:.4f} mm")

# Calculate uncertainty of wavelength
std_delta_x = np.std(delta_x_values, ddof=1)  # Sample standard deviation
uncertainty_wavelength = (std_delta_x / num_intervals) * 2
print(f"Wavelength Uncertainty: {uncertainty_wavelength:.4f} mm")
print(f"Wavelength: lambda = ({wavelength_alt:.4f} +/- {uncertainty_wavelength:.4f}) mm")

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(x_values, y_values, color='b', label='data line', s=20)
highlight_minima(ax, minima_results)
ax.set_xlabel(f'{x_col}')
ax.set_ylabel(f'{y_col}')
ax.set_title('Amplitude vs Distance - standing wave')
ax.grid(True, alpha=0.3)
ax.legend(loc='best')
fig.tight_layout()
plt.show()