import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# Load data from the repository
exp3_url = r'data files interferometry\helium.csv'
df_exp3 = pd.read_csv(exp3_url)

x_col = df_exp3.columns[0]
y_col = df_exp3.columns[1]

# Convert to numeric and drop NaN values
df_exp3[x_col] = pd.to_numeric(df_exp3[x_col], errors='coerce')
df_exp3[y_col] = pd.to_numeric(df_exp3[y_col], errors='coerce')
df_exp3_clean = df_exp3.dropna().reset_index(drop=True)

x_data = df_exp3_clean[x_col].values
y_data = df_exp3_clean[y_col].values

# Find rough approximate maxima
min_distance = len(x_data) // 900
rough_peaks, _ = find_peaks(y_data, distance=min_distance, prominence=np.max(y_data) * 0.05)

exact_maxima_x = []
exact_maxima_y = []
maxima_x_err = []
fwhm_list = []

window_size = 2

plt.figure(figsize=(12, 7))
plt.plot(x_data, y_data, color='blue', alpha=0.4, marker='.', label='Data Points')

# Iterate through rough peaks to find exact maxima and FWHM
for i, idx in enumerate(rough_peaks):
    start_idx = max(0, idx - window_size)
    end_idx = min(len(x_data), idx + window_size + 1)

    x_window = x_data[start_idx:end_idx]
    y_window = y_data[start_idx:end_idx]

    y_weights = y_window - np.min(y_window)

    if np.sum(y_weights) == 0:
        x_max = x_data[idx]
    else:
        x_max = np.average(x_window, weights=y_weights)

    y_max = np.max(y_window)

    resolution = np.mean(np.diff(x_data))
    err_x_max = resolution / 2

    exact_maxima_x.append(x_max)
    exact_maxima_y.append(y_max)
    maxima_x_err.append(err_x_max)

    # --- FWHM Calculation ---
    half_max = y_max / 2.0

    # Scan left for half-max crossing
    left_idx = idx
    while left_idx > 0 and y_data[left_idx] > half_max:
        left_idx -= 1

    # Linear interpolation for left X crossing
    x1, y1 = x_data[left_idx], y_data[left_idx]
    x2, y2 = x_data[left_idx + 1], y_data[left_idx + 1]
    if y2 != y1:
        x_left = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
    else:
        x_left = x1

    # Scan right for half-max crossing
    right_idx = idx
    while right_idx < len(y_data) - 1 and y_data[right_idx] > half_max:
        right_idx += 1

    # Linear interpolation for right X crossing
    x3, y3 = x_data[right_idx - 1], y_data[right_idx - 1]
    x4, y4 = x_data[right_idx], y_data[right_idx]
    if y4 != y3:
        x_right = x3 + (half_max - y3) * (x4 - x3) / (y4 - y3)
    else:
        x_right = x4

    fwhm = x_right - x_left
    fwhm_list.append(fwhm)

    # Plot the FWHM line
    label_fwhm = 'FWHM' if i == 0 else None  # Only add label to legend once
    plt.hlines(y=half_max, xmin=x_left, xmax=x_right, color='green', linestyle='--', linewidth=2, zorder=6,
               label=label_fwhm)

exact_maxima_x = np.array(exact_maxima_x)
exact_maxima_y = np.array(exact_maxima_y)
maxima_x_err = np.array(maxima_x_err)

# Plot the exact calculated maxima with error bars
plt.errorbar(exact_maxima_x, exact_maxima_y, xerr=maxima_x_err, fmt='none',
             ecolor='black', capsize=5, capthick=1.5, zorder=4, label='Uncertainties')

plt.plot(exact_maxima_x, exact_maxima_y, marker='o', linestyle='none', color='red',
         markeredgecolor='black', markersize=8, zorder=5, label='Maxima (Center of Mass)')

plt.title('Amplitude vs wave length of helium', fontsize=14)
plt.xlabel('Wave length [nm]', fontsize=12)
plt.ylabel('Amplitude [a.u]', fontsize=12)
plt.grid(True, alpha=0.3)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper left')

plt.tight_layout()
plt.show()

# --- Output the results ---
print("--- Center of Mass Maxima & FWHM Results ---")
for i in range(len(exact_maxima_x)):
    print(
        f"  M{i + 1}: x = ({exact_maxima_x[i]:.3f} +/- {maxima_x_err[i]:.3f}) nm, Amplitude = {exact_maxima_y[i]:.2f} mV, FWHM = {fwhm_list[i]:.4f} mm")

if len(exact_maxima_x) >= 2:
    distance_total = exact_maxima_x[-1] - exact_maxima_x[0]
    num_intervals = len(exact_maxima_x) - 1
    err_distance_total = np.sqrt(maxima_x_err[0] ** 2 + maxima_x_err[-1] ** 2)

    wavelength = (distance_total / num_intervals) * 2
    err_wavelength = (err_distance_total / num_intervals) * 2

    print(f"\n--- Wavelength Analysis ---")
    print(f"Total distance (M1 to M{len(exact_maxima_x)}): ({distance_total:.3f} +/- {err_distance_total:.3f}) mm")
    print(f"Number of intervals (m): {num_intervals}")
    print(f"Calculated Wavelength (lambda = 2*D/m): ({wavelength:.3f} +/- {err_wavelength:.3f}) mm")

print("\n--- Consecutive Maxima Intervals (Half-Wavelengths) ---")
intervals = []
interval_errs = []

for i in range(len(exact_maxima_x) - 1):
    interval = exact_maxima_x[i + 1] - exact_maxima_x[i]
    err_interval = np.sqrt(maxima_x_err[i + 1] ** 2 + maxima_x_err[i] ** 2)
    intervals.append(interval)
    interval_errs.append(err_interval)
    print(f"  Interval M{i + 1} -> M{i + 2}: d = ({interval:.3f} +/- {err_interval:.3f}) mm")

avg_interval = np.mean(intervals)
print(f"\n  Average of individual intervals: {avg_interval:.3f} mm")
avg_fwhm = np.mean(fwhm_list)
print(f"  Average FWHM: {avg_fwhm:.4f} mm")