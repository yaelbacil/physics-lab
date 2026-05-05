import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


# Define the parabolic function for fitting
def parabola(x, a, b, c):
    return a * x ** 2 + b * x + c


# Load data from the repository
exp3_url = 'https://raw.githubusercontent.com/yaelbacil/physics-lab/refs/heads/main/exp%203%20mesure%201.csv'
df_exp3 = pd.read_csv(exp3_url)

x_col = df_exp3.columns[0]
y_col = df_exp3.columns[1]

# Convert to numeric and drop NaN values
df_exp3[x_col] = pd.to_numeric(df_exp3[x_col], errors='coerce')
df_exp3[y_col] = pd.to_numeric(df_exp3[y_col], errors='coerce')
df_exp3_clean = df_exp3.dropna().reset_index(drop=True)

x_data = df_exp3_clean[x_col].values
y_data = df_exp3_clean[y_col].values

# Find rough approximate minima using find_peaks on inverted data
min_distance = len(x_data) // 8
rough_peaks, _ = find_peaks(-y_data, distance=min_distance, prominence=np.max(y_data) * 0.05)

exact_minima_x = []
exact_minima_y = []
minima_x_err = []

# Dynamically set window size to encompass a significant portion of the valley
fit_window = int(min_distance * 0.3)

plt.figure(figsize=(12, 7))
plt.plot(x_data, y_data, color='blue', alpha=0.4, marker='.', label='Data Points')

# Iterate through rough peaks, fit a parabola, and find exact vertex
for idx in rough_peaks:
    start_idx = max(0, idx - fit_window)
    end_idx = min(len(x_data), idx + fit_window + 1)

    x_fit = x_data[start_idx:end_idx]
    y_fit = y_data[start_idx:end_idx]

    try:
        popt, pcov = curve_fit(parabola, x_fit, y_fit)
    except Exception as e:
        print(f"Fit failed for peak near x={x_data[idx]}: {e}")
        continue

    a, b, c = popt

    x_min = -b / (2 * a)
    y_min = parabola(x_min, a, b, c)

    # Error propagation
    df_da = b / (2 * a ** 2)
    df_db = -1 / (2 * a)

    var_a = pcov[0, 0]
    var_b = pcov[1, 1]
    cov_ab = pcov[0, 1]

    var_x_min = (df_da ** 2) * var_a + (df_db ** 2) * var_b + 2 * df_da * df_db * cov_ab
    err_x_min = np.sqrt(var_x_min)

    exact_minima_x.append(x_min)
    exact_minima_y.append(y_min)
    minima_x_err.append(err_x_min)

    # Plot the fitted parabola
    # x_plot = np.linspace(x_fit[0], x_fit[-1], 100)
    # plt.plot(x_plot, parabola(x_plot, a, b, c), color='lime', linestyle='-', linewidth=2.5, alpha=0.9, zorder=3)

exact_minima_x = np.array(exact_minima_x)
exact_minima_y = np.array(exact_minima_y)
minima_x_err = np.array(minima_x_err)


# Plot the exact calculated minima with error bars
plt.errorbar(exact_minima_x, exact_minima_y, xerr=minima_x_err, fmt='none',
             ecolor='black', capsize=5, capthick=1.5, zorder=4, label='Uncertainties')

plt.plot(exact_minima_x, exact_minima_y, marker='o', linestyle='none', color='red',
         markeredgecolor='black', markersize=8, zorder=5, label='Minima')


# plt.plot([], [], color='lime', linestyle='-', linewidth=2.5, label='Fitted Parabola Curve')

plt.title('Amplitude vs Distance - Standing Wave', fontsize=14)
plt.xlabel('Distance [mm]', fontsize=12)
plt.ylabel('Amplitude [mV]', fontsize=12)
plt.grid(True, alpha=0.3)

handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='upper right')

plt.tight_layout()
plt.show()

# --- Output the results ---
print("--- Parabolic Fit Minima Results ---")
for i in range(len(exact_minima_x)):
    print(f"  M{i + 1}: x = ({exact_minima_x[i]:.3f} +/- {minima_x_err[i]:.3f}) mm, y = {exact_minima_y[i]:.1f} mV")

if len(exact_minima_x) >= 2:
    # Wavelength calculation based on the distance between first and last minima
    distance_total = exact_minima_x[-1] - exact_minima_x[0]
    num_intervals = len(exact_minima_x) - 1

    # Uncertainty of the total distance
    err_distance_total = np.sqrt(minima_x_err[0] ** 2 + minima_x_err[-1] ** 2)

    wavelength = (distance_total / num_intervals) * 2
    err_wavelength = (err_distance_total / num_intervals) * 2

    print(f"\n--- Wavelength Analysis ---")
    print(f"Total distance (M1 to M{len(exact_minima_x)}): ({distance_total:.3f} +/- {err_distance_total:.3f}) mm")
    print(f"Number of intervals (m): {num_intervals}")
    print(f"Calculated Wavelength (lambda = 2*D/m): ({wavelength:.3f} +/- {err_wavelength:.3f}) mm")

print("\n--- Consecutive Minima Intervals (Half-Wavelengths) ---")
intervals = []
interval_errs = []

for i in range(len(exact_minima_x) - 1):
    interval = exact_minima_x[i + 1] - exact_minima_x[i]
    err_interval = np.sqrt(minima_x_err[i + 1] ** 2 + minima_x_err[i] ** 2)

    intervals.append(interval)
    interval_errs.append(err_interval)

    print(f"  Interval M{i + 1} -> M{i + 2}: d = ({interval:.3f} +/- {err_interval:.3f}) mm")

avg_interval = np.mean(intervals)
print(f"\n  Average of individual intervals: {avg_interval:.3f} mm")