import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# Apply smoothing to reduce noise
from scipy.ndimage import uniform_filter1d
smoothing_window = 7
y_smoothed = uniform_filter1d(df_exp3_clean[y_col].values, size=smoothing_window, mode='nearest')

# Find local minima on smoothed data using a larger order initially
from scipy.signal import argrelextrema
minima_indices_large = argrelextrema(y_smoothed, np.less, order=15)[0]

# Divide data into 5 regions and find the deepest minimum in each region
n_regions = 5
region_size = len(df_exp3_clean) // n_regions
minima_per_region = []

for region_idx in range(n_regions):
    start_idx = region_idx * region_size
    end_idx = (region_idx + 1) * region_size if region_idx < n_regions - 1 else len(df_exp3_clean)

    region_y = y_smoothed[start_idx:end_idx]
    region_avg = np.mean(region_y)
    region_min = np.min(region_y)

    # Region-specific order ranges for better detection
    # Regions 1, 4, 5 need more sensitive detection
    if region_idx in [0, 3, 4]:  # Regions 1, 4, 5
        order_list = [12, 10, 8, 6, 4, 3, 2, 1]
    else:
        order_list = [15, 12, 10, 8, 6, 4, 3, 2]

    # Try different order values and keep only TRUE minima
    found_minima = False

    for order in order_list:
        local_minima = argrelextrema(region_y, np.less, order=order)[0]

        if len(local_minima) > 0:
            # Filter to keep only TRUE local minima - points lower than both neighbors
            true_minima = []
            for idx in local_minima:
                if idx > 0 and idx < len(region_y) - 1:
                    # Check if it's a true minimum (lower than both neighbors)
                    if region_y[idx] < region_y[idx-1] and region_y[idx] < region_y[idx+1]:
                        true_minima.append(idx)

            if len(true_minima) > 0:
                # For regions 1, 4, 5: prefer minima closer to region minimum
                # For regions 2, 3: use the original threshold logic
                if region_idx in [0, 3, 4]:
                    # More lenient: take the deepest among all true minima
                    valid_minima = true_minima
                else:
                    # Original logic: filter by region average
                    valid_minima = [idx for idx in true_minima if region_y[idx] < region_avg * 0.98]

                    if not valid_minima:
                        if len(true_minima) > 1:
                            valid_minima = true_minima
                        else:
                            continue

                # Get the deepest valid minimum
                deepest_local = valid_minima[np.argmin(region_y[valid_minima])]
                lowest_in_region = start_idx + deepest_local
                minima_per_region.append(lowest_in_region)
                found_minima = True
                break

    if not found_minima:
        # Improved fallback: find the absolute lowest valley in the region
        # Start with widest neighborhood check
        candidates = []
        for i in range(3, len(region_y) - 3):
            # Check in even wider neighborhood
            if (region_y[i] < region_y[i-3] and region_y[i] < region_y[i-1] and
                region_y[i] < region_y[i+1] and region_y[i] < region_y[i+3]):
                candidates.append((i, region_y[i]))

        if candidates:
            best_idx = min(candidates, key=lambda x: x[1])[0]
            lowest_in_region = start_idx + best_idx
            minima_per_region.append(lowest_in_region)
        else:
            # Last resort: find point with lowest value that has higher values on both sides
            final_candidates = []
            for i in range(1, len(region_y) - 1):
                if region_y[i] < region_y[i-1] and region_y[i] < region_y[i+1]:
                    final_candidates.append((i, region_y[i]))

            if final_candidates:
                best_idx = min(final_candidates, key=lambda x: x[1])[0]
                lowest_in_region = start_idx + best_idx
            else:
                # Absolute fallback: lowest point in region
                lowest_in_region = start_idx + np.argmin(region_y)

            minima_per_region.append(lowest_in_region)

top_5_minima_idx = np.array(minima_per_region)

minima_x = df_exp3_clean[x_col].iloc[top_5_minima_idx].values
minima_y = df_exp3_clean[y_col].iloc[top_5_minima_idx].values

plt.figure(figsize=(12, 6))
plt.plot(df_exp3_clean[x_col], df_exp3_clean[y_col], marker='o', linestyle='-', color='b', label='data line', markersize=4)

# Plot the minima points with filled circles (red fill, black outline)
if len(minima_x) > 0:
    plt.plot(minima_x, minima_y, marker='o', linestyle='none', color='black', markersize=12,
             markerfacecolor='red', markeredgecolor='black', markeredgewidth=2, label='Minima points', zorder=5)

    # Print minima x values
    print("Minima X values (sorted):")
    for i, x_val in enumerate(minima_x):
        print(f"  Minima {i+1}: x = {x_val:.4f} mm, amplitude = {minima_y[i]:.0f} mV")

    # Calculate and print delta x between consecutive minima points
    print("\nDelta X between consecutive minima:")
    delta_x_values = []
    for i in range(len(minima_x) - 1):
        delta_x = minima_x[i+1] - minima_x[i]
        delta_x_values.append(delta_x)
        print(f"  Delta x M{i+1} --> M{i+2}: {delta_x:.4f} mm")

    # Calculate uncertainty (standard deviation and standard error)


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

plt.title(f'Amplitude vs Distance - standing wave')
plt.xlabel(f'Distance [mm]')
plt.ylabel(f'Amplitude [mV]')

plt.grid(True, alpha=0.3)
plt.legend(loc='best')
plt.tight_layout()
plt.show()