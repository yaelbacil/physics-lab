import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


# Define the parabolic function for fitting the peak
def parabola(x, a, b, c):
    return a * x ** 2 + b * x + c


# Load data from the repository using the raw URL
green_led_url = 'https://raw.githubusercontent.com/yaelbacil/physics-lab/main/interferometry/data%20files%20interferometry/greenLight.csv'
df_led = pd.read_csv(green_led_url)

x_col = df_led.columns[0]
y_col = df_led.columns[1]

# Convert to numeric and drop NaN values
df_led[x_col] = pd.to_numeric(df_led[x_col], errors='coerce')
df_led[y_col] = pd.to_numeric(df_led[y_col], errors='coerce')
df_clean = df_led.dropna().reset_index(drop=True)

x_data = df_clean[x_col].values
y_data = df_clean[y_col].values

# Find rough approximate maximum using find_peaks
min_distance = len(x_data) // 10
rough_peaks, _ = find_peaks(y_data, distance=min_distance, prominence=np.max(y_data) * 0.1)

if len(rough_peaks) > 0:
    # Get the index of the highest peak
    main_peak_idx = rough_peaks[np.argmax(y_data[rough_peaks])]

    # Dynamically set window size to encompass the tip of the peak for the fit
    fit_window = 15

    start_idx = max(0, main_peak_idx - fit_window)
    end_idx = min(len(x_data), main_peak_idx + fit_window + 1)

    x_fit = x_data[start_idx:end_idx]
    y_fit = y_data[start_idx:end_idx]

    try:
        # Fit a parabola to find the exact vertex
        popt, pcov = curve_fit(parabola, x_fit, y_fit)
        a, b, c = popt

        # Calculate exact vertex coordinates
        exact_x_max = -b / (2 * a)
        exact_y_max = parabola(exact_x_max, a, b, c)

        # --- FWHM Calculation ---
        # Define baseline as minimum value in data to accurately calculate half-max
        baseline = np.min(y_data)
        half_max = baseline + (exact_y_max - baseline) / 2.0

        # Find exactly where the data crosses the half maximum line
        signs = np.sign(y_data - half_max)
        crossings = np.where(np.diff(signs))[0]

        # Separate crossings to the left and right of the peak
        left_crossings = crossings[crossings < main_peak_idx]
        right_crossings = crossings[crossings >= main_peak_idx]

        if len(left_crossings) > 0 and len(right_crossings) > 0:
            left_idx = left_crossings[-1]
            right_idx = right_crossings[0]

            # Linear interpolation for sub-pixel accuracy at the half maximum
            x_left = x_data[left_idx] + (half_max - y_data[left_idx]) * (x_data[left_idx + 1] - x_data[left_idx]) / (
                        y_data[left_idx + 1] - y_data[left_idx])
            x_right = x_data[right_idx] + (half_max - y_data[right_idx]) * (
                        x_data[right_idx + 1] - x_data[right_idx]) / (y_data[right_idx + 1] - y_data[right_idx])

            # Full Width at Half Maximum
            fwhm = x_right - x_left

            # Print the summary
            print("--- Analysis Results ---")
            print(f"Exact Maxima: λ = {exact_x_max:.3f} nm, Intensity = {exact_y_max:.3f}")
            print(f"Half Maximum Intensity = {half_max:.3f}")
            print(f"Left Intersect (λ1) = {x_left:.3f} nm")
            print(f"Right Intersect (λ2) = {x_right:.3f} nm")
            print(f"Calculated FWHM (Δλ) = {fwhm:.3f} nm")

            # --- Plotting ---
            plt.figure(figsize=(12, 7))

            # Plot the data points
            plt.plot(x_data, y_data, color='blue', marker='.', linestyle='-', linewidth=1, markersize=5, alpha=0.4,
                     label='Data Points')

            # Plot the FWHM line
            plt.hlines(half_max, x_left, x_right, color='green', linestyle='--', linewidth=2, label='FWHM')

            # Plot the Maxima
            plt.plot(exact_x_max, exact_y_max, marker='o', color='red', markeredgecolor='black', markersize=8,
                     linestyle='none', label='Maxima')

            # Vertical drop lines for lambda 1 and lambda 2
            plt.vlines([x_left, x_right], ymin=baseline, ymax=half_max, color='green', linestyle=':', alpha=0.7)

            # Add text labels for lambda 1 and lambda 2
            plt.text(x_left, baseline, r'$\lambda_1$', verticalalignment='bottom', horizontalalignment='right',
                     color='green', fontsize=12)
            plt.text(x_right, baseline, r'$\lambda_2$', verticalalignment='bottom', horizontalalignment='left',
                     color='green', fontsize=12)
            plt.hlines(half_max, x_left, x_right, color='green', linestyle='--', linewidth=2,
                       label=f'FWHM (λ1 = {x_left:.2f} nm, λ2 = {x_right:.2f} nm)')

            plt.title('Intensity vs Wave length of Green LED', fontsize=14)
            plt.xlabel('Wave length [nm]', fontsize=12)
            plt.ylabel('Intensity [a.u.]', fontsize=12)
            plt.grid(True, alpha=0.3)

            plt.legend(loc='upper left', framealpha=1.0)

            plt.tight_layout()
            plt.show()

        else:
            print("Error: Could not find both left and right intersections for the half maximum.")

    except Exception as e:
        print(f"Fit failed for peak: {e}")
else:
    print("No peaks were found in the data.")