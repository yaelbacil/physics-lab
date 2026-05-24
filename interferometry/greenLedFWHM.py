import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


# Define the parabolic function for fitting the peak
def parabola(x, a, b, c):
    return a * x ** 2 + b * x + c


# Load data from Git
green_led_url = 'https://raw.githubusercontent.com/yaelbacil/physics-lab/main/interferometry/data%20files%20interferometry/greenLight.csv'
df_led = pd.read_csv(green_led_url)

x_col = df_led.columns[0]
y_col = df_led.columns[1]

# Clean data
df_led[x_col] = pd.to_numeric(df_led[x_col], errors='coerce')
df_led[y_col] = pd.to_numeric(df_led[y_col], errors='coerce')
df_clean = df_led.dropna().reset_index(drop=True)

x_data = df_clean[x_col].values
y_data = df_clean[y_col].values

# Find rough approximate maximum
min_distance = len(x_data) // 10
rough_peaks, _ = find_peaks(y_data, distance=min_distance, prominence=np.max(y_data) * 0.1)

if len(rough_peaks) > 0:
    main_peak_idx = rough_peaks[np.argmax(y_data[rough_peaks])]
    fit_window = 15

    start_idx = max(0, main_peak_idx - fit_window)
    end_idx = min(len(x_data), main_peak_idx + fit_window + 1)

    x_fit = x_data[start_idx:end_idx]
    y_fit = y_data[start_idx:end_idx]

    try:
        # Fit parabola
        popt, pcov = curve_fit(parabola, x_fit, y_fit)
        a, b, c = popt

        exact_x_max = -b / (2 * a)
        exact_y_max = c - (b ** 2) / (4 * a)

        # Uncertainty in y_max
        dy_da = (b ** 2) / (4 * a ** 2)
        dy_db = -b / (2 * a)
        dy_dc = 1.0

        # Covariance matrix
        var_a = pcov[0, 0]
        var_b = pcov[1, 1]
        var_c = pcov[2, 2]
        cov_ab = pcov[0, 1]
        cov_ac = pcov[0, 2]
        cov_bc = pcov[1, 2]

        var_y_max = (dy_da ** 2 * var_a) + (dy_db ** 2 * var_b) + (dy_dc ** 2 * var_c) + \
                    (2 * dy_da * dy_db * cov_ab) + (2 * dy_da * dy_dc * cov_ac) + (2 * dy_db * dy_dc * cov_bc)
        err_y_max = np.sqrt(var_y_max)

        # Define base line as the minimum line of the graph (average of the first 50 points)
        baseline = np.mean(y_data[:50])
        half_max = baseline + (exact_y_max - baseline) / 2.0

        # Uncertainty in half_max (assuming baseline error is negligible due to averaging)
        err_half_max = err_y_max / 2.0

        # Find crossings
        signs = np.sign(y_data - half_max)
        crossings = np.where(np.diff(signs))[0]

        left_crossings = crossings[crossings < main_peak_idx]
        right_crossings = crossings[crossings >= main_peak_idx]

        if len(left_crossings) > 0 and len(right_crossings) > 0:
            left_idx = left_crossings[-1]
            right_idx = right_crossings[0]

            # Interpolation for lambda 1
            slope_left = (y_data[left_idx + 1] - y_data[left_idx]) / (x_data[left_idx + 1] - x_data[left_idx])
            x_left = x_data[left_idx] + (half_max - y_data[left_idx]) / slope_left
            err_x_left = err_half_max / abs(slope_left)  # Error propagation for intersection

            # Interpolation for lambda 2
            slope_right = (y_data[right_idx + 1] - y_data[right_idx]) / (x_data[right_idx + 1] - x_data[right_idx])
            x_right = x_data[right_idx] + (half_max - y_data[right_idx]) / slope_right
            err_x_right = err_half_max / abs(slope_right)  # Error propagation for intersection

            # FWHM and its final uncertainty
            fwhm = x_right - x_left
            err_fwhm = np.sqrt(err_x_left ** 2 + err_x_right ** 2)

            # Print results for our convenience
            print("--- Analysis Results with Uncertainties ---")
            print(f"Maxima: λ = {exact_x_max:.3f} nm, Amplitude = ({exact_y_max:.3f} ± {err_y_max:.3f})")
            print(f"Baseline Amplitude = {baseline:.3f}")
            print(f"Half Maximum = ({half_max:.3f} ± {err_half_max:.3f})")
            print(f"Left Intersect (λ1) = ({x_left:.3f} ± {err_x_left:.3f}) nm")
            print(f"Right Intersect (λ2) = ({x_right:.3f} ± {err_x_right:.3f}) nm")
            print(f"Calculated FWHM (Δλ) = ({fwhm:.3f} ± {err_fwhm:.3f}) nm")

            # Plotting
            plt.figure(figsize=(12, 7))

            plt.plot(x_data, y_data, color='blue', marker='.', linestyle='-', linewidth=1, markersize=5, alpha=0.4,
                     label='Data Points')
            plt.hlines(half_max, x_left, x_right, color='green', linestyle='--', linewidth=2,
                       label=f'FWHM = {fwhm:.2f} ± {err_fwhm:.2f} [nm]')
            plt.plot(exact_x_max, exact_y_max, marker='o', color='red', markeredgecolor='black', markersize=8,
                     linestyle='none', label='Maxima')

            plt.vlines([x_left, x_right], ymin=baseline, ymax=half_max, color='green', linestyle=':', alpha=0.7)

            y_text_pos = baseline - (half_max - baseline) * 0.07
            plt.text(x_left + 10, y_text_pos, rf'$\lambda_1={x_left:.2f}$ nm', verticalalignment='center',
                     horizontalalignment='right', color='green', fontsize=10)
            plt.text(x_right - 10, y_text_pos, rf'$\lambda_2={x_right:.2f}$ nm', verticalalignment='center',
                     horizontalalignment='left', color='green', fontsize=10)

            plt.title('Amplitude VS Wave length of Green LED', fontsize=14)
            plt.xlabel('Wave length [nm]', fontsize=12)
            plt.ylabel('Amplitude [a.u.]', fontsize=12)
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