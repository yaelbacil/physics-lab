import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

delta_t = 15.0      # Measurement time per interval in seconds
bg_rate = 0.52      # Background rate in cps
bg_err = 0.07       # Error in background rate

# Getting the barium data from GitHub
csv_url = "https://raw.githubusercontent.com/yaelbacil/physics-lab/main/nuclear%20radiation/nuclear%20data/Barium%20exp.csv"
df = pd.read_csv(csv_url)
indices = df['index'].values
counts_raw = df['counts'].values

# Time axis (assuming index 1 corresponds to t=0)
time_axis = (indices - 1) * delta_t

# The rates and their uncertainties after background subtraction
rate_raw = counts_raw / delta_t
err_rate_raw = np.sqrt(counts_raw) / delta_t
rate_neto = rate_raw - bg_rate
err_rate_net = np.sqrt(err_rate_raw**2 + bg_err**2)

# Filter out non-positive rates to avoid issues with log scale and fitting
valid_indices = rate_neto > 0
t_data = time_axis[valid_indices]
R_data = rate_neto[valid_indices]
R_err = err_rate_net[valid_indices]

# Define the exponential decay function
def decay_function(t, R0, lambd):
    return R0 * np.exp(-lambd * t)
p0 = [R_data[0], 0.0045]
popt, pcov = curve_fit(decay_function, t_data, R_data, p0=p0, sigma=R_err, absolute_sigma=True)
R0_fit, lambd_fit = popt
R0_err, lambd_err = np.sqrt(np.diag(pcov))

# Calculate Half-life and its uncertainty
T_half = np.log(2) / lambd_fit
T_half_err = T_half * (lambd_err / lambd_fit)

# Calculate Reduced Chi-Squared to evaluate fit quality
expected_R = decay_function(t_data, *popt)
chi_squared = np.sum(((R_data - expected_R) / R_err)**2)
dof = len(t_data) - len(popt)
reduced_chi_squared = chi_squared / dof if dof > 0 else np.nan

# Plotting
t_fit = np.linspace(0, max(t_data), 200)
R_fit = decay_function(t_fit, *popt)
plt.figure(figsize=(9, 7))
plt.errorbar(t_data, R_data, yerr=R_err, fmt='o', color='blue', capsize=4, label='Data', zorder=5)
plt.plot(t_fit, R_fit, color='red', linestyle='-', label='Fit')
plt.title(r'$^{137}_{56}$Ba Radioactive Decay - Exponential Fit', fontsize=15)
plt.xlabel('Time [second]', fontsize=13)
plt.ylabel('Neto Rate [1/second]', fontsize=13)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend(fontsize=12)
text_str = '\n'.join((
    r'$\lambda = %.5f \pm %.5f \text{ sec}^{-1}$' % (lambd_fit, lambd_err),
    r'$\chi^2 = %.2f$' % (reduced_chi_squared)
))

plt.text(0.05, 0.05, text_str, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.show()

# Print results
print(f"Decay Constant (lambda) = {lambd_fit:.5f} +/- {lambd_err:.5f} s^-1")
print(f"Half-life (T_1/2) = {T_half:.1f} +/- {T_half_err:.1f} s")
print(f"Reduced Chi-Squared = {reduced_chi_squared:.2f}")