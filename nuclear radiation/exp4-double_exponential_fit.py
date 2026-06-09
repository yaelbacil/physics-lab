import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Background radiation data [1/sec]
rate_bg = 0.52
err_bg = 0.07

# Initial Gamma net rate from the measurement with the beta blocker
B = 86.7

# Absorber thicknesses [mm] - Including x=0 (no absorber)
x_data = np.array([0.0, 0.53, 0.76, 1.96, 3.16])

# Total counts (gamma + beta)
N_data = np.array([315374, 139119, 52062, 12531, 6891])

# Measurement times [sec]
t_data = np.array([120, 60, 60, 60, 120])

# Calculate raw rates [1/sec]
rates_raw = N_data / t_data

# Calculate errors for raw rates: err = sqrt(N) / t
err_rates_raw = np.sqrt(N_data) / t_data

# Calculate Neto Total Rates (subtracting background)
rates_net = rates_raw - rate_bg

# Error propagation for net rates: sqrt((err_raw)^2 + (err_bg)^2)
err_rates_net = np.sqrt(err_rates_raw**2 + err_bg**2)

# Curve Fitting (Double Exponential with B)

# Define the double exponential model with Gamma amplitude B
def double_exponential_fixed_B(x, A, mu_beta, mu_gamma):
    return A * np.exp(-mu_beta * x) + B * np.exp(-mu_gamma * x)

# Initial guess [A, mu_beta, mu_gamma]
p0 = [2540, 3.5, 0.05]

# Bounds to prevent unphysical negative parameters
lower_bounds = [0, 0, 0]
upper_bounds = [np.inf, np.inf, np.inf]

# Perform weighted fit using the calculated errors (sigma)
popt, pcov = curve_fit(double_exponential_fixed_B, x_data, rates_net,
                       p0=p0, sigma=err_rates_net, absolute_sigma=True,
                       bounds=(lower_bounds, upper_bounds),
                       maxfev=10000)

# Extract fitted parameters and their 1-sigma errors
A_fit, mu_beta_fit, mu_gamma_fit = popt
perr = np.sqrt(np.diag(pcov))
A_err, mu_beta_err, mu_gamma_err = perr

# Chi-Squared Calculation
expected_rates = double_exponential_fixed_B(x_data, *popt)
chi_squared = np.sum(((rates_net - expected_rates) / err_rates_net)**2)
dof = len(x_data) - len(popt)
reduced_chi_squared = chi_squared / dof if dof > 0 else np.nan

print("--- Fit Results ---")
print(f"Beta:  A = {A_fit:.1f} +/- {A_err:.1f} [1/sec], mu_beta  = {mu_beta_fit:.3f} +/- {mu_beta_err:.3f} [1/mm]")
print(f"Gamma: B = {B:.1f} [1/sec], mu_gamma = {mu_gamma_fit:.3f} +/- {mu_gamma_err:.3f} [1/mm]")
print(f"Chi-Squared = {chi_squared:.2f}")
print(f"DOF = {dof}")
print(f"Reduced Chi-Squared = {reduced_chi_squared:.2f}")

# Plotting
x_fit = np.linspace(0, max(x_data) * 1.1, 200)
y_fit = double_exponential_fixed_B(x_fit, *popt)
plt.figure(figsize=(9, 7))
plt.errorbar(x_data, rates_net, yerr=err_rates_net, fmt='o', color='blue',
             capsize=4, label='Data', zorder=5)
plt.plot(x_fit, y_fit, color='red', linestyle='-', label='Fit')
plt.title(r'Total $\beta$ and $\gamma$ Rate VS Polyethylene Thickness - Double Exponential Fit', fontsize=14)
plt.xlabel('Absorber Thickness $x$ [mm]', fontsize=12)
plt.ylabel('Total Neto Rate [1/seconds]', fontsize=12)
plt.yscale('log')
plt.minorticks_off()
plt.grid(True, which='major', linestyle='-', linewidth=0.8, color='gray', alpha=0.6)
plt.legend(fontsize=11)
text_str = '\n'.join((
r'$A = %.1f \pm %.1f \text{ sec}^{-1}$' % (A_fit, A_err),
    r'$\mu_\beta = %.3f \pm %.3f \text{ mm}^{-1}$' % (mu_beta_fit, mu_beta_err),
    r'$\mu_\gamma = %.3f \pm %.3f \text{ mm}^{-1}$' % (mu_gamma_fit, mu_gamma_err),
    r'$\chi^2 = %.2f$' % (reduced_chi_squared)
))

plt.text(0.05, 0.025, text_str, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.show()