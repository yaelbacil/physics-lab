import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as stats

# calling data from github reposatory
exp1 = "https://raw.githubusercontent.com/yaelbacil/physics-lab/refs/heads/main/exp1.csv"
df_exp1 = pd.read_csv(exp1)

# fit and plot for exp1

def cos_offset_model(theta_rad, A, B):
    return A * np.cos(theta_rad) + B

theta_raw = pd.to_numeric(df_exp1['theta'], errors='coerce').to_numpy()
theta_rad = np.deg2rad(theta_raw)
E = pd.to_numeric(df_exp1['E'], errors='coerce').to_numpy()
E_err = 0.1 # uncertainty in E
sigma_E = np.full_like(E, E_err, dtype=float)
p0 = [0.5 * (np.nanmax(E) - np.nanmin(E)), np.nanmean(E)] # Initial guess

popt, pcov = curve_fit(cos_offset_model, theta_rad, E, p0=p0,
                       sigma=sigma_E, absolute_sigma=True)

A_fit, B_fit = popt
A_fit_err, B_fit_err = np.sqrt(np.diag(pcov))

# Chi-square and reduced chi-square
E_model = cos_offset_model(theta_rad, A_fit, B_fit)
residuals = E - E_model
chi2_val = np.sum((residuals / sigma_E) ** 2)

dof = len(E) - len(popt)  # number of data points - number of fit parameters
reduced_chi2 = chi2_val / dof

print(f'A = {A_fit:.5f} ± {A_fit_err:.5f}')
print(f'B = {B_fit:.5f} ± {B_fit_err:.5f}')
print(f'reduced chi^2 = {reduced_chi2:.3f}')

# Plot data + fit
theta_line = np.linspace(theta_rad.min(), theta_rad.max(), 500)
E_line = cos_offset_model(theta_line, A_fit, B_fit)

plt.figure(figsize=(7, 4.5))
plt.errorbar(theta_rad, E, yerr=sigma_E, fmt='o', capsize=3, label='Data')
plt.plot(theta_line, E_line, '-', color='r',label=r'Fit: $E=A\cos(\theta)+B$')

fit_text = (
    f'A = {A_fit:.3f} ± {A_fit_err:.3f}\n'
    f'B = {B_fit:.3f} ± {B_fit_err:.3f}\n'
    f'reduced $\\chi^2$ = {reduced_chi2:.2f}'
)
plt.text(0.03, 0.2, fit_text, transform=plt.gca().transAxes, va='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

plt.xlabel(r'$\theta (rad)$')
plt.ylabel('E (V)')
plt.title(r'$Amplitude\ vs\ \theta$')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
