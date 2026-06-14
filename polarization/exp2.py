import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def model_R_pi(theta, n):
    theta_t = np.arcsin(np.sin(theta) / n)
    return (np.tan(theta - theta_t) ** 2) / (np.tan(theta + theta_t) ** 2)


# calling data from GitHub repository (raw link)
pi_data = 'https://raw.githubusercontent.com/yaelbacil/physics-lab/main/polarization/exp2_pi_with_glass.csv'
df_pi = pd.read_csv(pi_data)

theta_raw = pd.to_numeric(df_pi['deg'], errors='coerce').to_numpy()
theta_rad = np.deg2rad(theta_raw)  # convert to radian
theta_rad_err = np.deg2rad(1.0)

I0_pi = 231.0
I0_pi_err = 1.0
I_err = 0.01

R_pi = pd.to_numeric(df_pi['R_pi'], errors='coerce').to_numpy()
# propagate simple intensity errors (keep existing form)
R_pi_err = np.sqrt((I_err / I0_pi) ** 2 + (R_pi * I0_pi_err / I0_pi) ** 2)

# Use R_pi_err as the sigma for fitting
p0 = [1.5]  # initial guess for refractive index
popt, pcov = curve_fit(model_R_pi, theta_rad, R_pi, p0=p0, sigma=R_pi_err, absolute_sigma=True, maxfev=10000)

perr = np.sqrt(np.diag(pcov))

# propagate x-axis (angle) errors into y using derivative: sigma_eff^2 = sigma_y^2 + (dR/dtheta * sigma_theta)^2
# numerical derivative of model w.r.t theta at measured points
h = 1e-6
dR_dtheta = (model_R_pi(theta_rad + h, *popt) - model_R_pi(theta_rad - h, *popt)) / (2 * h)
sigma_eff = np.sqrt(R_pi_err ** 2 + (dR_dtheta * theta_rad_err) ** 2)

# goodness of fit using effective sigma including x-errors
residuals = (R_pi - model_R_pi(theta_rad, *popt))
chi2 = np.sum((residuals / sigma_eff) ** 2)
dof = len(theta_rad) - len(popt)
red_chi2 = chi2 / dof if dof > 0 else np.nan

print(f"Fitted refractive index n = {popt[0]:.4f} ± {perr[0]:.4f}")
print(f"chi2/dof = {red_chi2:.2f} ({chi2:.2f}/{dof})")

# plot data and fit
theta_deg = theta_raw
theta_fit_deg = np.linspace(np.nanmin(theta_deg), np.nanmax(theta_deg), 400)
theta_fit_rad = np.deg2rad(theta_fit_deg)
fit_R = model_R_pi(theta_fit_rad, *popt)

plt.errorbar(theta_rad, R_pi, xerr=theta_rad_err, yerr=R_pi_err, fmt='o', label='data')
plt.plot(theta_fit_rad, fit_R, '-', color="red" ,label=f'n={popt[0]:.3f}±{perr[0]:.3f}')
plt.xlabel('theta (rad)')
plt.ylabel('R')
plt.title("R vs theta")
plt.legend()
plt.grid(True)
plt.show()