import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root_scalar

def model_R_pi(theta, n):
    theta_t = np.arcsin(np.sin(theta) / n)
    return (np.tan(theta - theta_t) ** 2) / (np.tan(theta + theta_t) ** 2)

def model_R_sigma(theta, n):
    theta_t = np.arcsin(np.sin(theta) / n)
    return (np.sin(theta - theta_t) ** 2) / (np.sin(theta + theta_t) ** 2)

def extrapolate_zero(n, n_err):
    R0 = ((n - 1) / (n + 1)) ** 2
    dR0_dn = 4 * (n - 1) / (n + 1) ** 3
    R0_err = abs(dR0_dn) * n_err
    return R0, R0_err

def calculate_chi2(theta_rad, R, R_err, model, popt, theta_rad_err):
    h = 1e-6
    dR_dtheta = (model(theta_rad + h, *popt) - model(theta_rad - h, *popt)) / (2 * h)
    unc_eff = np.sqrt(R_err ** 2 + (dR_dtheta * theta_rad_err) ** 2)
    residuals = R - model(theta_rad, *popt)
    chi2 = np.sum((residuals / unc_eff) ** 2)
    dof = len(theta_rad) - len(popt)
    red_chi2 = chi2 / dof if dof > 0 else np.nan
    return chi2, dof, red_chi2, unc_eff


def find_brewster(n, n_err):
    theta_b = np.arctan(n)
    theta_b_err = abs(n_err) / (1 + n ** 2)

    # numerical root of theta - arcsin(sin(theta)/n) = 0
    def f(t):
        return t - np.arcsin(np.sin(t) / n)

    try:
        sol = root_scalar(f, bracket=[1e-6, np.pi / 2 - 1e-6], method='bisect')
        theta_b_num = sol.root if sol.converged else theta_b
    except Exception:
        theta_b_num = theta_b

    return theta_b, theta_b_err, theta_b_num


# calling data from GitHub repository (raw link)
pi_data = 'https://raw.githubusercontent.com/yaelbacil/physics-lab/main/polarization/exp2_pi_with_glass.csv'
sigma_data = 'https://raw.githubusercontent.com/yaelbacil/physics-lab/refs/heads/main/polarization/exp2_sigma_with_glass.csv'

df_pi = pd.read_csv(pi_data)
df_sigma = pd.read_csv(sigma_data)

theta_pi_raw = pd.to_numeric(df_pi['deg'], errors='coerce').to_numpy()
theta_pi_rad = np.deg2rad(theta_pi_raw)  # convert to radian

theta_sigma_raw = pd.to_numeric(df_sigma['deg'], errors='coerce').to_numpy()
theta_sigma_rad = np.deg2rad(theta_sigma_raw)  # convert to radian

theta_rad_err = np.deg2rad(1.0)

I0_pi = 231.0
I0_sigma = 233.0

I0_err = 1.0
I_err = 0.01

R_pi = pd.to_numeric(df_pi['R_pi'], errors='coerce').to_numpy()
R_sigma = pd.to_numeric(df_sigma['R_sigma'], errors='coerce').to_numpy()

R_pi_err = np.sqrt((I_err / I0_pi) ** 2 + (R_pi * I0_err / I0_pi) ** 2)
R_sigma_err = np.sqrt((I_err / I0_sigma) ** 2 + (R_sigma * I0_err / I0_sigma) ** 2)

print(R_sigma_err)

p0 = [1.5]  # initial guess for refractive index
popt_pi, pcov_pi = curve_fit(model_R_pi, theta_pi_rad, R_pi, p0=p0, sigma=R_pi_err, absolute_sigma=True, maxfev=10000)
popt_sigma, pcov_sigma = curve_fit(model_R_sigma, theta_sigma_rad, R_sigma, p0=p0, sigma=R_sigma_err, absolute_sigma=True, maxfev=10000)

perr_pi = np.sqrt(np.diag(pcov_pi))
perr_sigma = np.sqrt(np.diag(pcov_sigma))

chi2_pi, dof_pi, red_chi2_pi, unc_eff_pi = calculate_chi2(
    theta_pi_rad,
    R_pi,
    R_pi_err,
    model_R_pi,
    popt_pi,
    theta_rad_err,
)

chi2_sigma, dof_sigma, red_chi2_sigma, unc_eff_sigma = calculate_chi2(
    theta_sigma_rad,
    R_sigma,
    R_sigma_err,
    model_R_sigma,
    popt_sigma,
    theta_rad_err,
)

print(f"Fitted refractive index n = {popt_pi[0]:.4f} ± {perr_pi[0]:.4f}")
print(f"chi2/dof = {red_chi2_pi:.2f} ({chi2_pi:.2f}/{dof_pi})")
print(f"sigma chi2/dof = {red_chi2_sigma:.2f} ({chi2_sigma:.2f}/{dof_sigma})")

# plot data and fit
theta_deg = theta_pi_raw
theta_fit_deg = np.linspace(0.1, 70, 400)
theta_fit_rad = np.deg2rad(theta_fit_deg)
fit_R = model_R_pi(theta_fit_rad, *popt_pi)

R0, R0_err = extrapolate_zero(popt_pi[0], perr_pi[0])

# compute Brewster angle from fitted n (analytic) and numeric check
theta_b, theta_b_err, theta_b_num = find_brewster(popt_pi[0], perr_pi[0])
theta_b_deg = np.rad2deg(theta_b)
theta_b_err_deg = np.rad2deg(theta_b_err)
print(f"Brewster angle (deg) = {theta_b_deg:.3f} ± {theta_b_err_deg:.3f}")

plt.errorbar(theta_pi_rad, R_pi, xerr=theta_rad_err, yerr=R_pi_err, fmt='o', label='data')
plt.plot(theta_fit_rad, fit_R, '-', color="red" ,label='fit')
plt.plot(theta_fit_rad, model_R_pi(theta_fit_rad, 1.52), '-', color='green', label='theoretical line')
plt.errorbar([0], [R0], yerr=[R0_err], fmt='s', color='black', zorder=5, label='extrapolation')
plt.axvline(theta_b, color='magenta', linestyle='--', label=f'Brewster {theta_b_deg:.2f}°')
plt.xlabel('theta (rad)')
plt.ylabel('R')
plt.title("R vs theta - pi plane")
plt.text(
    0.05,
    0.05,
    f"n = {popt_pi[0]:.3f} ± {perr_pi[0]:.3f}\nreduced $\\chi^2$ = {red_chi2_pi:.2f}\nR(0) = {R0:.5f} ± {R0_err:.5f}",
    transform=plt.gca().transAxes,
    va='bottom',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
)
plt.legend()
plt.grid(True)
plt.show()

# plot sigma data and fit
theta_sigma_fit_deg = np.linspace(0.1, 70, 400)
theta_sigma_fit_rad = np.deg2rad(theta_sigma_fit_deg)
fit_R_sigma = model_R_sigma(theta_sigma_fit_rad, *popt_sigma)

R0_sigma, R0_sigma_err = extrapolate_zero(popt_sigma[0], perr_sigma[0])

plt.errorbar(theta_sigma_rad, R_sigma, xerr=theta_rad_err, yerr=R_sigma_err, fmt='o', label='data')
plt.plot(theta_sigma_fit_rad, fit_R_sigma, '-', color="red", label='fit')
plt.plot(theta_fit_rad, model_R_sigma(theta_fit_rad, 1.52), '-', color='green', label='theoretical line')
plt.errorbar([0], [R0_sigma], yerr=[R0_sigma_err], fmt='s', color='black', zorder=5, label='extrapolation')
plt.xlabel('theta (rad)')
plt.ylabel('R')
plt.title("R vs theta - sigma plane")
plt.text(
    0.02,
    0.58,
    f"n = {popt_sigma[0]:.3f} ± {perr_sigma[0]:.3f}\nreduced $\\chi^2$ = {red_chi2_sigma:.2f}\nR(0) = {R0_sigma:.5f} ± {R0_sigma_err:.5f}",
    transform=plt.gca().transAxes,
    va='bottom',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
)
plt.legend(loc='upper left')
plt.grid(True)
plt.show()