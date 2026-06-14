import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# cosmic ray from exp1.py
COSMIC_RAY_PER_SEC = 0.5222
COSMIC_RAY_PER_SEC_ERR = 0.07617

# gamma ray data
d_gamma = np.array([1.5, 2.5, 4.5, 8.5, 12.5, 16.5, 20.5]) # cm
rew_N_gamma = np.array([24780, 15833, 8546, 3894, 2096, 1744, 1216])
t_gamma = np.array([180, 180, 180, 180, 180, 240, 240])

N_gamma = rew_N_gamma / t_gamma - COSMIC_RAY_PER_SEC # counts per second, without cosmic ray

N_err_gamma = np.sqrt((np.sqrt(rew_N_gamma) / t_gamma) ** 2 + COSMIC_RAY_PER_SEC_ERR ** 2)
d_err_gamma = 0.1 # cm

# gamma + beta ray data
d_gamma_beta = np.array([1.5, 2.5, 4.5, 8.5, 12.5, 16.5])
rew_N_gamma_beta = np.array([839319, 472844, 205056, 76697, 38977, 22495])
t_beta_gamma = 180 # sec

N_gamma_beta = rew_N_gamma_beta / t_beta_gamma - COSMIC_RAY_PER_SEC # counts per second, without cosmic ray

N_err_gamma_beta = np.sqrt((np.sqrt(rew_N_gamma_beta) / t_beta_gamma) ** 2 + COSMIC_RAY_PER_SEC_ERR ** 2)
d_err_gamma_beta = 0.1 # cm

def model(d, c, R0):
    return c / ((d + R0) ** 2)

def plot_and_fit(d, N, N_err, d_err, model, title):
    p0 = (N.max() * (d.min() ** 2), 0.0) # Initial guess for c and R0

    popt, pcov = curve_fit(model, d, N, sigma=N_err, absolute_sigma=True, p0=p0, maxfev=10000)
    c_fit, R0_fit = popt
    c_unc, R0_unc = np.sqrt(np.diag(pcov))

    d_fit = np.linspace(d.min() - 0.5, d.max() + 0.5, 400)
    N_fit = model(d_fit, c_fit, R0_fit)

    residuals = N - model(d, c_fit, R0_fit)
    chi2 = np.sum((residuals / N_err) ** 2)
    dof = len(N) - len(popt)
    red_chi2 = chi2 / dof

    print(f"c = {c_fit:.6g} ± {c_unc:.6g}")
    print(f"R0 = {R0_fit:.6g} ± {R0_unc:.6g} cm")
    print(f"reduced chi^2 = {red_chi2:.6g}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(d, N, yerr=N_err, xerr=d_err, fmt='o', capsize=3, color='blue', label='Data')
    ax.plot(d_fit, N_fit, color='red', label=r'Fit: $\frac{c}{(d + R_0)^2}$')
    ax.set_xlabel('distance [cm]')
    ax.set_ylabel('CPS')
    ax.set_title('CPS vs Distance - ' + title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.text(
        0.03,
        0.97,
        f'c = {c_fit:.2f} ± {c_unc:.4g} [$cm^2$]\n$R_0$ = {R0_fit:.4g} ± {R0_unc:.4g} [$cm$]\nreduced $\\chi^2$ = {red_chi2:.3g}',
        transform=ax.transAxes,
        va='top',
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='black')
    )

    plt.tight_layout()
    plt.show()
    
plot_and_fit(d_gamma, N_gamma, N_err_gamma, d_err_gamma, model, 'Gamma ray')
plot_and_fit(d_gamma_beta, N_gamma_beta, N_err_gamma_beta, d_err_gamma_beta, model, 'Gamma + Beta ray')