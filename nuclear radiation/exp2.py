import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# cosmic ray from exp1.py
COSMIC_RAY_PER_SEC = 0.5222
COSMIC_RAY_PER_SEC_ERR = 0.07617

d = np.array([1.5, 2.5, 4.5, 8.5, 12.5, 16.5, 20.5]) # cm
rew_N = np.array([24780, 15833, 8546, 3894, 2096, 1744, 1216])
t = np.array([180, 180, 180, 180, 180, 240, 240])

N = rew_N / t - COSMIC_RAY_PER_SEC # counts per second, without cosmic ray

N_err = np.sqrt((np.sqrt(rew_N) / t) ** 2 + COSMIC_RAY_PER_SEC_ERR ** 2)
d_err = 0.1 # cm

def model(d, c, R0):
    return c / ((d + R0) ** 2)


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
ax.set_ylabel('N')
ax.set_title('Counts per second vs Distance')
ax.grid(True, alpha=0.3)
ax.legend()
ax.text(
    0.03,
    0.97,
    f'c = {c_fit:.2f} ± {c_unc:.4g} [$cm^2$]\n$R_0$ = {R0_fit:.4g} ± {R0_unc:.4g} [cm]\nreduced $\\chi^2$ = {red_chi2:.3g}',
    transform=ax.transAxes,
    va='top',
    bbox=dict(facecolor='white', alpha=0.9, edgecolor='black')
)

plt.tight_layout()
plt.show()