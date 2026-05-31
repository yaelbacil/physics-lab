import matplotlib.pyplot as plt
import numpy as np
from scipy.odr import ODR, Model, RealData

# cosmic ray from exp1.py
COSMIC_RAY_PER_SEC = 0.5222
COSMIC_RAY_PER_SEC_ERR = 0.07617

# Pb
x = np.array([2.25, 5, 7, 10, 13]) # width, mm
rew_N = np.array([1692, 1170, 984, 841, 670])
t = 180

N = rew_N / t - COSMIC_RAY_PER_SEC # counts per second, without cosmic ray
N_err = np.sqrt((np.sqrt(rew_N) / t) ** 2 + COSMIC_RAY_PER_SEC_ERR ** 2)
x_err = np.array([0.02, 1, 1, 1, 1]) # mm

def model(x, mu, N0):
    return N0 * np.exp(-mu * x)

p0 = (0.1, N.max()) # Initial guess for mu and N0

odr_model = Model(lambda beta, x_vals: model(x_vals, beta[0], beta[1]))
data = RealData(x, N, sx=x_err, sy=N_err)
odr = ODR(data, odr_model, beta0=p0)
out = odr.run()

mu_fit, N0_fit = out.beta
mu_unc, N0_unc = out.sd_beta

x_fit = np.linspace(x.min() - 0.5, x.max() + 0.5, 400)
N_fit = model(x_fit, mu_fit, N0_fit)

residuals = N - model(x, mu_fit, N0_fit)
chi2 = np.sum((residuals / N_err) ** 2)
dof = len(N) - len(out.beta)
red_chi2 = chi2 / dof

print(f"mu = {mu_fit:.6g} ± {mu_unc:.6g} mm^-1")
print(f"N0 = {N0_fit:.6g} ± {N0_unc:.6g} 1/s")
print(f"chi^2 = {chi2:.6g}")
print(f"reduced chi^2 = {red_chi2:.6g}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(x, N, yerr=N_err, xerr=x_err, fmt='o', capsize=3, color='blue', label='Data')
ax.plot(x_fit, N_fit, color='red', label=r'Fit: $N = N_0 e^{-\mu x}$')
ax.set_xlabel('x (mm)')
ax.set_ylabel('N (1/s)')
ax.set_title('N vs x - Lead (Pb)')
ax.grid(True, alpha=0.3)
ax.legend()
ax.text(
    0.03,
    0.03,
    f'$\\mu$ = {mu_fit:.3g} ± {mu_unc:.3g} (mm$^{{-1}}$)\n'
    f'$N_0$ = {N0_fit:.3g} ± {N0_unc:.3g} (1/s)\n'
    f'reduced $\\chi^2$ = {red_chi2:.3g}',
    transform=ax.transAxes,
    va='bottom',
    bbox=dict(facecolor='white', alpha=0.9, edgecolor='black')
)

plt.tight_layout()
plt.show()
