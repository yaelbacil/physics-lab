import matplotlib.pyplot as plt
import numpy as np
from scipy.odr import ODR, Model, RealData

x = np.deg2rad(np.array([140, 150, 160, 170, 180,190,200,210,216,220,226,230,236])-50)
y = np.array([0, 4, 8, 17, 28,39,49,58,59,63,64,63,63])


x_err = np.deg2rad(np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,2,2]))
y_err = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,1,1])

def model(x, A):
    return A*np.cos(x)**2

p0 = [0.1] # Initial guess for A

odr_model = Model(lambda beta, x_vals: model(x_vals, beta[0]))
data = RealData(x, y, sx=x_err, sy=y_err)
odr = ODR(data, odr_model, beta0=p0)
out = odr.run()

A_fit = out.beta[0]
A_unc = out.sd_beta[0]

x_fit = np.linspace(x.min() - 0.5, x.max() + 0.5, 400)
y_fit = model(x_fit, A_fit)

residuals = y - model(x, A_fit)
chi2 = np.sum((residuals / y_err) ** 2)
dof = len(y) - len(out.beta)
red_chi2 = chi2 / dof

fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(np.rad2deg(x), y, yerr=y_err, xerr=x_err, fmt='o', capsize=3, color='blue', label='Data')
ax.plot(np.rad2deg(x_fit), y_fit, color='red', label=r'Fit: $I = A\cos^2(\theta)$')
ax.set_xlabel('theta [deg]')
ax.set_ylabel('Intensity[lux]')
ax.set_title('Intensity vs deg')
ax.grid(True, alpha=0.3)
ax.text(
    0.024,
    0.73,
    f'$A$ = {A_fit:.3g} ± {A_unc:.3g} [lux]\n'
    f'reduced $\\chi^2$ = {red_chi2:.3g}',
    transform=ax.transAxes,
    va='bottom',
    bbox=dict(facecolor='white', alpha=0.9, edgecolor='black')
)
ax.legend(loc='best')

plt.tight_layout()
plt.show()

fig, ax2 = plt.subplots(figsize=(8, 8))
ax2.set_xlabel('Intensity [lux]')
ax2.set_ylabel('Intensity[lux]')
ax2.set_title('Intensity in polar coordinates')
ax2.grid(True, alpha=0.3)
deg = np.linspace(0, 2*np.pi, 400)
r_full = model(deg, A_fit)
polarX = r_full*np.cos(deg)
polarY = r_full*np.sin(deg)

data_polarX = y * np.cos(x)
data_polarY = y * np.sin(x)
ax2.plot(data_polarX, data_polarY, 'o', color='blue', label='Data')

ax2.plot(polarX, polarY, color='red', label=r'Fit: $I = A\cos^2(\theta)(cos(\theta),sin(\theta))$')
ax2.legend()
ax2.axis('equal')
plt.show()