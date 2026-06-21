import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import scipy.constants as const

def B(I, L, N):
    return const.mu_0 * N * I / L

def B_err(I_err, L, N):
    return const.mu_0 * N * I_err / L

def linear(x, a, b):
    return a * x + b

I = np.array([0, 1.02, 1.98, 3.03, 4, 4.9])
theta_deg = np.array([65, 70, 72, 77, 80, 84])

I_err = np.full_like(I, 0.01, dtype=float) # [A]
theta_deg_err = np.full_like(theta_deg, 2.0, dtype=float)

L = 10e-2 # m
N = 2508

B_values = B(I, L, N)
B_values_err = B_err(I_err, L, N)

for i in range(len(B_values)):
    print(f'{B_values[i]:.4f} ± {B_values_err[i]:.4f}')

p0 = np.polyfit(B_values, theta_deg, 1)

popt, pcov = curve_fit(
    linear,
    B_values,
    theta_deg,
    p0=p0,
    sigma=theta_deg_err,
    absolute_sigma=True,
)

a_opt, b_opt = popt
a_err, b_err = np.sqrt(np.diag(pcov))

theta_fit = linear(B_values, a_opt, b_opt)
chi_squared = np.sum(((theta_deg - theta_fit) / theta_deg_err) ** 2)
dof = len(theta_deg) - len(popt)
reduced_chi_squared = chi_squared / dof

B_fit = np.linspace(B_values.min(), B_values.max(), 500)
theta_fit_curve = linear(B_fit, a_opt, b_opt)

plt.figure(figsize=(10, 6))
plt.errorbar(
    B_values,
    theta_deg,
    xerr=B_values_err,
    yerr=theta_deg_err,
    fmt='o',
    color='blue',
    capsize=3,
    label='Data',
)
plt.plot(B_fit, theta_fit_curve, color='red', linestyle='-', label='linear fit')
plt.xlabel('magnetic field [T]')
plt.ylabel('angle [deg]')
plt.title('angle vs magnetic field - linear fit')
plt.grid(True, which='both', ls='--', alpha=0.6)
plt.legend()

text_str = '\n'.join((
    fr'$a = {a_opt:.3f} \pm {a_err:.3f}$ deg/T',
    fr'$b = {b_opt:.3f} \pm {b_err:.3f}$ deg',
    fr'reduced $\chi^2 = {reduced_chi_squared:.2f}$',
))
plt.text(
    0.02,
    0.98,
    text_str,
    transform=plt.gca().transAxes,
    fontsize=11,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
)

plt.tight_layout()
plt.show()

print('Fit results:')
print(f'a = {a_opt:.6e} ± {a_err:.6e} deg/T')
print(f'b = {b_opt:.6f} ± {b_err:.6f} deg')
print(f'chi^2 = {chi_squared:.3f}')
print(f'reduced chi^2 = {reduced_chi_squared:.3f}')
