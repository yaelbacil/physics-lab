import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

rate_bg = 0.52
err_bg = 0.07
rate_gamma_net = 86.7
err_gamma_net = 0.9

x_data = np.array([0.0, 0.17, 0.76, 1.96])
N_data = np.array([315374, 118137, 52062, 12531])
t_data = np.array([120, 60, 60, 60])

rates_raw = N_data / t_data
err_rates_raw = np.sqrt(N_data) / t_data

rates_total_net = rates_raw - rate_bg
err_rates_total_net = np.sqrt(err_rates_raw**2 + err_bg**2)

rates_beta = rates_total_net - rate_gamma_net
err_rates_beta = np.sqrt(err_rates_total_net**2 + err_gamma_net**2)

# Linearization: ln(R) = ln(R0) - mu * x
ln_rates_beta = np.log(rates_beta)

# Uncertainty propagation for ln
err_ln_rates_beta = err_rates_beta / rates_beta

# Define a linear fit y = mx + b
def linear_function(x, m, b):
    return m * x + b

# Initial guess for m,b
p0 = [-1.5, 7.8]

# Perform weighted linear fit
popt, pcov = curve_fit(linear_function, x_data, ln_rates_beta,
                       p0=p0, sigma=err_ln_rates_beta, absolute_sigma=True)

m_fit, n_fit = popt
perr = np.sqrt(np.diag(pcov))
m_err, n_err = perr

# Extract physical parameters
mu = -m_fit
mu_err = m_err
R0 = np.exp(n_fit)
R0_err = R0 * n_err

# Chi-Squared Calculation
expected_ln_rates = linear_function(x_data, *popt)
chi_squared = np.sum(((ln_rates_beta - expected_ln_rates) / err_ln_rates_beta)**2)
dof = len(x_data) - len(popt)
reduced_chi_squared = chi_squared / dof if dof > 0 else np.nan

# Plotting
x_fit = np.linspace(0, max(x_data) * 1.1, 200)
y_fit = linear_function(x_fit, *popt)
plt.figure(figsize=(9, 7))
plt.errorbar(x_data, ln_rates_beta, yerr=err_ln_rates_beta, fmt='o', color='blue',
             capsize=4, label=r'Data', zorder=5)
plt.plot(x_fit, y_fit, color='red', linestyle='-', label='Fit')
plt.title('Beta Attenuation - Linearization', fontsize=14)
plt.xlabel('Absorber Thickness $x$ [mm]', fontsize=12)
plt.ylabel(r'$\ln(R_\beta)$', fontsize=12)
plt.minorticks_off()
plt.grid(True, which='major', linestyle='--', linewidth=0.8, color='gray', alpha=0.6)
plt.legend(fontsize=11)
text_str = '\n'.join((
    r'$m$ = $%.3f \pm %.3f \text{ mm}^{-1}$' % (m_fit, m_err),
    r'$b$ = $%.3f \pm %.3f$' % (n_fit, n_err),
    r'$\mu = -m = %.3f \pm %.3f \text{ mm}^{-1}$' % (mu, mu_err),
    r'$R_{\beta,0} = e^b = %.1f \pm %.1f \text{ sec}^{-1}$' % (R0, R0_err),
    r'$\chi^2 = %.2f$' % (reduced_chi_squared)
))

plt.text(0.05, 0.05, text_str, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

plt.show()