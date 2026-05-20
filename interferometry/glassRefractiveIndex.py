import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load the data from GitHub
data_url = 'https://raw.githubusercontent.com/yaelbacil/physics-lab/refs/heads/main/interferometry/interferometer%20data.csv'

# Added an option to load from local file in case it doesnt work from the Git
local_file = 'interferometer data.csv'
try:
    import urllib.request
    with urllib.request.urlopen(data_url, timeout=5) as response:
        df_data_url = pd.read_csv(response)

except Exception as e:
    df_data_url = pd.read_csv(local_file)

#taking only the relevant columns for m and alpha
df_exp2 = df_data_url[['exp 2 ', 'Unnamed: 7']].copy()
df_exp2.columns = ['m', 'alpha_deg']
df_exp2 = df_exp2.iloc[2:].dropna()
m = pd.to_numeric(df_exp2['m']).values
alpha_deg = pd.to_numeric(df_exp2['alpha_deg']).values
alpha_rad = np.radians(alpha_deg)

# Uncertainties
m_err = np.full(len(m), 2.0)
delta_alpha_rad = np.radians(0.01)


x_data = alpha_rad**2
y_data = m
y_err = m_err
x_err = 2 * alpha_rad * delta_alpha_rad

# Define a linear model y = a*x + b
def linear_model(x, a, b):
    return a * x + b

# Perform the fit using curve_fit
# absolute_sigma=True ensures the covariance matrix pcov reflects the true y_err
popt, pcov = curve_fit(linear_model, x_data, y_data, sigma=y_err, absolute_sigma=True)
a, b = popt
err_a, err_b = np.sqrt(np.diag(pcov))

# Refractive Index Calculation
d = 0.001
wavelength = 532e-9
K = d / wavelength
n = K / (K - a)
err_n = (K / ((K - a)**2)) * err_a

# Reduced Chi-Squared Calculation
y_fit = linear_model(x_data, *popt)
residuals = y_data - y_fit
chi_squared = np.sum((residuals / y_err)**2)
dof = len(x_data) - len(popt)
reduced_chi_squared = chi_squared / dof

# Results
print(f"alpha_rad: {alpha_rad}")
print(delta_alpha_rad)
print("--- Fit Results ---")
print(f"Slope (a): {a:.2f} +/- {err_a:.2f}")
print(f"Intercept (b): {b:.2f} +/- {err_b:.2f}")
print(f"Reduced Chi-Squared: {reduced_chi_squared:.2f}")
print(f"Refractive Index (n): {n:.4f} +/- {err_n:.4f}")

# Plotting
plt.figure(figsize=(10, 6))
plt.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, fmt='o', color='red', capsize=5, label='Data')
x_fit_line = np.linspace(0, max(x_data) * 1.1, 100)
plt.plot(x_fit_line, linear_model(x_fit_line, a, b), color='blue', linestyle='-',
         label=f'Fit')
text_str = f"a = {a:.2f} $\\pm$ {err_a:.2f}\nb = {b:.2f} $\\pm$ {err_b:.2f}\n$\\chi^2 / ndof$ = {reduced_chi_squared:.2f}"
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
plt.text(0.78, 0.17, text_str, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=props)

plt.title('Number of Rings VS Angle Squared - linear fit', fontsize=14)
plt.xlabel(r'$\alpha^2$ [rad$^2$]', fontsize=12)
plt.ylabel('Number of Rings', fontsize=12)
plt.grid(True, alpha=0.4)
plt.legend(loc='upper left')
plt.tight_layout()

plt.show()

# Calculation with another d which seems to be more accurate
d_corrected = 0.004
K_corrected = d_corrected / wavelength
n_corrected = K_corrected / (K_corrected - a)
err_n_corrected = (K_corrected / ((K_corrected - a)**2)) * err_a
print(f"Corrected Refractive Index (n): {n_corrected:.4f} +/- {err_n_corrected:.4f}")