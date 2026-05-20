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

# Taking only the relevant columns for m and P
df_exp3 = df_data_url[['exp 3', 'Unnamed: 10']].copy()
df_exp3.columns = ['m', 'P']
df_exp3 = df_exp3.iloc[2:].dropna()
m = pd.to_numeric(df_exp3['m']).values
P = pd.to_numeric(df_exp3['P']).values

# Uncertainties
m_err = np.full(len(m), 2.0)
P_err = np.full(len(P), 2.0)

x_data = 760 - P # Convert pressure to Delta P
y_data = m
x_err = P_err
y_err = m_err

# Define a linear model y = a*x + b
def linear_model(x, a, b):
    return a * x + b
popt, pcov = curve_fit(linear_model, x_data, y_data, sigma=y_err, absolute_sigma=True)
a, b = popt
err_a, err_b = np.sqrt(np.diag(pcov))

# Refractive Index Calculation
d = 0.1
wavelength = 532e-9
r = (wavelength * 760) / (2 * d)
n = a * r + 1
err_n = r * err_a

# Reduced Chi-Squared Calculation
y_fit = linear_model(x_data, *popt)
residuals = y_data - y_fit
chi_squared = np.sum((residuals / y_err)**2)
dof = len(x_data) - len(popt)
reduced_chi_squared = chi_squared / dof

# Plotting
plt.figure(figsize=(10, 6))
plt.errorbar(x_data, y_data, xerr=x_err, yerr=y_err, fmt='o', color='red', capsize=5, label='Data')
# x_fit_line = np.linspace(0, max(x_data) * 1.1, 100)
x_fit_line = np.linspace(min(x_data) * 0.9, max(x_data) * 1.05, 100)
plt.plot(x_fit_line, linear_model(x_fit_line, a, b), color='blue', linestyle='-',
         label=f'Fit')
text_str = f"a = {a:.3f} $\\pm$ {err_a:.3f}\nb = {b:.3f} $\\pm$ {err_b:.3f}\n$\\chi^2 / ndof$ = {reduced_chi_squared:.2f}"
props = dict(boxstyle='round', facecolor='white', alpha=0.8)
plt.text(0.78, 0.17, text_str, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='top', bbox=props)

plt.title('Number of Rings VS Pressure Difference - linear fit', fontsize=14)
plt.xlabel(r'$\Delta P$ [mmHg]', fontsize=12)
plt.ylabel('Number of Rings', fontsize=12)
plt.grid(True, alpha=0.4)
plt.legend(loc='upper left')
plt.tight_layout()

plt.show()

# Results
print("--- Fit Results ---")
print(f"Slope (a): {a:.4f} +/- {err_a:.4f}")
print(f"Intercept (b): {b:.4f} +/- {err_b:.4f}")
print(f"Reduced Chi-Squared: {reduced_chi_squared:.2f}")
print(f"Refractive Index (n): {n:.4f} +/- {err_n:.4f}")