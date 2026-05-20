import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#Load Data from GitHub
data_url = 'https://raw.githubusercontent.com/yaelbacil/physics-lab/refs/heads/main/interferometry/interferometer%20data.csv'
df_data_url = pd.read_csv(data_url)

#taking only the relevant columns for m and alpha
df_exp2 = df_data_url[['exp 2 ', 'Unnamed: 7']].copy()
df_exp2.columns = ['m', 'alpha_deg']
df_exp2 = df_exp2.iloc[2:].dropna()
m = pd.to_numeric(df_exp2['m']).values
alpha_deg = pd.to_numeric(df_exp2['alpha_deg']).values

# Uncertainty is +/- 2 rings for all measurements
m_err = np.full(len(m), 2.0)

# --- 2. Data Conversion ---
# Convert degrees to radians, then square it for the x-axis
alpha_rad = np.radians(alpha_deg)
x_data = alpha_rad**2
y_data = m
y_err = m_err

# --- 3. Linear Fit ---
# Define a linear model y = a*x + b
def linear_model(x, a, b):
    return a * x + b

# Perform the fit using curve_fit
# absolute_sigma=True ensures the covariance matrix pcov reflects the true y_err
popt, pcov = curve_fit(linear_model, x_data, y_data, sigma=y_err, absolute_sigma=True)
a, b = popt
err_a, err_b = np.sqrt(np.diag(pcov))

# --- 4. Reduced Chi-Squared Calculation ---
# Calculate the expected y values based on our fit
y_fit = linear_model(x_data, *popt)

# Calculate residuals (difference between measured and fitted data)
residuals = y_data - y_fit

# Calculate Chi-Squared
chi_squared = np.sum((residuals / y_err)**2)

# Degrees of freedom = (Number of data points) - (Number of fitted parameters)
dof = len(x_data) - len(popt)

# Calculate Reduced Chi-Squared
reduced_chi_squared = chi_squared / dof

# --- 5. Output Results to Console ---
print("--- Fit Results ---")
print(f"Slope (a): {a:.2f} +/- {err_a:.2f}")
print(f"Intercept (b): {b:.2f} +/- {err_b:.2f}")
print(f"Chi-Squared: {chi_squared:.2f}")
print(f"Degrees of Freedom: {dof}")
print(f"Reduced Chi-Squared: {reduced_chi_squared:.2f}")

# --- 6. Plotting ---
plt.figure(figsize=(10, 6))

# Plot the raw data points with error bars
plt.errorbar(x_data, y_data, yerr=y_err, fmt='o', color='blue',
             markeredgecolor='black', capsize=5, label='Data points (+/- 2 rings)')

# Plot the fitted line
# Create a smooth line for the x-axis starting from 0
x_fit_line = np.linspace(0, max(x_data) * 1.1, 100)
plt.plot(x_fit_line, linear_model(x_fit_line, a, b), color='red', linestyle='--',
         label=f'Linear Fit: y = {a:.0f}x + {b:.1f}\n$\\chi^2_{{red}}$ = {reduced_chi_squared:.2f}')

# Formatting the plot
plt.title('Number of Rings vs. Angle Squared', fontsize=14)
plt.xlabel(r'$\alpha^2$ [rad$^2$]', fontsize=12)
plt.ylabel('Number of Rings (m)', fontsize=12)
plt.grid(True, alpha=0.4)
plt.legend(loc='upper left')
plt.tight_layout()

# Show the plot
plt.show()