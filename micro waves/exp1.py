import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.stats as stats

def get_experiment_data(df, start_col, end_col):
  # the function gets the relevant coloumns
  # returns specific sub_experiment data from the csv file 'data micro'
  exp_df = df.iloc[1:, start_col:end_col].copy()
  exp_df.columns = exp_df.iloc[0]
  exp_df = exp_df.iloc[1:].dropna(how='all').reset_index(drop=True)
  return exp_df

# calling data from github reposatory
# m1 - meusrment number 1
exp1 = "https://raw.githubusercontent.com/yaelbacil/physics-lab/refs/heads/main/exp1.csv"
exp3 = 'https://raw.githubusercontent.com/yaelbacil/physics-lab/refs/heads/main/exp%203%20mesure%201.csv'
exp4_m1 = 'https://raw.githubusercontent.com/yaelbacil/physics-lab/refs/heads/main/exp%204%20mesure%201.csv'
exp4_m2 = 'https://raw.githubusercontent.com/yaelbacil/physics-lab/refs/heads/main/exp%204%20mesure%202.csv'
micro_data_path = "https://raw.githubusercontent.com/yaelbacil/physics-lab/refs/heads/main/data%20micro.csv"

# creating dataframes
df_exp1 = pd.read_csv(exp1)
df_exp3 = pd.read_csv(exp3)
df_exp4_m1 = pd.read_csv(exp4_m1)
df_exp4_m2 = pd.read_csv(exp4_m2)
df_full = pd.read_csv(micro_data_path)

# calling subexperiments from 'data micro' file
df_exp2 = get_experiment_data(df_full, 4, 7)
df_exp5 = get_experiment_data(df_full, 8, 10)
# fit and plot for exp1

def cos_offset_model(theta_rad, A, B):
    return A * np.cos(theta_rad) + B

theta_raw = pd.to_numeric(df_exp1['theta'], errors='coerce').to_numpy()
theta_rad = np.deg2rad(theta_raw)
E = pd.to_numeric(df_exp1['E'], errors='coerce').to_numpy()
E_err = 0.1 # uncertainty in E
sigma_E = np.full_like(E, E_err, dtype=float)
p0 = [0.5 * (np.nanmax(E) - np.nanmin(E)), np.nanmean(E)] # Initial guess

popt, pcov = curve_fit(cos_offset_model, theta_rad, E, p0=p0,
                       sigma=sigma_E, absolute_sigma=True)

A_fit, B_fit = popt
A_fit_err, B_fit_err = np.sqrt(np.diag(pcov))

# Chi-square and reduced chi-square
E_model = cos_offset_model(theta_rad, A_fit, B_fit)
residuals = E - E_model
chi2_val = np.sum((residuals / sigma_E) ** 2)

dof = len(E) - len(popt)  # number of data points - number of fit parameters
reduced_chi2 = chi2_val / dof

print(f'A = {A_fit:.5f} ± {A_fit_err:.5f}')
print(f'B = {B_fit:.5f} ± {B_fit_err:.5f}')
print(f'reduced chi^2 = {reduced_chi2:.3f}')

# Plot data + fit
theta_line = np.linspace(theta_rad.min(), theta_rad.max(), 500)
E_line = cos_offset_model(theta_line, A_fit, B_fit)

plt.figure(figsize=(7, 4.5))
plt.errorbar(theta_rad, E, yerr=sigma_E, fmt='o', capsize=3, label='Data')
plt.plot(theta_line, E_line, '-', color='r',label=r'Fit: $E=A\cos(\theta)+B$')

fit_text = (
    f'A = {A_fit:.3f} ± {A_fit_err:.3f}\n'
    f'B = {B_fit:.3f} ± {B_fit_err:.3f}\n'
    f'reduced $\\chi^2$ = {reduced_chi2:.2f}'
)
plt.text(0.03, 0.2, fit_text, transform=plt.gca().transAxes, va='top',
    bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

plt.xlabel(r'$\theta (rad)$')
plt.ylabel('E (V)')
plt.title(r'$Amplitude\ vs\ \theta$')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()