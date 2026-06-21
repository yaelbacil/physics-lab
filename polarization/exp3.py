import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# data of the first board
first_board_url = "https://raw.githubusercontent.com/yaelbacil/physics-lab/main/polarization/exp3_first_board.csv"
first_board_df = pd.read_csv(first_board_url)
I1 = pd.to_numeric(first_board_df['I'], errors='coerce').to_numpy()
I1_err = 0.01
theta1 = pd.to_numeric(first_board_df['deg'], errors='coerce').to_numpy()
theta_rad1 = np.deg2rad(theta1)
theta_err1 = np.deg2rad(2)

# data of the second board
second_board_url = "https://raw.githubusercontent.com/yaelbacil/physics-lab/main/polarization/exp3_second_board.csv"
second_board_df = pd.read_csv(second_board_url)
I2 = pd.to_numeric(second_board_df['I'], errors='coerce').to_numpy()
I2_err = 0.01
theta2 = pd.to_numeric(second_board_df['deg'], errors='coerce').to_numpy()
theta_rad2 = np.deg2rad(theta2)
theta_err2 = np.deg2rad(2)

# define the Malus's law function for fitting the first board
def malus_law(theta_rad1, I0, theta0, c):
    return I0 * np.cos((theta_rad1 - theta0)) ** 2 + c

# initial parameter guess for the first board
index_of_max_I1 = np.argmax(I1) # maximum intensity index
theta_guess1 = theta_rad1[index_of_max_I1]
parameters_initial_guess1 = [max(I1) - min(I1), theta_guess1, min(I1)]
popt1, pcov1 = curve_fit(malus_law, theta_rad1, I1, p0=parameters_initial_guess1,sigma=np.full_like(I1, I1_err), absolute_sigma=True)
I0_opt, theta0_opt, c_opt = popt1
I0_err, theta0_err, c_err = np.sqrt(np.diag(pcov1))

# define a constant function for fitting the second board
def constant_fit(theta_rad2, m):
    return m * np.ones_like(theta_rad2)

# initial parameter guess for the second board (constant fit)
parameters_initial_guess_const = [np.mean(I2)]
popt_const, pcov_const = curve_fit(constant_fit, theta_rad2, I2, p0=parameters_initial_guess_const, sigma=np.full_like(I2, I2_err), absolute_sigma=True)
m_opt = popt_const[0]
m_err = np.sqrt(np.diag(pcov_const))[0]

# define the Malus's law function for fitting the second board
def quarter_malus_law(theta_rad2, A, theta_0, B):
    return A * (np.cos(theta_rad2 - theta_0)) ** 2 + B

# initial parameter guess for the second board
index_of_max_I2 = np.argmax(I2) # maximum intensity index
theta_guess2 = theta_rad2[index_of_max_I2]
parameters_initial_guess2 = [max(I2) - min(I2), theta_guess2, min(I2)]
popt2, pcov2 = curve_fit(quarter_malus_law, theta_rad2, I2, p0=parameters_initial_guess2, sigma=np.full_like(I2, I2_err), absolute_sigma=True)
A_opt, theta_0_opt, B_opt = popt2
A_err, theta_0_err, B_err = np.sqrt(np.diag(pcov2))

# chi-squared calculations for the first board
I_model1 = malus_law(theta_rad1, I0_opt, theta0_opt, c_opt)
chi_squared1 = np.sum(((I_model1 - I1)/I1_err)**2)
dof1 = len(I1) - 3
reduced_chi_squared1 = chi_squared1 / dof1

# chi-squared calculations for the second board - constant model
I_model_const = constant_fit(theta_rad2, m_opt)
chi_squared_const = np.sum(((I_model_const - I2)/I2_err)**2)
dof_const = len(I2) - 1
reduced_chi_squared_const = chi_squared_const / dof_const

# chi-squared calculations for the second board - Malus's law
I_model2 = quarter_malus_law(theta_rad2, A_opt, theta_0_opt, B_opt)
chi_squared2 = np.sum(((I_model2 - I2)/I2_err)**2)
dof2 = len(I2) - 3
reduced_chi_squared2 = chi_squared2 / dof2

# plot for the first board
theta_fit1 = np.linspace(min(theta_rad1), max(theta_rad1), 500)
I_fit1 = malus_law(theta_fit1, I0_opt,theta0_opt, c_opt)
plt.figure(figsize=(10, 5))
plt.errorbar(theta_rad1, I1,xerr=theta_err1, yerr=I1_err, fmt='o',color='blue', capsize=3, label='Data')
plt.plot(theta_fit1, I_fit1, color='red', linestyle='-', label='Fit')
plt.title(r"Intensity VS Degree - first board - Malus's law Fit", fontsize=15)
plt.xlabel(r'$\theta$ [rad]', fontsize=13)
plt.ylabel('I [lux]', fontsize=13)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend(fontsize=12)
text_str = '\n'.join((
    fr'$I_0 = {I0_opt:.5f} \pm {I0_err:.5f}$ lux',
    fr'$c = {c_opt:.5f} \pm {c_err:.5f}$ lux',
    fr'$\theta_0 = {theta0_opt:.4f} \pm {theta0_err:.4f}$ rad = ${np.rad2deg(theta0_opt):.2f}^\circ \pm {np.rad2deg(theta0_err):.2f}^\circ$',
    fr'$\chi^2 = {reduced_chi_squared1:.2f}$'
))
plt.text(0.02, 0.05, text_str, transform=plt.gca().transAxes, fontsize=12,
         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
plt.show()

# plot for the second board - constant fit
theta_fit_const = np.linspace(min(theta_rad2), max(theta_rad2), 500)
I_fit_const = constant_fit(theta_fit_const, m_opt)

plt.figure(figsize=(10, 5))
plt.errorbar(theta_rad2, I2, xerr=theta_err2, yerr=I2_err, fmt='o', color='blue', capsize=3, label='Data')
plt.plot(theta_fit_const, I_fit_const, color='red', linestyle='-', label='Fit')
plt.title(r"Intensity VS Degree - second board - Constant Fit", fontsize=15)
plt.xlabel(r'$\theta$ [rad]', fontsize=13)
plt.ylabel('I [lux]', fontsize=13)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend(fontsize=12, loc='upper left')

text_str_const = '\n'.join((
    fr'$m = {m_opt:.3f} \pm {m_err:.3f}$ lux',
    fr'$\chi^2 = {reduced_chi_squared_const:.2f}$'
))

plt.text(0.98, 0.04, text_str_const, transform=plt.gca().transAxes, fontsize=10,
         horizontalalignment='right', verticalalignment='bottom', multialignment='left',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
plt.show()

# plot for the second board - Malus's law
theta_fit2 = np.linspace(min(theta_rad2), max(theta_rad2), 500)
I_fit2 = quarter_malus_law(theta_fit2, A_opt, theta_0_opt, B_opt)
plt.figure(figsize=(10, 5))
plt.errorbar(theta_rad2, I2,xerr=theta_err2, yerr=I2_err, fmt='o',color='blue', capsize=3, label='Data')
plt.plot(theta_fit2, I_fit2, color='red', linestyle='-', label='Fit')
plt.title(r"Intensity VS Degree - second board - Malus's law Fit", fontsize=15)
plt.xlabel(r'$\theta$ [rad]', fontsize=13)
plt.ylabel('I [lux]', fontsize=13)
plt.grid(True, which="both", ls="--", alpha=0.6)
plt.legend(fontsize=12)
text_str = '\n'.join((
    fr'$I_0 = {A_opt:.5f} \pm {A_err:.5f}$ lux',
    fr'$B = {B_opt:.5f} \pm {B_err:.5f}$ lux',
    fr'$\theta_0 = {theta_0_opt:.4f} \pm {theta_0_err:.4f}$ rad = ${np.rad2deg(theta_0_opt):.2f}^\circ \pm {np.rad2deg(theta_0_err):.2f}^\circ$',
    fr'$\chi^2 = {reduced_chi_squared2:.2f}$'
))
plt.text(0.98, 0.04, text_str, transform=plt.gca().transAxes, fontsize=10, horizontalalignment='right',
         verticalalignment='bottom', multialignment='left', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
plt.show()