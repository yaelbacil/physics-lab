import matplotlib.pyplot as plt
import numpy as np
from scipy.odr import ODR, Model, RealData

# cosmic ray from exp1.py
COSMIC_RAY_PER_SEC = 0.5222
COSMIC_RAY_PER_SEC_ERR = 0.07617

def model_gamma_ray(x, mu, N0):
    return N0 * np.exp(-mu * x)

def model_beta_gamma_ray(x, a, b, N0, N1):
    return N0 * np.exp(-a * x) + N1 * np.exp(-b * x)

def plot_graph(x, N, N_err, x_err, model, title):
    if model == model_gamma_ray:
        p0 = (0.1, N.max()) 
        odr_model = Model(lambda beta, x_vals: model(x_vals, beta[0], beta[1]))
    elif model == model_beta_gamma_ray:
        p0 = (44, 0.1, 5, 20) 
        odr_model = Model(lambda beta, x_vals: model(x_vals, beta[0], beta[1], beta[2], beta[3]))

    data = RealData(x, N, sx=x_err, sy=N_err)
    odr = ODR(data, odr_model, beta0=p0)
    out = odr.run()

    if model == model_gamma_ray:
        mu_fit, N0_fit = out.beta
        mu_unc, N0_unc = out.sd_beta
    elif model == model_beta_gamma_ray:
        a_fit, b_fit, N0_fit, N1_fit = out.beta
        a_unc, b_unc, N0_unc, N1_unc = out.sd_beta

    x_fit = np.linspace(0, x.max() + 0.05, 500)
    N_fit = model(x_fit, *out.beta)

    residuals = N - model(x, *out.beta)
    chi2 = np.sum((residuals / N_err) ** 2)
    dof = len(N) - len(out.beta)
    red_chi2 = chi2 / dof

    print(f"\n--- {title} ---")
    if model == model_gamma_ray:
        print(f"mu = {mu_fit:.6g} ± {mu_unc:.6g} cm^-1")
        print(f"N0 = {N0_fit:.6g} ± {N0_unc:.6g}")
    elif model == model_beta_gamma_ray:
        print(f"a (beta decay)  = {a_fit:.6g} ± {a_unc:.6g} cm^-1")
        print(f"b (gamma decay) = {b_fit:.6g} ± {b_unc:.6g} cm^-1")
        print(f"N0 (beta amp)   = {N0_fit:.6g} ± {N0_unc:.6g}")
        print(f"N1 (gamma amp)  = {N1_fit:.6g} ± {N1_unc:.6g}")
    print(f"chi^2 = {chi2:.6g}, reduced chi^2 = {red_chi2:.6g}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, N, yerr=N_err, xerr=x_err, fmt='o', capsize=3, color='blue', label='Data Points', zorder=3)
    
    if model == model_gamma_ray:
        fit_label = r'Fit: $N = N_0 e^{-\mu x}$'
    else:
        fit_label = r'Fit: $N = N_0 e^{-a x} + N_1 e^{-b x}$'
        
    ax.plot(x_fit, N_fit, color='red', linestyle='-', linewidth=2, label=fit_label, zorder=2)
    
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('CPS')
    ax.set_title("CPS vs x, " + title, fontsize=12)
    ax.grid(True, which="both", linestyle='--', alpha=0.5)
    
    ax.set_ylim(N.min() - 1, N.max() + 1)
    ax.set_xlim(-0.01, x.max() + 0.05)

    if model == model_gamma_ray:
        info_text = (
            f'$\\mu$ = {mu_fit:.3g} ± {mu_unc:.3g} [cm$^{{-1}}$]\n'
            f'$N_0$ = {N0_fit:.3g} ± {N0_unc:.3g}\n'
            f'reduced $\\chi^2$ = {red_chi2:.3g}'
        )
    else:
        info_text = (
            f'$a$ = {a_fit:.3g} ± {a_unc:.3g} [cm$^{{-1}}$]\n'
            f'$b$ = {b_fit:.3g} ± {b_unc:.3g} [cm$^{{-1}}$]\n'
            f'$N_0$ = {N0_fit:.3g} ± {N0_unc:.3f}\n'
            f'$N_1$ = {N1_fit:.3g} ± {N1_unc:.3g}\n'
            f'reduced $\\chi^2$ = {red_chi2:.3g}'
        )
        
    ax.text(
        0.03, 0.05,
        info_text,
        transform=ax.transAxes,
        va='bottom', ha='left',
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.5')
    )

    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

# Pb - gamma ray
x_Pb = np.array([0.225, 0.5, 0.7, 1.0, 1.3]) # width, cm
rew_N_Pb = np.array([1692, 1170, 984, 841, 670])
t_Pb = 180 # sec
N = rew_N_Pb / t_Pb - COSMIC_RAY_PER_SEC # counts per second, without cosmic ray
N_err_Pb = np.sqrt((np.sqrt(rew_N_Pb) / t_Pb) ** 2 + COSMIC_RAY_PER_SEC_ERR ** 2)
x_err_Pb = np.array([0.002, 0.1, 0.1, 0.1, 0.1]) # cm
plot_graph(x_Pb, N, N_err_Pb, x_err_Pb, model_gamma_ray, 'Pb - gamma ray')

# Cu - beta + gamma ray
d_Cu = np.array([0.016, 0.03, 0.05, 0.16, 0.29, 0.5]) # width, cm
rew_N_Cu = np.array([4111, 3965, 3692, 3628, 3553, 3477])
t_Cu = 180 # sec
N = rew_N_Cu / t_Cu - COSMIC_RAY_PER_SEC # counts per second, without cosmic ray
N_err_Cu = np.sqrt((np.sqrt(rew_N_Cu) / t_Cu) ** 2 + COSMIC_RAY_PER_SEC_ERR ** 2)
d_err_Cu = np.array([0.002, 0.002, 0.002, 0.002, 0.002, 0.1]) # cm
plot_graph(d_Cu, N, N_err_Cu, d_err_Cu, model_beta_gamma_ray, 'Cu - beta + gamma ray')