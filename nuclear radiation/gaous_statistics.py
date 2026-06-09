import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

# --- 1. קריאת הנתונים וחישוב השכיחויות ---
file_path = 'nuclear data/gous.csv'
df = pd.read_csv(file_path)

distribution = df['counts'].value_counts().sort_index()
x_data = distribution.index.values
f_data = distribution.values

# נרמול הנתונים כדי שציר ה-Y ייצג הסתברות
total_measurements = np.sum(f_data)
normalized_f_data = f_data / total_measurements

# --- 2. מציאת הפרמטרים בעזרת נראות מקסימלית (MLE) ---
def negative_log_likelihood_gaussian(params):
    mu = params[0]
    if mu <= 0:
        return np.inf
    sigma = np.sqrt(mu)
    log_likelihoods = f_data * (-np.log(sigma) - 0.5 * np.log(2 * np.pi) - ((x_data - mu)**2) / (2 * sigma**2))
    return -np.sum(log_likelihoods)

# ניחוש התחלתי מתוך הנתונים עצמם
mean_guess = np.average(x_data, weights=f_data)

result_gaussian = minimize(negative_log_likelihood_gaussian, [mean_guess], method='Nelder-Mead')
optimal_lambda = result_gaussian.x[0]
optimal_sigma = np.sqrt(optimal_lambda)

# --- 3. ציור הגרף המשולב ---
plt.figure(figsize=(10, 6))

# ציור ההיסטוגרמה של הנתונים (עמודות תכלת)
plt.bar(x_data, normalized_f_data, color='lightblue', edgecolor='black', width=0.8, alpha=0.7, label='Measured Data')

# יצירת ציר X רציף לציור הפעמון הגאוסי (עם קצת שוליים בצדדים)
x_theoretical = np.linspace(min(x_data) - 2, max(x_data) + 2, 200)

# חישוב ערכי הגאוס התיאורטיים (PDF - Probability Density Function)
y_theoretical = norm.pdf(x_theoretical, optimal_lambda, optimal_sigma)

# ציור התפלגות גאוס (קו כתום רציף)
plt.plot(x_theoretical, y_theoretical, color='black', linewidth=2.5,
         label=f'Gaussian Fit\n($\mu $={optimal_lambda:.3f}')

# עיצוב הגרף
plt.title('Gaussian Fit on Measured Data Histogram', fontsize=16)
plt.xlabel('Number of Reads (k)', fontsize=14)
plt.ylabel('Probability / Relative Frequency', fontsize=14)
analytical_mean = np.average(x_data, weights=f_data)

variance_actual = np.average((x_data - analytical_mean)**2, weights=f_data)

# חישוב סטיית התקן (Standard Deviation - סיגמא) - עם שורש
std_dev = np.sqrt(variance_actual)

print(f"Analytical Mean (Expected): {analytical_mean:.5f}")
print(f"Actual Variance (Should equal Mean): {variance_actual:.5f}")
print(f"Standard Deviation (Error): {std_dev:.5f}")
# מציג את המספרים השלמים על ציר ה-X לנוחות הקריאה
plt.xticks(range(int(min(x_data)), int(max(x_data)) + 1))
plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


# נחשב את הלוג-נראות (בלי המינוס) עבור כל נקודה כדי לראות את השיא
lambda_vals = np.linspace(optimal_lambda * 0.5, optimal_lambda * 1.5, 200)
ll_vals = [-negative_log_likelihood_gaussian([l]) for l in lambda_vals]

fig, ax = plt.subplots()
ticks = list(ax.get_xticks()) # לקיחת הקפיצות הנוכחיות
ticks.append(optimal_lambda)          # הוספת הערך שלך לרשימה
ax.set_xticks(ticks)
# ציור עקומת הנראות
plt.plot(lambda_vals, ll_vals, color='purple', linewidth=2.5, label='Log-Likelihood Function')
plt.title('Log-Likelihood Function', fontsize=16)
plt.xlabel('$\mu $', fontsize=14)
plt.ylabel('Log(L)', fontsize=14)
# סימון קו אנכי בנקודת המקסימום
plt.axvline(optimal_lambda, color='red', linestyle='dashed',
            label=f'Maximum Likelihood ($\lambda$={optimal_lambda:.2f})')
plt.show()