import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.special import gammaln  # פונקציה יעילה לחישוב (x!)ln
from scipy.stats import poisson

# --- 1. קריאת הנתונים וסידורם ---
file_path = 'nuclear data/puason.csv'
df = pd.read_csv(file_path)
distribution = df['counts'].value_counts().sort_index()

x_data = distribution.index.values  # מספר הקריאות (k)
f_data = distribution.values  # שכיחות כל קריאה (כמה פעמים הופיע k)
total_measurements = np.sum(f_data)
normalized_f_data = f_data / total_measurements

# --- 2. הגדרת פונקציית הלוג-נראות (Log-Likelihood) ---
# הערה: רוב ספריות האופטימיזציה יודעות *למזער* פונקציות.
# לכן, כדי למצוא מקסימום, אנחנו פשוט נחזיר את ה*מינוס* של הלוג-נראות ונקטין אותו.
def negative_log_likelihood(lam_array):
    lam = lam_array[0]

    # למבדה חייבת להיות מספר חיובי, נחזיר אינסוף אם האלגוריתם מנסה לבדוק מספר שלילי
    if lam <= 0:
        return np.inf

        # חישוב הלוג-נראות לכל נקודה: f * (x * ln(lam) - lam - ln(x!))
    # משתמשים ב-gammaln(x+1) שמקביל מתמטית ל-ln(x!) כדי למנוע קריסה במספרים ענקיים
    log_likelihoods = f_data * (x_data * np.log(lam) - lam - gammaln(x_data + 1))

    # סכום כל הערכים (והפיכה למינוס כדי שהאלגוריתם ימזער)
    return -np.sum(log_likelihoods)


def negative_log_likelihood_gaussian(params):
    # הפעם הפונקציה מקבלת מערך של שני פרמטרים שאותם האלגוריתם ינסה למטב
    mu = params[0]
    sigma = params[1]

    # סטיית תקן חייבת להיות מספר חיובי ממש.
    # אם האלגוריתם מנסה להציב ערך שלילי או אפס, נחזיר "אינסוף" כדי להרחיק אותו משם.
    if sigma <= 0:
        return np.inf

        # חישוב הלוג-נראות (Log-Likelihood) של התפלגות גאוס לכל נקודה:
    # ln(L) = -ln(sigma) - 0.5*ln(2*pi) - ((x - mu)^2) / (2 * sigma^2)
    log_likelihoods = f_data * (-np.log(sigma) - 0.5 * np.log(2 * np.pi) - ((x_data - mu) ** 2) / (2 * sigma ** 2))

    # מחזירים את המינוס של סכום הלוג-נראות כדי שהאלגוריתם יוכל למזער את הערך
    return -np.sum(log_likelihoods)

# --- 3. הפעלת אלגוריתם האופטימיזציה לחקר המקסימום ---
# נתחיל מניחוש ראשוני הגיוני (למשל הערך 1) וניתן לאלגוריתם לטפס לשיא
initial_guess = [1.0]

# שימוש באלגוריתם אופטימיזציה נפוץ של Scipy
result = minimize(negative_log_likelihood, initial_guess, method='Nelder-Mead')
optimal_lambda = result.x[0]

# חישוב הממוצע האנליטי הרגיל רק לשם השוואה
analytical_mean = np.average(x_data, weights=f_data)

analytical_mean = np.average(x_data, weights=f_data)

variance_actual = np.average((x_data - analytical_mean)**2, weights=f_data)

# חישוב סטיית התקן (Standard Deviation - סיגמא) - עם שורש
std_dev = np.sqrt(variance_actual)

print(f"Analytical Mean (Expected): {analytical_mean:.6f}")
print(f"Actual Variance (Should equal Mean): {variance_actual:.5f}")
print(f"Standard Deviation (Error): {std_dev:.5f}")



x_theoretical = np.arange(0, max(x_data) + 1)
# נחשב את ההסתברות של פואסון עבור הלמבדה שלנו
y_theoretical = poisson.pmf(x_theoretical, optimal_lambda)

# --- 4. ציור הגרף ---
plt.figure(figsize=(10, 6))

# ציור ההיסטוגרמה המנורמלת של הנתונים
plt.bar(x_data, normalized_f_data, color='lightblue', edgecolor='black', width=0.8, alpha=0.7, label='Measured Data (Normalized)')

# ציור התפלגות פואסון התיאורטית על גבי הברים
plt.plot(x_theoretical, y_theoretical, 'o--', color='black', linewidth=2, markersize=8, label=f'Poisson Fit ($\lambda$ = {optimal_lambda:.5f})')

# עיצוב הגרף
plt.title('Poisson Fit on Measured Data Histogram', fontsize=16)
plt.xlabel('Number of Reads (k)', fontsize=14)
plt.ylabel('Probability / Relative Frequency', fontsize=14)
plt.xticks(x_theoretical) # מציג את כל המספרים השלמים על ציר ה-X
plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
lambda_vals = np.linspace(optimal_lambda * 0.5, optimal_lambda * 1.5, 200)

# נחשב את הלוג-נראות (בלי המינוס) עבור כל נקודה כדי לראות את השיא
ll_vals = [-negative_log_likelihood([l]) for l in lambda_vals]

fig, ax = plt.subplots()
ticks = list(ax.get_xticks()) # לקיחת הקפיצות הנוכחיות
ticks.append(optimal_lambda)          # הוספת הערך שלך לרשימה
ax.set_xticks(ticks)
# ציור עקומת הנראות
plt.plot(lambda_vals, ll_vals, color='purple', linewidth=2.5, label='Log-Likelihood Function')
plt.title('Log-Likelihood Function', fontsize=16)
plt.xlabel('$\lambda$', fontsize=14)
plt.ylabel('Log(L)', fontsize=14)
# סימון קו אנכי בנקודת המקסימום
plt.axvline(optimal_lambda, color='red', linestyle='dashed',
            label=f'Maximum Likelihood ($\lambda$={optimal_lambda:.2f})')
plt.show()