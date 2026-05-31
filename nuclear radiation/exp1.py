import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# calling data from github repository
volt_check = r"https://raw.githubusercontent.com/yaelbacil/physics-lab/refs/heads/main/nuclear%20radiation/nuclear%20data/VoltCheck.csv"
cosmic_ray = r"https://raw.githubusercontent.com/yaelbacil/physics-lab/refs/heads/main/nuclear%20radiation/nuclear%20data/cosmic%20ray.csv"

df_volt_check = pd.read_csv(volt_check)
df_cosmic_ray = pd.read_csv(cosmic_ray)

counts_volt_check = df_volt_check['counts'].dropna().astype(float)
err_volt_check = np.sqrt(counts_volt_check)

# plot the particle counts against the operating voltage with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(df_volt_check['Volt'], counts_volt_check, yerr=err_volt_check, fmt='o-', color='blue', ecolor='red', capsize=3, label='data')
plt.xlabel('Voltage (V)', fontsize=14)
plt.ylabel('Counts', fontsize=14)
plt.title('Counts vs Operating Voltage', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.show()

# summing the total counts from the cosmic ray data
total_cosmic_ray = df_cosmic_ray['counts'].sum()

# for time interval the cosmic ray is
cosmic_ray_per_second = total_cosmic_ray / 90
cosmic_ray_per_second_err = np.sqrt(total_cosmic_ray) / 90

print(f"Total cosmic ray counts: {total_cosmic_ray}")
print(f"Cosmic ray counts per second: {cosmic_ray_per_second:.4f} ± {cosmic_ray_per_second_err:.4f}")